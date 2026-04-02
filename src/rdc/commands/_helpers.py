"""Shared CLI command helpers for daemon communication."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, NoReturn, cast

import click
from click.shell_completion import CompletionItem

from rdc.bridge_client import send_bridge_request
from rdc.capture_core import CaptureResult
from rdc.daemon_client import send_request, send_request_binary
from rdc.discover import find_renderdoc
from rdc.protocol import _request
from rdc.session_state import SessionState, load_session

__all__ = [
    "require_session",
    "require_renderdoc",
    "call",
    "call_binary",
    "try_call",
    "completion_call",
    "fetch_remote_file",
    "write_capture_to_path",
    "_json_mode",
    "split_session_active",
    "complete_eid",
    "complete_pass_name",
    "complete_pass_identifier",
    "_sort_numeric_like",
    "_emit_error",
    "resolve_shader_target_eid",
]

_BRIDGE_METHODS = {
    "selected_subtree_candidates",
    "shader_encodings",
    "shader_build",
    "shader_replace",
    "shader_restore",
    "shader_restore_all",
}


def _sort_numeric_like(values: set[str] | list[str]) -> list[str]:
    """Sort values with numeric strings first (ascending), then alphabetic."""
    return sorted(values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _json_mode() -> bool:
    """Return True if the current Click context has a JSON output flag set."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    params = ctx.params
    return bool(params.get("use_json"))


def _emit_error(msg: str) -> NoReturn:
    """Emit error message in JSON or text format and exit."""
    if _json_mode():
        click.echo(json.dumps({"error": {"message": msg}}), err=True)
    else:
        click.echo(f"error: {msg}", err=True)
    raise SystemExit(1)


def _match_score(text: str, target: str) -> tuple[int, int]:
    text_lower = text.lower()
    target_lower = target.lower()
    if text_lower == target_lower:
        return (2, len(text_lower))
    if target_lower in text_lower:
        return (1, -len(text_lower))
    return (0, 0)


def _rank_target_matches(
    rows: list[dict[str, Any]], target: str
) -> list[tuple[tuple[int, int, int, int], dict[str, Any]]]:
    ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for row in rows:
        texts = [
            str(row.get("action_name") or ""),
            str(row.get("shader_name") or ""),
            str(row.get("marker") or ""),
            str(row.get("pass") or ""),
        ]
        best = (0, 0)
        for text in texts:
            best = max(best, _match_score(text, target))
        if best[0] == 0:
            continue
        score = (
            best[0],
            1 if bool(row.get("is_selected")) else 0,
            -int(row.get("depth", 9999)),
            best[1],
        )
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _unique_top_match(
    rows: list[tuple[tuple[int, int, int, int], dict[str, Any]]]
) -> dict[str, Any] | None:
    if not rows:
        return None
    top_score = rows[0][0]
    top_rows = [row for score, row in rows if score == top_score]
    if len(top_rows) == 1:
        return top_rows[0]
    return None


def _build_daemon_fallback_candidates(target: str) -> list[dict[str, Any]]:
    draws = try_call("draws", {}) or {}
    events = try_call("events", {}) or {}
    rows: list[dict[str, Any]] = []
    for row in draws.get("draws", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "source": "draws",
                "eid": row.get("eid", 0),
                "action_name": row.get("marker") or row.get("name") or "",
                "marker": row.get("marker") or "",
                "pass": row.get("pass") or "",
                "depth": 9999,
            }
        )
    for row in events.get("events", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "source": "events",
                "eid": row.get("eid", 0),
                "action_name": row.get("name") or "",
                "marker": row.get("name") or "",
                "pass": row.get("type") or "",
                "depth": 9999,
            }
        )
    return [row for _score, row in _rank_target_matches(rows, target)]


def resolve_shader_target_eid(
    target: str,
    stage: str,
    *,
    max_depth: int = 32,
) -> tuple[int | None, str, list[dict[str, Any]]]:
    """Resolve a shader-edit target EID using bridge-first, daemon-fallback search."""
    bridge_rows: list[dict[str, Any]] = []
    ranked_bridge: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    bridge_status = try_call("selected_subtree_candidates", {"stage": stage, "max_depth": max_depth})
    if isinstance(bridge_status, dict):
        bridge_rows = [
            {**row, "source": "bridge"}
            for row in bridge_status.get("candidates", [])
            if isinstance(row, dict)
        ]
        ranked_bridge = _rank_target_matches(bridge_rows, target)
        unique_bridge = _unique_top_match(ranked_bridge)
        if unique_bridge is not None:
            row = unique_bridge
            return int(row.get("eid", 0)), "bridge:selected-subtree", bridge_rows

    fallback_rows = _build_daemon_fallback_candidates(target)
    ranked_fallback = _rank_target_matches(fallback_rows, target)
    unique_fallback = _unique_top_match(ranked_fallback)
    if unique_fallback is not None:
        row = unique_fallback
        return int(row.get("eid", 0)), f"{row.get('source', 'daemon')}:fallback", fallback_rows

    all_rows = [row for _score, row in ranked_bridge] if ranked_bridge else fallback_rows
    if not all_rows:
        return None, "no-match", []
    return None, "ambiguous", all_rows


def require_renderdoc() -> Any:
    """Find and return the renderdoc module, or exit with error."""
    rd = find_renderdoc()
    if rd is None:
        click.echo("error: renderdoc module not found", err=True)
        raise SystemExit(1)
    return rd


def require_session() -> tuple[str, int, str]:
    """Load active session or exit with error.

    Returns:
        Tuple of (host, port, token).
    """
    from rdc.protocol import ping_request
    from rdc.session_state import delete_session, is_pid_alive

    session = load_session()
    if session is None:
        _emit_error("no active session (run 'rdc open' first)")
    pid = getattr(session, "pid", None)
    if isinstance(pid, int) and pid <= 0:
        try:
            ping = ping_request(session.token)
            resp = send_request(session.host, session.port, ping, timeout=2.0)
            if resp.get("result", {}).get("ok") is True:
                return session.host, session.port, session.token
        except Exception:  # noqa: BLE001
            pass
        delete_session()
        _emit_error("stale session cleaned (daemon died); run 'rdc open' to restart")
    if isinstance(pid, int) and not is_pid_alive(pid):
        delete_session()
        _emit_error("stale session cleaned (daemon died); run 'rdc open' to restart")
    return session.host, session.port, session.token


def call(method: str, params: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    """Send a JSON-RPC request to the daemon and return the result.

    Args:
        method: The JSON-RPC method name.
        params: Request parameters.
        timeout: Socket timeout in seconds.

    Returns:
        The result dict from the daemon response.

    Raises:
        SystemExit: If the daemon returns an error.
    """
    session = load_session()
    if session is None:
        _emit_error("no active session (run 'rdc open' first)")
    if method in _BRIDGE_METHODS:
        if not bool(getattr(session, "bridge_enabled", False) or getattr(session, "backend", "daemon") == "gui_bridge"):
            _emit_error("gui bridge not attached; run 'rdc bridge attach' first")
        try:
            return send_bridge_request(method, params, bridge_dir=session.bridge_dir, timeout=timeout)
        except ValueError as exc:
            _emit_error(str(exc))
    host, port, token = require_session()
    payload = _request(method, 1, {"_token": token, **params}).to_dict()
    try:
        response = send_request(host, port, payload, timeout=timeout)
    except (OSError, ValueError) as exc:
        _emit_error(f"daemon unreachable: {exc}")
    if "error" in response:
        _emit_error(response["error"]["message"])
    return cast(dict[str, Any], response["result"])


def try_call(method: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Send a JSON-RPC request, returning None on failure.

    Unlike call(), this never exits -- failures are silent.
    Use for optional features where partial success is acceptable.
    """
    session = load_session()
    if session is None:
        return None
    if method in _BRIDGE_METHODS:
        if not bool(getattr(session, "bridge_enabled", False) or getattr(session, "backend", "daemon") == "gui_bridge"):
            return None
        try:
            return send_bridge_request(method, params, bridge_dir=session.bridge_dir, timeout=1.0)
        except Exception:
            return None
    try:
        host, port, token = require_session()
    except SystemExit:
        return None
    payload = _request(method, 1, {"_token": token, **params}).to_dict()
    try:
        response = send_request(host, port, payload)
    except (OSError, ValueError):
        return None
    if "error" in response:
        return None
    return cast(dict[str, Any], response.get("result", {}))


def completion_call(method: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Send a request for shell completion, silently returning None on failure."""
    with contextlib.redirect_stderr(io.StringIO()):
        return try_call(method, params)


def call_binary(method: str, params: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    """Send a JSON-RPC request expecting an optional binary payload.

    Returns:
        Tuple of (result_dict, binary_data_or_None).

    Raises:
        SystemExit: If the daemon returns an error or is unreachable.
    """
    session = load_session()
    if session is None:
        _emit_error("no active session (run 'rdc open' first)")
    host, port, token = require_session()
    payload = _request(method, 1, {"_token": token, **params}).to_dict()
    try:
        response, binary = send_request_binary(host, port, payload)
    except (OSError, ValueError) as exc:
        _emit_error(f"daemon unreachable: {exc}")
    if "error" in response:
        _emit_error(response["error"]["message"])
    return cast(dict[str, Any], response["result"]), binary


def fetch_remote_file(path: str) -> bytes:
    """Fetch a file from the daemon machine, transparently handling local/remote.

    Returns:
        Raw file bytes.

    Raises:
        SystemExit: On any error.
    """
    session = load_session()
    pid = getattr(session, "pid", 1) if session else 1
    if pid > 0:
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            click.echo(f"error: {path}: {exc}", err=True)
            raise SystemExit(1) from exc
    result, binary = call_binary("file_read", {"path": path})
    if binary is None:
        click.echo("error: daemon returned no binary data", err=True)
        raise SystemExit(1)
    return binary


def write_capture_to_path(result: CaptureResult, dest: Path) -> CaptureResult:
    """Fetch ``result.path`` and persist it locally.

    ``fetch_remote_file`` errors (SystemExit) propagate; local ``OSError``
    writes are caught and converted into a failed ``CaptureResult``.  On
    success, ``result.path`` points to ``dest`` and ``result.local`` is True.
    """
    if not result.success or not result.path:
        return result
    try:
        data = fetch_remote_file(result.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    except OSError as exc:
        result.success = False
        result.error = f"failed to write capture to {dest}: {exc}"
        result.path = ""
        return result
    result.path = str(dest)
    result.local = True
    return result


def _split_session() -> SessionState | None:
    session = load_session()
    if session is None:
        return None
    if getattr(session, "backend", "daemon") != "daemon":
        return None
    pid = getattr(session, "pid", 1)
    if isinstance(pid, int) and pid == 0:
        return session
    return None


def split_session_active() -> bool:
    """Return True if the session file indicates Split mode (pid == 0).

    This is a routing hint only—the next RPC will still rely on
    ``require_session()`` to ping the daemon and clean up stale sessions.
    """
    return _split_session() is not None


def complete_eid(
    _ctx: click.Context | None,
    _param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Return shell-completion items for event-id arguments.

    Completion is best-effort and must stay silent/fail-safe when session or
    daemon access is unavailable.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            result = try_call("events", {})
            if result is None:
                return []
            rows = result.get("events", [])
            if not isinstance(rows, list):
                return []

            items: list[CompletionItem] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                eid = row.get("eid")
                if not isinstance(eid, int):
                    continue
                value = str(eid)
                if incomplete and not value.startswith(incomplete):
                    continue
                label = row.get("name")
                if isinstance(label, str) and label:
                    items.append(CompletionItem(value, help=label))
                else:
                    items.append(CompletionItem(value))
            return items
        except Exception:  # noqa: BLE001
            return []


def complete_pass_name(
    _ctx: click.Context | None,
    _param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete render pass names from daemon metadata.

    Fail-safe behavior: return an empty list if session/RPC is unavailable.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            result = try_call("passes", {})
            if not isinstance(result, dict):
                return []
            tree = result.get("tree")
            if not isinstance(tree, dict):
                return []
            passes = tree.get("passes", [])
            if not isinstance(passes, list):
                return []

            prefix = incomplete.casefold()
            names: set[str] = set()
            items: list[CompletionItem] = []
            for p in passes:
                if not isinstance(p, dict):
                    continue
                name = p.get("name")
                if not isinstance(name, str):
                    continue
                if prefix and not name.casefold().startswith(prefix):
                    continue
                if name in names:
                    continue
                names.add(name)
                items.append(CompletionItem(name))
            return items
        except Exception:  # noqa: BLE001
            return []


def complete_pass_identifier(
    _ctx: click.Context | None,
    _param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete pass identifier as index or name."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            result = try_call("passes", {})
            if not isinstance(result, dict):
                return []
            tree = result.get("tree", {})
            passes = tree.get("passes", []) if isinstance(tree, dict) else []
            if not isinstance(passes, list):
                return []

            prefix = incomplete.casefold()
            items: list[CompletionItem] = []
            seen_names: set[str] = set()
            for index, p in enumerate(passes):
                if not isinstance(p, dict):
                    continue
                index_text = str(index)
                if index_text.startswith(incomplete):
                    items.append(CompletionItem(index_text))

                name = p.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if prefix and not name.casefold().startswith(prefix):
                    continue
                if name in seen_names:
                    continue
                seen_names.add(name)
                items.append(CompletionItem(name))
            return items
        except Exception:  # noqa: BLE001
            return []
