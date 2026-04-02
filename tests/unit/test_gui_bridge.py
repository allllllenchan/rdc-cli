from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from rdc.cli import main
from rdc.commands import _helpers as helpers_mod
from rdc.services import session_service
from rdc.session_state import SessionState, load_session, save_session


def _daemon_session(tmp_path: Path, *, bridge_enabled: bool = False) -> SessionState:
    capture = str(tmp_path / "capture.rdc")
    return SessionState(
        capture=capture,
        current_eid=0,
        opened_at="2026-04-01T00:00:00+00:00",
        host="127.0.0.1",
        port=12345,
        token="tok",
        pid=999,
        backend="daemon",
        bridge_enabled=bridge_enabled,
        bridge_dir=str(tmp_path / "bridge"),
        bridge_capture=capture if bridge_enabled else "",
    )


def test_attach_gui_bridge_marks_existing_daemon_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rdc._platform.data_dir", lambda: tmp_path / ".rdc")
    monkeypatch.delenv("RDC_SESSION", raising=False)
    session = _daemon_session(tmp_path)
    save_session(session)
    monkeypatch.setattr(session_service, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr("rdc.session_state.is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        session_service,
        "send_bridge_request",
        lambda *args, **kwargs: {"filename": session.capture, "capture_loaded": True},
    )

    ok, msg = session_service.attach_gui_bridge(tmp_path / "bridge")

    assert ok is True
    assert "attached: gui bridge" in msg
    current = load_session()
    assert current is not None
    assert current.backend == "daemon"
    assert current.bridge_enabled is True
    assert current.bridge_dir == str(tmp_path / "bridge")
    assert current.bridge_capture == session.capture


def test_attach_gui_bridge_rejects_capture_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rdc._platform.data_dir", lambda: tmp_path / ".rdc")
    monkeypatch.delenv("RDC_SESSION", raising=False)
    save_session(_daemon_session(tmp_path))
    monkeypatch.setattr(session_service, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        session_service,
        "send_bridge_request",
        lambda *args, **kwargs: {"filename": str(tmp_path / "other.rdc"), "capture_loaded": True},
    )

    ok, msg = session_service.attach_gui_bridge(tmp_path / "bridge")

    assert ok is False
    assert "capture does not match" in msg


def test_call_routes_only_shader_methods_to_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rdc._platform.data_dir", lambda: tmp_path / ".rdc")
    monkeypatch.delenv("RDC_SESSION", raising=False)
    save_session(_daemon_session(tmp_path, bridge_enabled=True))
    monkeypatch.setattr(session_service, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr("rdc.session_state.is_pid_alive", lambda pid: True)

    bridge_calls: list[str] = []
    daemon_calls: list[str] = []

    def fake_bridge(method: str, params: dict[str, object], **kwargs: object) -> dict[str, object]:
        bridge_calls.append(method)
        if method == "selected_subtree_candidates":
            return {
                "selected_event": 100,
                "selected_action_name": "OceanFogPostProcess",
                "candidates": [
                    {
                        "eid": 101,
                        "depth": 0,
                        "action_name": "MI_OceanFogPostProcess",
                        "shader_name": "MI_OceanFogPostProcess",
                        "is_selected": True,
                        "shader_id": 7,
                    }
                ],
            }
        if method == "shader_encodings":
            return {"encodings": [{"name": "DXBC"}, {"name": "HLSL"}]}
        return {"ok": True}

    def fake_send(host: str, port: int, payload: dict[str, object], **kwargs: object) -> dict[str, object]:
        daemon_calls.append(str(payload.get("method", "")))
        method = payload.get("method", "")
        if method == "ping":
            return {"result": {"ok": True}}
        if method == "status":
            return {"result": {"current_eid": 0}}
        if method == "draws":
            return {"result": {"draws": [{"eid": 1}]}}
        return {"result": {}}

    monkeypatch.setattr(helpers_mod, "send_bridge_request", fake_bridge)
    monkeypatch.setattr(helpers_mod, "send_request", fake_send)

    encodings = helpers_mod.call("shader_encodings", {})
    draws = helpers_mod.call("draws", {})
    subtree = helpers_mod.call("selected_subtree_candidates", {"stage": "ps"})

    assert encodings["encodings"][0]["name"] == "DXBC"
    assert draws["draws"][0]["eid"] == 1
    assert subtree["selected_event"] == 100
    assert bridge_calls == ["shader_encodings", "selected_subtree_candidates"]
    assert "draws" in daemon_calls


def test_status_session_includes_bridge_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rdc._platform.data_dir", lambda: tmp_path / ".rdc")
    monkeypatch.delenv("RDC_SESSION", raising=False)
    save_session(_daemon_session(tmp_path, bridge_enabled=True))
    monkeypatch.setattr(session_service, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr("rdc.session_state.is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        session_service,
        "send_request",
        lambda *args, **kwargs: {"result": {"current_eid": 7}},
    )
    monkeypatch.setattr(
        session_service,
        "send_bridge_request",
        lambda *args, **kwargs: {
            "filename": str(tmp_path / "capture.rdc"),
            "capture_loaded": True,
            "api": "Vulkan",
            "built_shader_count": 2,
            "replacement_count": 1,
            "current_event": 8,
            "selected_event": 9,
            "action_name": "DrawIndexed",
            "bound_shaders": {"ps": {"resource_id": 12, "name": "MI_OceanFogPostProcess"}},
        },
    )

    ok, result = session_service.status_session()

    assert ok is True
    assert isinstance(result, dict)
    assert result["backend"] == "daemon"
    assert result["bridge_attached"] is True
    assert result["bridge_capture"].endswith("capture.rdc")
    assert result["built_shader_count"] == 2
    assert result["replacement_count"] == 1


def test_bridge_attach_cli_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rdc._platform.data_dir", lambda: tmp_path / ".rdc")
    monkeypatch.delenv("RDC_SESSION", raising=False)
    monkeypatch.setattr(
        "rdc.commands.session.attach_gui_bridge",
        lambda bridge_dir=None: (True, f"attached: gui bridge ({bridge_dir})"),
    )

    result = CliRunner().invoke(main, ["bridge", "attach"])

    assert result.exit_code == 0
    assert "attached: gui bridge" in result.output


