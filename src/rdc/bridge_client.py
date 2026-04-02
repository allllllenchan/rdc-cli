"""File-based client for the local RenderDoc shader bridge."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_BRIDGE_DIR = Path(tempfile.gettempdir()) / "renderdoc_shader_bridge"


def default_bridge_dir() -> Path:
    return DEFAULT_BRIDGE_DIR


def _request_file(bridge_dir: Path) -> Path:
    return bridge_dir / "request.json"


def _response_file(bridge_dir: Path) -> Path:
    return bridge_dir / "response.json"


def _lock_file(bridge_dir: Path) -> Path:
    return bridge_dir / "lock"


def send_bridge_request(
    method: str,
    params: dict[str, Any],
    *,
    bridge_dir: str | Path | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send a request to the local shader bridge and return the result payload."""
    bridge_root = Path(bridge_dir) if bridge_dir is not None else default_bridge_dir()
    if not bridge_root.exists():
        raise ValueError(f"bridge directory not found: {bridge_root}")

    request_path = _request_file(bridge_root)
    response_path = _response_file(bridge_root)
    lock_path = _lock_file(bridge_root)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not request_path.exists() and not lock_path.exists():
            break
        time.sleep(0.05)
    else:
        raise ValueError("bridge busy: timed out waiting to send request")

    if response_path.exists():
        response_path.unlink(missing_ok=True)

    request = {"id": 1, "method": method, "params": params}
    bridge_root.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("1", encoding="utf-8")
    try:
        request_path.write_text(json.dumps(request), encoding="utf-8")
    finally:
        lock_path.unlink(missing_ok=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            if "error" in response:
                message = response["error"].get("message", "bridge request failed")
                raise ValueError(str(message))
            return dict(response.get("result", {}))
        time.sleep(0.05)

    raise ValueError("bridge timeout waiting for response")


def bridge_available(bridge_dir: str | Path | None = None, *, timeout: float = 1.0) -> bool:
    """Best-effort availability probe for the shader bridge."""
    try:
        send_bridge_request("ping", {}, bridge_dir=bridge_dir, timeout=timeout)
    except Exception:
        return False
    return True