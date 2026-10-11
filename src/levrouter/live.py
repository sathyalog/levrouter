"""Live, in-memory usage over a local Unix socket (no network port, no web route).

Each app process that enabled levrouter listens on <run dir>/<pid>.sock, in a directory only
its user can open (0700). `levrouter usage --live` connects to every socket there, asks each
process for its in-memory last-24h report, and merges the answers. Numbers live only in
memory, so they reset when the process restarts; the rolling hourly files cover that case.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import socket
import socketserver
import tempfile
import threading
import time
from pathlib import Path

from . import usage_store as usage

log = logging.getLogger("levrouter")

MAX_REQUEST = 4096
# macOS limits Unix socket paths to 104 bytes (Linux 108); leave room for "<pid>.sock".
_MAX_DIR_LEN = 80

SUPPORTED = hasattr(socket, "AF_UNIX")


def run_dir(home: str | Path | None = None) -> Path:
    """Socket directory for a levrouter home. Falls back to a short, deterministic temp path
    when the home path is too long for a socket address."""
    base = usage.home_dir(str(home) if home else None) / "run"
    if len(str(base)) <= _MAX_DIR_LEN:
        return base
    digest = hashlib.sha256(str(base).encode()).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / f"levrouter-{os.getuid() if hasattr(os, 'getuid') else 'u'}-{digest}"


def _answer(store: usage.MemoryStore, request: dict) -> dict:
    if request.get("op") != "usage":
        return {"error": "unknown op"}
    since, note = usage.parse_since(request.get("since"), store.window_s / 3600)
    records = store.query(since)
    report = usage.aggregate(records, by=request.get("by", "model"), user=request.get("user"), task=request.get("task"))
    recent = request.get("recent") or 0
    return {
        "pid": os.getpid(),
        "started_at": store.started_at,
        "records": len(records),
        "note": note,
        "report": report,
        "recent": [r.__dict__ for r in sorted(records, key=lambda r: r.ts)[-int(recent):]] if recent else [],
    }


class LiveServer:
    def __init__(self, store: usage.MemoryStore, directory: Path):
        self.store = store
        self.dir = directory
        self.path = directory / f"{os.getpid()}.sock"
        self._server: socketserver.BaseServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not SUPPORTED:
            log.info("levrouter: this platform has no Unix sockets; `levrouter usage --live` is unavailable")
            return False
        usage._private_dir(self.dir)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        store = self.store

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    raw = self.rfile.readline(MAX_REQUEST + 1)
                    if len(raw) > MAX_REQUEST:
                        reply = {"error": "request too large"}
                    else:
                        reply = _answer(store, json.loads(raw or b"{}"))
                except (ValueError, TypeError) as exc:
                    reply = {"error": f"bad request: {type(exc).__name__}"}
                self.wfile.write((json.dumps(reply, default=str) + "\n").encode())

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        try:
            self._server = Server(str(self.path), Handler)
        except OSError as exc:
            log.info("levrouter: live usage socket not started (%s)", exc)
            return False
        os.chmod(self.path, 0o600)
        # Short poll interval so stop() (disable(), process exit) returns promptly.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, name="levrouter-live", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def query_all(directory: Path, request: dict, timeout: float = 2.0) -> list[dict]:
    """Ask every live levrouter process in `directory`. Removes stale socket files."""
    if not SUPPORTED or not directory.is_dir():
        return []
    answers = []
    for sock_path in sorted(directory.glob("*.sock")):
        try:
            pid = int(sock_path.stem)
        except ValueError:
            continue
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect(str(sock_path))
            s.sendall((json.dumps(request) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            answers.append(json.loads(buf))
        except (OSError, ValueError) as exc:
            if not _pid_alive(pid):
                # Leftover from a process that exited without cleanup (crash, kill -9).
                try:
                    sock_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                answers.append({"pid": pid, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            s.close()
    return answers


def wait_for_socket(path: Path, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False
