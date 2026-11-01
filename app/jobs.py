"""Background job runner for long tasks started from the UI (corpus building).

One job at a time; each step is a subprocess of the platform's own Python so
memory is released afterwards.  Progress is streamed to data/jobs/<name>.log
and exposed by /api/admin/corpus/status.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import BASE_DIR, DATA_DIR

JOB_DIR = DATA_DIR / "jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)

KEYS_FILE = DATA_DIR / "api_keys.env"
KEY_NAMES = ("HARVARD_API_KEY", "SMITHSONIAN_API_KEY", "EUROPEANA_API_KEY")


def load_keys() -> dict[str, str]:
    """Museum API keys saved from the admin page (data/api_keys.env, KEY=value per line)."""
    out: dict[str, str] = {}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in KEY_NAMES and v.strip():
                    out[k.strip()] = v.strip()
    return out


def save_keys(keys: dict[str, str]):
    cur = load_keys()
    for k, v in keys.items():
        if k in KEY_NAMES:
            if v and v.strip():
                cur[k] = v.strip()
            else:
                cur.pop(k, None)
    KEYS_FILE.write_text("".join(f"{k}={v}\n" for k, v in cur.items()), encoding="utf-8")
    try:
        os.chmod(KEYS_FILE, 0o600)
    except OSError:
        pass


_state = {"running": False, "name": None, "started_at": None, "finished_at": None, "ok": None, "steps": [], "current": None}
_lock = threading.Lock()


def _run(name: str, steps: list[tuple[str, list[str]]], on_done=None):
    log_path = JOB_DIR / f"{name}.log"
    env = {**os.environ, **load_keys(), "GLOG_minloglevel": "2", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    ok = True
    with open(log_path, "w", encoding="utf-8") as log:
        for label, cmd in steps:
            with _lock:
                _state["current"] = label
            log.write(f"\n===== {label} =====\n$ {' '.join(cmd)}\n")
            log.flush()
            try:
                p = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=6 * 3600)
                if p.returncode != 0:
                    log.write(f"\n!! step failed with exit code {p.returncode}\n")
                    ok = False
                    break
            except Exception as e:  # pragma: no cover
                log.write(f"\n!! {e}\n")
                ok = False
                break
        log.write("\n===== done =====\n" if ok else "\n===== stopped with errors =====\n")
    if on_done:
        try:
            on_done(ok)
        except Exception:
            pass
    with _lock:
        _state.update(running=False, finished_at=time.time(), ok=ok, current=None)


def start(name: str, steps: list[tuple[str, list[str]]], on_done=None) -> bool:
    with _lock:
        if _state["running"]:
            return False
        _state.update(running=True, name=name, started_at=time.time(), finished_at=None, ok=None,
                      steps=[s[0] for s in steps], current=None)
    threading.Thread(target=_run, args=(name, steps, on_done), daemon=True).start()
    return True


def status(tail_lines: int = 60) -> dict:
    with _lock:
        st = dict(_state)
    log_path = JOB_DIR / f"{st['name']}.log" if st.get("name") else None
    tail = ""
    if log_path and log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-tail_lines:])
        except Exception:
            pass
    st["log_tail"] = tail
    return st


def python() -> str:
    """Interpreter for subprocess steps: the console python.exe next to a pythonw.exe (Windows launcher)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return sys.executable
