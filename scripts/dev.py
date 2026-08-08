"""Run the FastAPI backend and Observable frontend as one local application."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
API_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:3000"


def backend_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def frontend_ready() -> bool:
    try:
        with urllib.request.urlopen(FRONTEND_URL, timeout=1) as response:
            return response.status == 200 and b"RiskDesk" in response.read(32_768)
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    env = os.environ.copy()
    env.setdefault("RISKDESK_API_URL", API_URL)
    env.setdefault("ALLOWED_ORIGINS", FRONTEND_URL)
    processes: list[subprocess.Popen] = []
    owns_backend = not backend_ready()

    if owns_backend:
        backend = subprocess.Popen(
            [str(ROOT / ".venv/bin/uvicorn"), "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
            env=env,
        )
        processes.append(backend)
        for _ in range(60):
            if backend.poll() is not None:
                return backend.returncode or 1
            if backend_ready():
                break
            time.sleep(0.25)
        else:
            print("Backend did not become healthy within 15 seconds.", file=sys.stderr)
            backend.terminate()
            return 1
    else:
        print(f"Using backend already listening at {API_URL}.")

    if frontend_ready():
        print(f"Using frontend already listening at {FRONTEND_URL}.")
        print(f"RiskDesk frontend: {FRONTEND_URL}")
        print(f"RiskDesk backend:  {API_URL}")
        return 0

    frontend = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host=127.0.0.1", "--port=3000", "--no-open"],
        cwd=FRONTEND,
        env=env,
    )
    processes.append(frontend)
    print(f"RiskDesk frontend: {FRONTEND_URL}")
    print(f"RiskDesk backend:  {API_URL}")

    def stop(_signum=None, _frame=None):
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return frontend.wait()
    finally:
        stop()
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
