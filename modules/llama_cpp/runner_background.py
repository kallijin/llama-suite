from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def run_capture(args: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def get_running_servers() -> list[str]:
    """현재 실행 중인 llama-server 명령줄 목록."""
    out = run_capture(["ps", "-eo", "pid=,args="])
    lines: list[str] = []
    for line in out.splitlines():
        if ("llama-server" in line or "llama_server" in line) and "llama-launcher.py" not in line:
            lines.append(line.strip())
    return lines


def parse_model_path_from_cmdline(cmdline: str) -> str | None:
    """ps 명령줄에서 -m/--model 뒤의 모델 경로를 대충 안전하게 추출."""
    try:
        parts = shlex.split(cmdline)
    except ValueError:
        parts = cmdline.split()

    for flag in ("-m", "--model"):
        if flag in parts:
            idx = parts.index(flag)
            if idx + 1 < len(parts):
                return parts[idx + 1]

    # 혹시 '-m/path' 같은 비표준 형태로 들어간 경우의 약한 fallback
    for p in parts:
        if p.startswith("-m") and len(p) > 2:
            return p[2:]
    return None


def get_running_model() -> str | None:
    for line in get_running_servers():
        model_path = parse_model_path_from_cmdline(line)
        if model_path:
            p = Path(model_path)
            # 기존 UI와 맞게 폴더명을 우선 보여준다.
            return p.parent.name or p.name
    return None


def kill_running_servers() -> None:
    lines = get_running_servers()
    if not lines:
        return

    pids: list[int] = []
    for line in lines:
        try:
            pids.append(int(line.split(maxsplit=1)[0]))
        except Exception:
            pass

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    time.sleep(1.5)

    for pid in pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def run_script(script_path: str, model_name: str | None = None) -> None:
    running = get_running_model()
    if running:
        confirm_kill = input(f"  🔴 '{running}' 실행 중. 교체할까요? (y/n) > ").strip().lower()
        if confirm_kill == "y":
            kill_running_servers()
            print("  🛑 기존 서버 종료됨")
        else:
            return

    log_path = str(script_path) + ".log"
    with open(log_path, "ab") as log:
        process = subprocess.Popen(["bash", script_path], stdout=log, stderr=log, start_new_session=True)
    print(f"  ✅ 백그라운드 실행됨: {script_path}")
    print(f"     PID: {process.pid}")
    print(f"     로그: {log_path}")
