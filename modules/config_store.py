from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(os.path.expanduser("~/.hermes/llama-launcher.json"))

# llama.cpp 디렉터리에서 실행할 수도 있고, 절대경로로도 쓸 수 있게 후보를 넉넉히 둔다.
LLAMA_SERVER_CANDIDATES = [
    "./build-rocm/bin/llama-server",
    "./build/bin/llama-server",
    "~/src/llama.cpp/build-rocm/bin/llama-server",
    "~/src/llama.cpp/build/bin/llama-server",
    "llama-server",
]


def _run_capture(args: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def is_executable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_default_llama_bin() -> str:
    for candidate in LLAMA_SERVER_CANDIDATES:
        if candidate == "llama-server":
            found = shutil.which("llama-server")
            if found:
                return found
            continue

        expanded = expand_path(candidate)
        if is_executable_file(expanded):
            return expanded

        # 현재 작업 디렉터리 기준 상대 경로도 검사
        cwd_candidate = os.path.abspath(candidate)
        if is_executable_file(cwd_candidate):
            return cwd_candidate

    # 못 찾으면 사용자가 설정에서 고치게 기본값만 반환
    return expand_path("~/src/llama.cpp/build-rocm/bin/llama-server")


def detect_tailscale_ip() -> str | None:
    out = _run_capture(["tailscale", "ip", "-4"])
    if not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line and line.count(".") == 3:
            return line
    return None


def default_host() -> str:
    return detect_tailscale_ip() or "127.0.0.1"


def default_config() -> dict[str, Any]:
    return {
        "ctx_size": 95000,
        "host": default_host(),
        "port": 8080,
        "last_model": None,
        "llama_bin": find_default_llama_bin(),

        # Qwen / reasoning 계열 모델이 reasoning_content만 뱉는 문제를 막기 위한 기본값.
        "jinja": True,
        "alias_by_file": True,
        "reasoning": "off",            # off | auto | on
        "reasoning_budget": 0,         # 0 = 즉시 thinking 종료
        "enable_thinking": False,      # chat_template_kwargs용
        "extra_args": [],
    }


def load_config() -> dict[str, Any]:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open() as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                cfg.update(saved)
        except Exception as e:
            print(f"  ⚠️  설정 파일 읽기 실패: {e}")
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def normalize_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return value.split()
    return []
