from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(os.path.expanduser("~/.hermes/llama-launcher.json"))
KV_CACHE_SAFETY_EXTRA_ARGS = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on"]
KV_CACHE_VALUE_OPTIONS = {"--cache-type-k", "--cache-type-v", "--flash-attn"}

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


def has_control_char(text: str) -> bool:
    return any(ord(c) < 32 or ord(c) == 127 for c in text)


def validate_llama_bin(path: Any) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("llama_bin 경로가 비어 있습니다.")
    if has_control_char(path):
        raise ValueError("llama_bin 경로에 control/ESC 문자가 포함되어 있습니다.")

    expanded = expand_path(path)
    if not os.path.isfile(expanded):
        raise ValueError(f"llama_bin 파일이 없습니다: {expanded}")
    if not os.access(expanded, os.X_OK):
        raise ValueError(f"llama_bin 실행 권한이 없습니다: {expanded}")
    return expanded


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
        # 긴 ctx 로컬 실행에서는 KV cache 압축을 기본 안전장치로 둔다.
        "extra_args": KV_CACHE_SAFETY_EXTRA_ARGS.copy(),
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
    cfg["extra_args"] = ensure_kv_cache_safety_args(cfg.get("extra_args"))
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    cfg["llama_bin"] = validate_llama_bin(cfg.get("llama_bin"))
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def normalize_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return dedupe_extra_args([str(x) for x in value if str(x).strip()])
    if isinstance(value, str):
        try:
            return dedupe_extra_args(shlex.split(value))
        except ValueError:
            return dedupe_extra_args(value.split())
    return []


def extra_arg_has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def extra_arg_has_value_option(args: list[str], option: str) -> bool:
    for i, arg in enumerate(args):
        if arg.startswith(f"{option}="):
            return bool(arg.split("=", 1)[1])
        if arg == option:
            return i + 1 < len(args) and not args[i + 1].startswith("-")
    return False


def extra_arg_option_name(arg: str) -> str | None:
    name = arg.split("=", 1)[0]
    if name in KV_CACHE_VALUE_OPTIONS:
        return name
    return None


def dedupe_extra_args(args: list[str]) -> list[str]:
    result: list[str] = []
    seen_safety_options: set[str] = set()
    i = 0
    while i < len(args):
        arg = args[i]
        option = extra_arg_option_name(arg)
        if option in KV_CACHE_VALUE_OPTIONS:
            if option in seen_safety_options:
                i += 2 if arg == option and i + 1 < len(args) and not args[i + 1].startswith("-") else 1
                continue
            seen_safety_options.add(option)
            result.append(arg)
            if arg == option and i + 1 < len(args) and not args[i + 1].startswith("-"):
                result.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue
        result.append(arg)
        i += 1
    return result


def ensure_kv_cache_safety_args(value: Any) -> list[str]:
    args = normalize_extra_args(value)
    defaults: list[str] = []
    if not extra_arg_has_value_option(args, "--cache-type-k"):
        defaults.extend(["--cache-type-k", "q8_0"])
    if not extra_arg_has_value_option(args, "--cache-type-v"):
        defaults.extend(["--cache-type-v", "q8_0"])
    if not extra_arg_has_value_option(args, "--flash-attn"):
        defaults.extend(["--flash-attn", "on"])
    return dedupe_extra_args(defaults + args)
