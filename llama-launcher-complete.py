#!/usr/bin/env python3
"""
🦙 LLAMA.CPP 모델 실행기 — Hermes / llama.cpp ROCm용 완성판

주요 기능
- MODELS_DIR 아래의 .gguf 모델 자동 탐색
- 설정 선택 → ~/.hermes/llama-scripts/ 아래 실행 .sh 생성
- 기존 llama-server 교체 실행
- Tailscale IP 자동 감지 지원
- Qwen thinking/reasoning 출력 차단 옵션 기본 적용
- OpenAI-compatible /v1 모델 ID 안정화를 위한 --alias 자동 적용
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.model_scan import get_model_list
from modules.probes import quick_no_think_test, show_status


# ─── 설정 ──────────────────────────────────────────────

MODELS_DIR = os.environ.get("LLAMA_MODELS_DIR", "/mnt/data_main/downloads/models")
CONFIG_PATH = Path(os.path.expanduser("~/.hermes/llama-launcher.json"))
SCRIPTS_DIR = Path(os.path.expanduser("~/.hermes/llama-scripts"))

# llama.cpp 디렉터리에서 실행할 수도 있고, 절대경로로도 쓸 수 있게 후보를 넉넉히 둔다.
LLAMA_SERVER_CANDIDATES = [
    "./build-rocm/bin/llama-server",
    "./build/bin/llama-server",
    "~/src/llama.cpp/build-rocm/bin/llama-server",
    "~/src/llama.cpp/build/bin/llama-server",
    "llama-server",
]


# ─── 작은 유틸 ─────────────────────────────────────────

def run_capture(args: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


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
    out = run_capture(["tailscale", "ip", "-4"])
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


def print_header() -> None:
    print("\n" + "=" * 64)
    print("  🦙  LLAMA.CPP 모델 실행기 — Hermes / ROCm 완성판")
    print("=" * 64)


def pause() -> None:
    input("\n  (계속하려면 Enter)")


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


def safe_script_name(text: str, limit: int = 64) -> str:
    allowed = []
    for c in text[:limit]:
        if c.isalnum() or c in ("-", "_", "."):
            allowed.append(c)
        else:
            allowed.append("_")
    name = "".join(allowed).strip("_")
    return name or "model"


# ─── 서버 상태 감지 ─────────────────────────────────────

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
    for i, p in enumerate(parts):
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


# ─── 설정 변경 ─────────────────────────────────────────

def change_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    print("\n  ── 설정 변경 ──")
    print(
        f"  현재: ctx={cfg['ctx_size']}, "
        f"host={cfg['host']}:{cfg['port']}, "
        f"reasoning={cfg.get('reasoning', 'off')}, "
        f"budget={cfg.get('reasoning_budget', 0)}"
    )
    print(f"  llama-server: {cfg.get('llama_bin')}\n")

    presets = [
        ("2k", 2000),
        ("4k", 4000),
        ("8k", 8000),
        ("16k", 16000),
        ("32k", 32000),
        ("62k", 62000),
        ("64k", 65536),
        ("95k", 95000),
        ("96k", 98304),
        ("120k", 120000),
    ]

    print("  Context size 프리셋:")
    for i, (label, value) in enumerate(presets, 1):
        marker = " ◀ 현재" if int(cfg["ctx_size"]) == value else ""
        print(f"    [{i:>2}] {label:>5s} = {value}{marker}")
    print(f"    [C] 커스텀 입력 (현재: {cfg['ctx_size']})")

    choice = input("  선택 > ").strip().upper()
    if choice == "C":
        val = input(f"  숫자 입력 [{cfg['ctx_size']}] > ").strip()
        if val:
            cfg["ctx_size"] = int(val)
    elif choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                cfg["ctx_size"] = presets[idx][1]
        except ValueError:
            pass

    tailscale_ip = detect_tailscale_ip()
    host_hint = f" / T={tailscale_ip}" if tailscale_ip else ""
    val = input(f"\n  Host [{cfg['host']}]{host_hint} > ").strip()
    if val:
        if val.upper() == "T" and tailscale_ip:
            cfg["host"] = tailscale_ip
        else:
            cfg["host"] = val

    val = input(f"  Port [{cfg['port']}] > ").strip()
    if val:
        cfg["port"] = int(val)

    val = input(f"  llama-server 경로 [{cfg.get('llama_bin')}] > ").strip()
    if val:
        cfg["llama_bin"] = expand_path(val)

    print("\n  ── Thinking / Reasoning 설정 ──")
    print("  Hermes용 Qwen thinking-only 응답을 막으려면 기본값 그대로 두는 걸 추천.")
    val = input(f"  reasoning 모드 off/auto/on [{cfg.get('reasoning', 'off')}] > ").strip().lower()
    if val in {"off", "auto", "on"}:
        cfg["reasoning"] = val

    val = input(f"  reasoning budget [{cfg.get('reasoning_budget', 0)}] > ").strip()
    if val:
        cfg["reasoning_budget"] = int(val)

    current_thinking = bool(cfg.get("enable_thinking", False))
    val = input(f"  chat_template enable_thinking true/false [{str(current_thinking).lower()}] > ").strip().lower()
    if val in {"true", "t", "yes", "y", "1"}:
        cfg["enable_thinking"] = True
    elif val in {"false", "f", "no", "n", "0"}:
        cfg["enable_thinking"] = False

    current_jinja = bool(cfg.get("jinja", True))
    val = input(f"  --jinja 사용 y/n [{'y' if current_jinja else 'n'}] > ").strip().lower()
    if val in {"y", "yes", "1", "true"}:
        cfg["jinja"] = True
    elif val in {"n", "no", "0", "false"}:
        cfg["jinja"] = False

    current_alias = bool(cfg.get("alias_by_file", True))
    val = input(f"  --alias 를 GGUF 파일명으로 고정 y/n [{'y' if current_alias else 'n'}] > ").strip().lower()
    if val in {"y", "yes", "1", "true"}:
        cfg["alias_by_file"] = True
    elif val in {"n", "no", "0", "false"}:
        cfg["alias_by_file"] = False

    extra_str = " ".join(shlex.quote(x) for x in normalize_extra_args(cfg.get("extra_args", [])))
    val = input(f"  추가 llama-server 인자 [{extra_str}] > ").strip()
    if val:
        cfg["extra_args"] = normalize_extra_args(val)

    save_config(cfg)
    print("  ✅ 설정 저장됨!")
    return cfg


# ─── 스크립트 관리 ─────────────────────────────────────

def list_scripts() -> list[tuple[str, str]]:
    scripts: list[tuple[str, str]] = []
    if not SCRIPTS_DIR.is_dir():
        return scripts
    files = sorted(
        SCRIPTS_DIR.glob("*.sh"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in files:
        scripts.append((p.stem, str(p)))
    return scripts


def get_latest_script(model_name: str) -> tuple[str, str] | None:
    if not SCRIPTS_DIR.is_dir():
        return None

    safe_part = safe_script_name(model_name[:40])
    candidates = [
        p for p in SCRIPTS_DIR.glob("*.sh")
        if model_name in p.name or safe_part in p.name
    ]

    if not candidates:
        return None

    newest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(newest), newest.name


def read_script_field(path: str, field: str) -> str | None:
    prefix = f"{field}="
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(prefix):
                    raw = line[len(prefix):].strip()
                    return raw.strip("'").strip('"')
    except Exception:
        pass
    return None


def script_is_modern(path: str) -> bool:
    try:
        text = Path(path).read_text()
    except Exception:
        return False
    required = [
        "MODEL_PATH=",
        "--reasoning",
        "--reasoning-budget",
        "--chat-template-kwargs",
        "enable_thinking",
        "--alias",
    ]
    return all(x in text for x in required)


def show_scripts() -> None:
    scripts = list_scripts()
    if not scripts:
        print("\n  📜 저장된 스크립트가 없습니다.\n")
        return

    running_lines = "\n".join(get_running_servers())

    print(f"\n  📜 저장된 스크립트 ({len(scripts)}개)\n")
    for i, (name, path) in enumerate(scripts, 1):
        model_info = read_script_field(path, "MODEL") or name
        model_path = read_script_field(path, "MODEL_PATH") or read_script_field(path, "PATH") or ""
        running_marker = " 🔴 실행 중" if model_path and model_path in running_lines else ""
        modern_marker = "" if script_is_modern(path) else " ⚠️ old"
        print(f"    [{i}] {model_info}{running_marker}{modern_marker}")
    print()


def delete_script(index: int) -> None:
    scripts = list_scripts()
    if not (0 <= index < len(scripts)):
        print("  ⚠️  유효하지 않은 번호입니다.")
        return

    name, path = scripts[index]
    try:
        os.remove(path)
        pid_file = path + ".pid"
        if os.path.exists(pid_file):
            os.remove(pid_file)
        print(f"  🗑️  '{name}' 삭제됨!")
    except OSError as e:
        print(f"  ⚠️  삭제 실패: {e}")


def manage_scripts() -> None:
    while True:
        show_scripts()
        scripts = list_scripts()
        if not scripts:
            break

        print("  [번호] 삭제")
        print("  [A] 모두 삭제")
        print("  [B] 돌아가기\n")

        choice = input("  선택 > ").strip().upper()

        if choice == "B":
            break

        if choice == "A":
            confirm = input("  정말 모두 삭제할까요? (y/n) > ").strip().lower()
            if confirm == "y":
                for _, path in scripts:
                    try:
                        os.remove(path)
                        pid_file = path + ".pid"
                        if os.path.exists(pid_file):
                            os.remove(pid_file)
                    except OSError:
                        pass
                print("  🗑️  모두 삭제됨!")
            continue

        if choice.isdigit():
            delete_script(int(choice) - 1)


# ─── 스크립트 생성 및 실행 ──────────────────────────────

def resolve_llama_bin(cfg: dict[str, Any]) -> str:
    bin_path = str(cfg.get("llama_bin") or find_default_llama_bin())
    if not os.path.isabs(os.path.expanduser(bin_path)):
        candidate = os.path.abspath(os.path.join(os.getcwd(), bin_path))
        if is_executable_file(candidate):
            return candidate

    expanded = expand_path(bin_path)
    if is_executable_file(expanded):
        return expanded

    fallback = find_default_llama_bin()
    return fallback


def generate_script(model_name: str, model_path: str, cfg: dict[str, Any]) -> tuple[str, str]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = safe_script_name(model_name)
    script_name = f"{safe_name}_{ts}.sh"
    script_path = SCRIPTS_DIR / script_name

    bin_path = resolve_llama_bin(cfg)
    model_id = Path(model_path).name if cfg.get("alias_by_file", True) else model_name
    extra_args = normalize_extra_args(cfg.get("extra_args", []))
    extra_args_shell = " ".join(shlex.quote(x) for x in extra_args)

    enable_thinking = "true" if bool(cfg.get("enable_thinking", False)) else "false"
    reasoning = str(cfg.get("reasoning", "off"))
    reasoning_budget = int(cfg.get("reasoning_budget", 0))

    cmd_lines = [
        '    -m "$MODEL_PATH" \\',
        '    --host "$HOST" \\',
        '    --port "$PORT" \\',
        '    --ctx-size "$CTX_SIZE" \\',
    ]

    if cfg.get("jinja", True):
        cmd_lines.append('    --jinja \\')

    if cfg.get("alias_by_file", True):
        cmd_lines.append('    --alias "$MODEL_ID" \\')

    # Qwen thinking-only 응답 방지용. reasoning=auto/on으로 바꾸면 사용자가 의도한 대로 따라간다.
    if reasoning in {"off", "auto", "on"}:
        cmd_lines.append('    --reasoning "$REASONING_MODE" \\')
    cmd_lines.append('    --reasoning-budget "$REASONING_BUDGET" \\')
    cmd_lines.append('    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \\')
    cmd_lines.append('    "${EXTRA_ARGS[@]}"')

    cmd_block = "\n".join(cmd_lines)

    script_content = f"""#!/usr/bin/env bash
# 🦙 LLAMA.CPP 실행 스크립트
# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 생성기: llama-launcher.py

set -euo pipefail

SERVER_BIN={shlex.quote(bin_path)}
MODEL={shlex.quote(model_name)}
MODEL_ID={shlex.quote(model_id)}
MODEL_PATH={shlex.quote(model_path)}
HOST={shlex.quote(str(cfg['host']))}
PORT={int(cfg['port'])}
CTX_SIZE={int(cfg['ctx_size'])}
REASONING_MODE={shlex.quote(reasoning)}
REASONING_BUDGET={reasoning_budget}
CHAT_TEMPLATE_KWARGS='{{"enable_thinking":{enable_thinking}}}'
EXTRA_ARGS=({extra_args_shell})

echo "🚀 Starting $MODEL"
echo "   model id : $MODEL_ID"
echo "   endpoint : http://$HOST:$PORT/v1"
echo "   ctx      : $CTX_SIZE"
echo "   reasoning: $REASONING_MODE, budget=$REASONING_BUDGET, template=$CHAT_TEMPLATE_KWARGS"
echo

exec "$SERVER_BIN" \\
{cmd_block}
"""

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_name, str(script_path)


def tmux_session_name(model_name: str) -> str:
    return "llama_" + safe_script_name(model_name, limit=32)


def run_script(script_path: str, model_name: str | None = None) -> None:
    running = get_running_model()
    if running:
        confirm_kill = input(f"  🔴 '{running}' 실행 중. 교체할까요? (y/n) > ").strip().lower()
        if confirm_kill == "y":
            kill_running_servers()
            print("  🛑 기존 서버 종료됨")
        else:
            return

    session = tmux_session_name(model_name or Path(script_path).stem)

    if command_exists("tmux"):
        # 세션 이름 충돌 방지
        existing = run_capture(["tmux", "list-sessions", "-F", "#{session_name}"])
        if session in existing.splitlines():
            session = f"{session}_{datetime.now().strftime('%H%M%S')}"

        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, "bash", script_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  ✅ tmux 세션에서 실행됨: {session}")
            print(f"     접속: tmux attach -t {session}")
            return
        print(f"  ⚠️  tmux 실행 실패: {result.stderr.strip()}")

    # tmux가 없으면 백그라운드 실행 + 로그 파일
    log_path = str(script_path) + ".log"
    with open(log_path, "ab") as log:
        subprocess.Popen(["bash", script_path], stdout=log, stderr=log)
    print(f"  ✅ 백그라운드 실행됨: {script_path}")
    print(f"     로그: {log_path}")


def run_existing_script(script_path: str) -> None:
    model_name = read_script_field(script_path, "MODEL") or Path(script_path).stem
    if not script_is_modern(script_path):
        print("  ⚠️  이 스크립트는 old 형식입니다.")
        print("     --reasoning off / --reasoning-budget 0 / enable_thinking=false 가 없을 수 있습니다.")
        print("     Qwen thinking-only 문제가 재발할 수 있으니 새 스크립트 생성을 추천합니다.")
        confirm = input("  그래도 실행할까요? (y/n) > ").strip().lower()
        if confirm != "y":
            return
    run_script(script_path, model_name=model_name)


# ─── 메인 루프 ─────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    save_config(cfg)  # 새 필드가 생겼으면 즉시 반영

    models = get_model_list(MODELS_DIR)
    if not models:
        print(f"\n⚠️  {MODELS_DIR} 에서 GGUF 파일을 찾을 수 없습니다.")
        sys.exit(1)

    while True:
        print_header()
        print(f"  모델 디렉터리: {MODELS_DIR}")
        print(f"  endpoint 설정: http://{cfg['host']}:{cfg['port']}/v1")
        print(f"  ctx={cfg['ctx_size']}, reasoning={cfg.get('reasoning')}, budget={cfg.get('reasoning_budget')}")
        print(f"  모델 목록 ({len(models)}개)\n")

        running = get_running_model()
        if running:
            print(f"  🔴 실행 중: {running}\n")

        numbered = list(enumerate(models.items(), 1))
        for i, (name, _path) in numbered:
            marker = ""
            if name == running:
                marker = " ◀ 실행 중"
            elif name == cfg.get("last_model"):
                marker = " ◀ 최근 사용"
            print(f"  [{i:>2}] {name}{marker}")

        existing_scripts = list_scripts()
        script_info = f" ({len(existing_scripts)}개)" if existing_scripts else ""

        print("\n  [A] 설정 변경")
        print(f"  [S] 스크립트 관리{script_info}")
        print("  [H] 서버 상태 확인")
        print("  [T] no-thinking 채팅 테스트")
        print("  [R] 모델 목록 새로고침")
        print("  [Q] 종료\n")

        choice = input("  선택 > ").strip()

        if not choice:
            continue

        upper = choice.upper()

        if upper == "Q":
            print("\n👋 안녕!\n")
            break

        if upper == "A":
            cfg = change_settings(cfg)
            continue

        if upper == "S":
            manage_scripts()
            continue

        if upper == "H":
            show_status(cfg, get_running_servers())
            pause()
            continue

        if upper == "T":
            quick_no_think_test(cfg)
            pause()
            continue

        if upper == "R":
            models = get_model_list(MODELS_DIR)
            print("  ✅ 목록 새로고침!")
            pause()
            continue

        # ── 번호 또는 이름 검색 ──
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(numbered):
                model_name, model_path = numbered[idx][1]
            else:
                print("\n⚠️  유효하지 않은 번호입니다.")
                pause()
                continue
        except ValueError:
            matches = [m for m in models if choice.lower() in m.lower()]
            if len(matches) == 1:
                model_name = matches[0]
                model_path = models[model_name]
            elif len(matches) > 1:
                print("\n  ⚠️  여러 개 일치:")
                for m in matches[:10]:
                    print(f"    - {m}")
                if len(matches) > 10:
                    print(f"    ... 외 {len(matches) - 10}개")
                pause()
                continue
            else:
                print("\n⚠️  일치하는 모델이 없습니다.")
                pause()
                continue

        cfg["last_model"] = model_name
        save_config(cfg)

        result = get_latest_script(model_name)
        existing_script, existing_name = (result if result else (None, None))

        print(f"\n  📦 모델 : {model_name}")
        print(f"  📄 경로 : {model_path}")
        print(f"  ⚙️  설정 : ctx={cfg['ctx_size']}, {cfg['host']}:{cfg['port']}")
        print(
            f"  🧠 reasoning={cfg.get('reasoning')}, "
            f"budget={cfg.get('reasoning_budget')}, "
            f"enable_thinking={cfg.get('enable_thinking')}"
        )

        if existing_script:
            status = "modern" if script_is_modern(existing_script) else "old"
            print(f"  📝 기존 스크립트: {existing_name} ({status})")
            default_choice = "E" if status == "modern" else "N"
            choice2 = input(
                f"\n  [E] 기존 스크립트 실행 / [N] 새 스크립트 만들기 [{default_choice}] > "
            ).strip().upper() or default_choice
            if choice2 == "E":
                run_existing_script(existing_script)
                pause()
                continue
        else:
            choice2 = "N"

        if choice2 == "N":
            confirm = input("\n  새 스크립트 생성하고 실행할까요? (y/n) > ").strip().lower()
            if confirm != "y":
                continue

            script_name, script_path = generate_script(model_name, model_path, cfg)
            print(f"  📝 스크립트 생성됨: {script_path}")

            run_confirm = input("  지금 실행할까요? (y/n) > ").strip().lower()
            if run_confirm == "y":
                run_script(script_path, model_name=model_name)

            pause()


if __name__ == "__main__":
    main()
