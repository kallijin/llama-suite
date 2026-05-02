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

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from modules.config_store import (
    detect_tailscale_ip,
    expand_path,
    load_config,
    normalize_extra_args,
    save_config,
)
from modules.model_scan import get_model_list
from modules.profiles import get_model_profile, load_profiles, save_profiles
from modules.probes import quick_no_think_test, show_status
from modules.runner_tmux import get_running_model, get_running_servers, run_script
from modules.script_builder import generate_script, resolve_ctx_size
from modules.system_info import collect_system_info


# ─── 설정 ──────────────────────────────────────────────

MODELS_DIR = os.environ.get("LLAMA_MODELS_DIR", "/mnt/data_main/downloads/models")
SCRIPTS_DIR = Path(os.path.expanduser("~/.hermes/llama-scripts"))


# ─── 작은 유틸 ─────────────────────────────────────────
def print_header() -> None:
    print("\n" + "=" * 64)
    print("  🦙  LLAMA.CPP 모델 실행기 — Hermes / ROCm 완성판")
    print("=" * 64)


def pause() -> None:
    input("\n  (계속하려면 Enter)")


def safe_script_name(text: str, limit: int = 64) -> str:
    allowed = []
    for c in text[:limit]:
        if c.isalnum() or c in ("-", "_", "."):
            allowed.append(c)
        else:
            allowed.append("_")
    name = "".join(allowed).strip("_")
    return name or "model"


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


def show_system_info() -> None:
    info = collect_system_info(use_cache=True)

    print("\n  ── 시스템 정보 ──")
    print(f"  kernel: {info.kernel or 'unknown'}")
    print(f"  arch: {info.arch or 'unknown'}")
    print(f"  gpu_vendor_guess: {info.gpu_vendor_guess or 'unknown'}")
    print(f"  gpu_devices: {', '.join(info.gpu_devices) if info.gpu_devices else 'unknown'}")
    print(f"  rocm_available: {info.rocm_available}")
    print(f"  rocm_summary: {info.rocm_summary or 'unknown'}")
    print(f"  vulkan_available: {info.vulkan_available}")
    print(f"  vulkan_summary: {info.vulkan_summary or 'unknown'}")
    if info.warnings:
        print("  warnings:")
        for warning in info.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings: none")


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
        print("  [I] 시스템 정보")
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

        if upper == "I":
            show_system_info()
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

        profiles = load_profiles()
        profile = get_model_profile(profiles, model_name, model_path)
        save_profiles(profiles)

        result = get_latest_script(model_name)
        existing_script, existing_name = (result if result else (None, None))
        effective_ctx_size = resolve_ctx_size(model_name, model_path, cfg)

        print(f"\n  📦 모델 : {model_name}")
        print(f"  📄 경로 : {model_path}")
        print(f"  ⚙️  설정 : ctx={effective_ctx_size} (global={cfg['ctx_size']}), {cfg['host']}:{cfg['port']}")
        print(
            f"  🧾 profile: ctx={profile.get('stable_ctx_size') or 'unknown'}, "
            f"backend={profile.get('recommended_backend') or 'unknown'}, "
            f"tool={profile.get('hermes_tool_call', {}).get('status') or 'unknown'}"
        )
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
