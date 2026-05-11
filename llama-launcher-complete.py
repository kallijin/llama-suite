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

import copy
import json
import os
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.config_store import (
    detect_tailscale_ip,
    expand_path,
    get_option_value,
    load_config,
    normalize_extra_args,
    save_config,
    set_option_value,
)
from modules.hermes_integration import build_hermes_vllm_sync_plan, format_hermes_vllm_sync_plan, write_hermes_vllm_sync_plan
from modules.hermes_runner import build_hermes_vllm_smoke_plan, build_hermes_vllm_tool_agent_smoke_plan, run_hermes_vllm_smoke, run_hermes_vllm_tool_agent_smoke
from modules.model_scan import get_model_list
from modules.profiles import default_model_profile, get_model_profile, load_profiles, save_profiles
from modules.probes import quick_no_think_test, show_status
from modules.runner_tmux import get_running_model, get_running_servers, run_script
from modules.script_builder import command_preview, generate_script, parse_generated_script, resolve_ctx_size
from modules.system_info import collect_system_info
from modules.vllm_api_probe import run_vllm_api_smoke
from modules.vllm_doctor import format_vllm_doctor_report, run_vllm_doctor
from modules.vllm_profile_store import backup_vllm_profile_draft, default_vllm_profile_path, delete_vllm_profile_draft, format_vllm_profile_draft_json, list_vllm_profile_drafts, load_selected_vllm_profile_draft, load_vllm_profile_draft, load_vllm_profile_json_file, save_selected_vllm_profile_id, save_verified_gemma4_26b_awq_beta_profile, save_vllm_profile_draft, validate_vllm_profile_json_file
from modules.vllm_profiles import add_vllm_extra_arg, build_vllm_command, builtin_vllm_profile_presets, common_vllm_extra_arg_options, default_vllm_profile, editable_vllm_profile_field_specs, editable_vllm_profile_fields, format_vllm_profile_report, future_launch_preset_id, host_guidance_lines, large_model_guidance_lines, launch_confirmation_guidance_lines, remove_vllm_extra_arg_token, run_vllm_preflight, tokenize_vllm_extra_args, update_vllm_profile_field, validate_vllm_profile, vllm_port_conflict_guidance_lines
from modules.vllm_runner import check_vllm_run_status, latest_vllm_run_record, latest_vllm_run_summary, launch_vllm_profile_once, launch_vllm_smoke_once, read_vllm_run_log, read_vllm_run_record, stop_vllm_run
from modules.vllm_script_builder import build_vllm_script_preview, save_vllm_script


# ─── 설정 ──────────────────────────────────────────────

MODELS_DIR = os.environ.get("LLAMA_MODELS_DIR", "/mnt/data_main/downloads/models")
SCRIPTS_DIR = Path(os.path.expanduser("~/.hermes/llama-scripts"))
LAST_RUN_RECORD_PATH = Path(os.path.expanduser("~/.hermes/llama-suite-last-run.json"))
STRUCTURED_ARG_OPTIONS = {
    "-m",
    "--model",
    "--host",
    "--port",
    "--ctx-size",
    "--alias",
    "--jinja",
    "--reasoning",
    "--reasoning-budget",
    "--chat-template-kwargs",
    "--cache-type-k",
    "--cache-type-v",
    "--flash-attn",
}
KV_PRESETS = ("q8_0", "f16", "q4_0", "q5_0", "q6_0", "tbq3_0", "tbq4_0")
HERMES_CONFIG_CANDIDATES = (
    "~/.hermes/config.yaml",
    "~/.config/hermes/config.yaml",
    "~/Hermes/config.yaml",
)
OPENCLAW_CONFIG_CANDIDATES = (
    "~/.openclaw/config.yaml",
    "~/.config/openclaw/config.yaml",
    "~/OpenClaw/config.yaml",
)
USE_COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None
COLORS = {
    "title": "\033[1;36m",
    "section": "\033[1;34m",
    "ok": "\033[1;32m",
    "warn": "\033[1;33m",
    "reset": "\033[0m",
}


def color(text: str, key: str) -> str:
    if not USE_COLOR:
        return text
    return f"{COLORS.get(key, '')}{text}{COLORS['reset']}"


# ─── 작은 유틸 ─────────────────────────────────────────
def print_header() -> None:
    print("\n" + color("=" * 64, "title"))
    print(color("  🦙  llama-suite local AI engine control", "title"))
    print(color("=" * 64, "title"))


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

SAVE_SETTINGS_ACTION = "[A] 설정 변경 → [W] 현재 설정 저장"

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

    print("  ✅ 임시 작업 설정에 반영됨. 아직 저장되지 않았습니다.")
    return cfg


def settings_menu(cfg: dict[str, Any], draft: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    while True:
        print("\n  ── 설정 변경 ──")
        print("  설정은 먼저 현재 작업 설정에만 반영됩니다.")
        print(f"  저장하려면 {SAVE_SETTINGS_ACTION}을 선택하세요.")
        print()
        print("  [1] 기본 설정 변경")
        print("  [2] 파라미터")
        print("  [W] 현재 설정 저장")
        print("  [R] 작업 화면으로 돌아가기")
        choice = input("  선택 > ").strip().upper()

        if choice == "R":
            return cfg, draft

        if choice == "1":
            before = copy.deepcopy(draft)
            draft = change_settings(draft)
            if draft != before:
                draft["dirty"] = True
                draft["status"] = "설정 변경으로 생긴 임시 작업 설정입니다."
            pause()
            continue

        if choice == "2":
            before = copy.deepcopy(draft)
            draft = edit_parameters(draft)
            if draft != before:
                draft["dirty"] = True
                draft["status"] = "파라미터 변경으로 생긴 임시 작업 설정입니다."
            pause()
            continue

        if choice == "W":
            ok, message, cfg = save_working_draft(cfg, draft)
            print("  " + ("✅ " if ok else "⚠️  ") + message)
            pause()
            continue


def extra_arg_value(cfg: dict[str, Any], option: str) -> str:
    return get_option_value(normalize_extra_args(cfg.get("extra_args", [])), option) or "-"


def parameter_source(cfg: dict[str, Any], key: str) -> str:
    sources = cfg.setdefault("param_sources", {})
    return str(sources.get(key) or "default")


def set_parameter_source(cfg: dict[str, Any], key: str, source: str) -> None:
    sources = cfg.setdefault("param_sources", {})
    sources[key] = source


def custom_arg_conflicts(cfg: dict[str, Any]) -> list[str]:
    conflicts: list[str] = []
    custom_args = normalize_extra_args(cfg.get("custom_args", []))
    for arg in custom_args:
        option = arg.split("=", 1)[0]
        if option in STRUCTURED_ARG_OPTIONS and option not in conflicts:
            conflicts.append(option)
    return conflicts


def custom_args_status(cfg: dict[str, Any]) -> str:
    custom_args = normalize_extra_args(cfg.get("custom_args", []))
    if not custom_args:
        return "empty"
    return "conflict" if custom_arg_conflicts(cfg) else "user_experimental"


def print_parameter_card(name: str, value: Any, source: str, description: str, action: str) -> None:
    print(f"\n  {name}:")
    print(f"    {value}")
    print(f"    출처: {source}")
    print(f"    설명: {description}")
    print(f"    바꾸려면 {action}을 선택하세요.")


def show_parameter_overview(cfg: dict[str, Any]) -> None:
    print("\n  ── 파라미터 ──")
    print_parameter_card(
        "Context Size",
        cfg.get("ctx_size"),
        parameter_source(cfg, "ctx_size"),
        "모델이 한 번에 다룰 수 있는 대화/문서 길이입니다.",
        "[1] 변경",
    )
    print_parameter_card(
        "KV Cache K",
        extra_arg_value(cfg, "--cache-type-k"),
        parameter_source(cfg, "cache_type_k"),
        "긴 context 사용 시 안정성과 VRAM 사용량에 영향을 주는 KV cache K 설정입니다.",
        "[2] 변경",
    )
    print_parameter_card(
        "KV Cache V",
        extra_arg_value(cfg, "--cache-type-v"),
        parameter_source(cfg, "cache_type_v"),
        "긴 context 사용 시 안정성과 VRAM 사용량에 영향을 주는 KV cache V 설정입니다.",
        "[3] 변경",
    )
    print_parameter_card(
        "Flash Attention",
        extra_arg_value(cfg, "--flash-attn"),
        parameter_source(cfg, "flash_attn"),
        "긴 context에서 속도와 메모리 효율에 영향을 주는 llama.cpp 옵션입니다.",
        "[4] 변경",
    )

    custom_args = normalize_extra_args(cfg.get("custom_args", []))
    print("\n  사용자 추가 파라미터:")
    if custom_args:
        print("    " + " ".join(shlex.quote(x) for x in custom_args))
    else:
        print("    -")
    status = custom_args_status(cfg)
    print(f"\n  상태: {status}")
    if status == "user_experimental":
        print("  주의: 이 값은 llama-suite가 안정값으로 보장하지 않습니다.")
        print("  현재 등록된 llama-server 바이너리에서 지원되는지 확인해야 합니다.")
    elif status == "conflict":
        print("  충돌: 사용자 추가 파라미터가 구조화 설정과 같은 옵션을 다시 지정합니다.")
        print("  충돌 옵션: " + ", ".join(custom_arg_conflicts(cfg)))
        print("  구조화 설정을 쓰거나, 실험 옵션에서 중복 옵션을 제거하세요.")
    print("\n  [5] 실험 옵션 변경")
    print("  [6] Advanced raw extra args 변경")
    print("  [7] 뒤로")


def choose_kv_value(label: str, current: str) -> tuple[str | None, str | None]:
    print(f"\n  {label} preset:")
    for index, value in enumerate(KV_PRESETS, start=1):
        marker = " ◀ 현재" if current == value else ""
        print(f"    [{index}] {value}{marker}")
    print("    [C] custom")
    choice = input("  선택 > ").strip().upper()
    if choice == "C":
        value = input(f"  custom value [{current}] > ").strip()
        return (value or current, "user_experimental")
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(KV_PRESETS):
            value = KV_PRESETS[index]
            source = "llama-suite 안정성 기본값" if value == "q8_0" else "user"
            return value, source
    return None, None


def edit_parameters(cfg: dict[str, Any]) -> dict[str, Any]:
    while True:
        show_parameter_overview(cfg)
        choice = input("  선택 > ").strip().upper()
        if choice == "7":
            return cfg
        if choice == "1":
            value = input(f"  Context Size [{cfg.get('ctx_size')}] > ").strip()
            if value:
                cfg["ctx_size"] = int(value)
                set_parameter_source(cfg, "ctx_size", "user")
            continue
        if choice == "2":
            value, source = choose_kv_value("KV Cache K", extra_arg_value(cfg, "--cache-type-k"))
            if value:
                cfg["extra_args"] = set_option_value(normalize_extra_args(cfg.get("extra_args", [])), "--cache-type-k", value)
                set_parameter_source(cfg, "cache_type_k", source or "user")
            continue
        if choice == "3":
            value, source = choose_kv_value("KV Cache V", extra_arg_value(cfg, "--cache-type-v"))
            if value:
                cfg["extra_args"] = set_option_value(normalize_extra_args(cfg.get("extra_args", [])), "--cache-type-v", value)
                set_parameter_source(cfg, "cache_type_v", source or "user")
            continue
        if choice == "4":
            value = input(f"  Flash Attention on/off [{extra_arg_value(cfg, '--flash-attn')}] > ").strip().lower()
            if value in {"on", "off"}:
                cfg["extra_args"] = set_option_value(normalize_extra_args(cfg.get("extra_args", [])), "--flash-attn", value)
                set_parameter_source(cfg, "flash_attn", "user")
            continue
        if choice == "5":
            current = " ".join(shlex.quote(x) for x in normalize_extra_args(cfg.get("custom_args", [])))
            value = input(f"  사용자 추가 파라미터 [{current}] > ").strip()
            cfg["custom_args"] = normalize_extra_args(value)
            continue
        if choice == "6":
            current = " ".join(shlex.quote(x) for x in normalize_extra_args(cfg.get("extra_args", [])))
            print("  Advanced 모드입니다. 구조화된 안정값을 직접 바꿉니다.")
            value = input(f"  raw extra args [{current}] > ").strip()
            if value:
                cfg["extra_args"] = normalize_extra_args(value)
                set_parameter_source(cfg, "raw_extra_args", "advanced")


def apply_draft_to_config(cfg: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    updated = dict(cfg)
    for key in (
        "ctx_size",
        "host",
        "port",
        "llama_bin",
        "jinja",
        "alias_by_file",
        "reasoning",
        "reasoning_budget",
        "enable_thinking",
        "extra_args",
        "custom_args",
        "param_sources",
    ):
        if key in draft:
            updated[key] = draft[key]
    updated["last_model"] = draft.get("model_name")
    return updated


def save_working_draft(cfg: dict[str, Any], draft: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    if not draft.get("model_name") or not draft.get("model_path"):
        return False, "저장할 모델이 선택되지 않았습니다. [M] 모델 변경을 먼저 선택하세요.", cfg

    updated = apply_draft_to_config(cfg, draft)
    try:
        save_config(updated)
    except Exception as exc:
        return False, f"설정 저장 실패: {exc}", cfg

    try:
        profiles = load_profiles()
        profile = get_model_profile(profiles, str(draft["model_name"]), str(draft["model_path"]))
        profile["stable_ctx_size"] = int(draft["ctx_size"])
        profile["reasoning"]["mode"] = draft.get("reasoning")
        profile["reasoning"]["budget"] = draft.get("reasoning_budget")
        profile["reasoning"]["enable_thinking"] = draft.get("enable_thinking")
        save_profiles(profiles)
    except Exception as exc:
        return False, f"기본 설정은 저장했지만 profile 저장 실패: {exc}", updated

    draft["dirty"] = False
    draft["status"] = "저장된 profile/config와 같은 값입니다."
    return True, "현재 설정을 저장했습니다.", updated


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def draft_from_config(cfg: dict[str, Any], models: dict[str, str]) -> dict[str, Any]:
    model_name = cfg.get("last_model") if cfg.get("last_model") in models else None
    model_path = models.get(model_name) if model_name else None
    return {
        "model_name": model_name,
        "model_path": model_path,
        "ctx_size": safe_int(cfg.get("ctx_size", 95000), 95000),
        "host": cfg.get("host", "127.0.0.1"),
        "port": safe_int(cfg.get("port", 8080), 8080),
        "llama_bin": cfg.get("llama_bin"),
        "jinja": bool(cfg.get("jinja", True)),
        "alias_by_file": bool(cfg.get("alias_by_file", True)),
        "reasoning": cfg.get("reasoning", "off"),
        "reasoning_budget": safe_int(cfg.get("reasoning_budget", 0), 0),
        "enable_thinking": bool(cfg.get("enable_thinking", False)),
        "extra_args": normalize_extra_args(cfg.get("extra_args", [])),
        "custom_args": normalize_extra_args(cfg.get("custom_args", [])),
        "param_sources": safe_dict(cfg.get("param_sources", {})),
        "dirty": False,
        "loaded_from": "defaults/profile",
        "status": "저장된 profile/config에서 불러온 값입니다.",
    }


def draft_snapshot(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": draft.get("model_name"),
        "model_path": draft.get("model_path"),
        "ctx_size": safe_int(draft.get("ctx_size", 95000), 95000),
        "host": draft.get("host", "127.0.0.1"),
        "port": safe_int(draft.get("port", 8080), 8080),
        "llama_bin": draft.get("llama_bin"),
        "jinja": bool(draft.get("jinja", True)),
        "alias_by_file": bool(draft.get("alias_by_file", True)),
        "reasoning": draft.get("reasoning", "off"),
        "reasoning_budget": safe_int(draft.get("reasoning_budget", 0), 0),
        "enable_thinking": bool(draft.get("enable_thinking", False)),
        "extra_args": normalize_extra_args(draft.get("extra_args", [])),
        "custom_args": normalize_extra_args(draft.get("custom_args", [])),
        "param_sources": safe_dict(draft.get("param_sources", {})),
    }


def write_last_run_record(draft: dict[str, Any], action: str, path: Path = LAST_RUN_RECORD_PATH) -> tuple[bool, str]:
    record = {
        "schema": "llama-suite-last-run-v1",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "draft": draft_snapshot(draft),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    except Exception as exc:
        return False, f"last run record 저장 실패: {exc}"
    return True, f"last run record 저장됨: {path}"


def load_last_run_record(models: dict[str, str], draft: dict[str, Any], path: Path = LAST_RUN_RECORD_PATH) -> tuple[bool, str]:
    if not path.exists():
        return False, "last run record가 없습니다."
    try:
        record = json.loads(path.read_text())
    except Exception as exc:
        return False, f"last run record 읽기 실패: {exc}"
    if not isinstance(record, dict) or record.get("schema") != "llama-suite-last-run-v1":
        return False, "last run record 형식이 올바르지 않습니다."
    saved = record.get("draft")
    if not isinstance(saved, dict):
        return False, "last run record에 설정값이 없습니다."

    model_name = saved.get("model_name")
    model_path = saved.get("model_path")
    if model_name in models:
        model_path = models[model_name]
    if not model_name or not model_path:
        return False, "last run record에 모델 정보가 없습니다."

    for key, value in draft_snapshot(saved).items():
        draft[key] = value
    draft["model_name"] = model_name
    draft["model_path"] = model_path
    draft["dirty"] = True
    draft["loaded_from"] = "last run record"
    draft["status"] = "마지막 실행 기록에서 불러온 임시 작업 설정입니다. 아직 저장되지 않았습니다."
    saved_at = record.get("saved_at") or "unknown time"
    return True, f"last run record를 현재 작업 설정으로 불러왔습니다. 저장 시각: {saved_at}"


def load_script_into_draft(script_path: str, draft: dict[str, Any]) -> tuple[bool, str]:
    try:
        snapshot = parse_generated_script(script_path)
    except Exception as exc:
        return False, f"스크립트 읽기 실패: {exc}"
    if not snapshot.get("model_name") or not snapshot.get("model_path"):
        return False, "스크립트에서 MODEL/MODEL_PATH를 찾지 못했습니다."

    draft["model_name"] = snapshot["model_name"]
    draft["model_path"] = snapshot["model_path"]
    draft.update(snapshot.get("cfg") or {})
    draft["extra_args"] = normalize_extra_args(draft.get("extra_args", []))
    draft["dirty"] = True
    draft["loaded_from"] = f"script:{Path(script_path).name}"
    draft["status"] = "기존 스크립트에서 불러온 임시 작업 설정입니다. 스크립트 파일은 수정되지 않았습니다."
    return True, "스크립트 설정을 현재 작업 설정으로 불러왔습니다."


def print_working_draft_status(draft: dict[str, Any]) -> None:
    dirty = bool(draft.get("dirty"))
    print("\n" + color("  현재 설정 상태:", "section"))
    if dirty:
        print("    저장되지 않은 임시 작업 설정입니다.")
    else:
        print(f"    {draft.get('status') or '임시 작업 설정입니다.'}")
    print()
    if dirty:
        print("    이 값들은 아직 저장된 프로필에 반영되지 않았습니다.")
        print("    저장하지 않고 실행하면 이번 실행에만 사용됩니다.")
        print("    프로그램을 종료하면 저장되지 않은 변경값은 사라집니다.")
        print(f"    저장하려면 {SAVE_SETTINGS_ACTION}을 선택하세요.")
        print("    현재 값으로 한 번만 실행하려면 [O] 1회 실행을 선택하세요.")
    else:
        print("    값을 바꾸면 먼저 임시 작업 설정으로만 반영됩니다.")
        print(f"    저장은 {SAVE_SETTINGS_ACTION}을 눌렀을 때만 수행됩니다.")
    print()
    print(f"  모델: {draft.get('model_name') or '선택 없음'}")
    print(f"  endpoint: http://{draft.get('host')}:{draft.get('port')}/v1")
    print(f"  ctx: {draft.get('ctx_size')}")
    print(f"  KV: k={extra_arg_value(draft, '--cache-type-k')}, v={extra_arg_value(draft, '--cache-type-v')}, flash-attn={extra_arg_value(draft, '--flash-attn')}")
    print(f"  reasoning: {draft.get('reasoning')}, budget={draft.get('reasoning_budget')}, enable_thinking={draft.get('enable_thinking')}")
    print(f"  사용자 추가 파라미터: {custom_args_status(draft)}")
    print(f"  llama-server: {draft.get('llama_bin') or '미등록'}")


def planned_run_summary_lines(draft: dict[str, Any], running_model: str | None = None) -> list[str]:
    model = draft.get("model_name") or "선택 없음"
    endpoint = f"http://{draft.get('host')}:{draft.get('port')}/v1"
    saved_state = "저장 안 됨" if draft.get("dirty") else "저장된 값"
    custom_state = custom_args_status(draft)
    running = running_model or "없음"
    return [
        color("  ── 실행 예정 요약 ──", "section"),
        f"  실행 중: {running}",
        f"  실행될 모델: {model}",
        f"  endpoint: {endpoint}",
        f"  주요 파라미터: ctx={draft.get('ctx_size')}, kv-k={extra_arg_value(draft, '--cache-type-k')}, kv-v={extra_arg_value(draft, '--cache-type-v')}, flash-attn={extra_arg_value(draft, '--flash-attn')}",
        f"  사용자 추가 파라미터: {custom_state}",
        f"  현재 설정 저장 상태: {saved_state}",
    ]


def print_planned_run_summary(draft: dict[str, Any], running_model: str | None = None) -> None:
    print()
    for line in planned_run_summary_lines(draft, running_model):
        print(line)


def recent_vllm_run_summary_line(summary: Any = None) -> str:
    value = summary or latest_vllm_run_summary()
    if not value.ok:
        return "  Recent vLLM run: no run record"
    return f"  Recent vLLM run: {value.preset_id or '-'} / {value.model or '-'} / {value.endpoint or '-'} / {value.status}"


def recent_vllm_run_startup_warnings(summary: Any) -> list[str]:
    if getattr(summary, "ok", False):
        return []
    messages = [str(message) for message in getattr(summary, "messages", [])]
    if not messages:
        return []
    quiet_fragments = ("no vLLM run records found", "no record")
    if all(any(fragment in message for fragment in quiet_fragments) for message in messages):
        return []
    return ["latest vLLM run record could not be loaded cleanly"] + messages


def print_recent_vllm_run_summary(summary: Any = None) -> None:
    print(recent_vllm_run_summary_line(summary))


def selected_vllm_profile_summary_line(profile: Any, profile_id: str = "custom-draft") -> str:
    served_name = served_model_name_from_vllm_profile(profile, fallback=profile_id)
    model = getattr(profile, "model", "") or "(empty model)"
    host = getattr(profile, "host", "") or "-"
    port = getattr(profile, "port", "") or "-"
    return f"  Selected vLLM profile: {served_name} / {model} / http://{host}:{port}/v1"


def selected_vllm_profile_path_line(profile_id: str = "custom-draft") -> str:
    return f"  Selected vLLM profile path: {default_vllm_profile_path(profile_id)}"


def print_selected_vllm_profile_summary(profile: Any, profile_id: str = "custom-draft") -> None:
    print(selected_vllm_profile_summary_line(profile, profile_id))
    print(selected_vllm_profile_path_line(profile_id))


def served_model_name_from_vllm_profile(profile: Any, *, fallback: str) -> str:
    extra_args = str(getattr(profile, "extra_args", "") or "")
    try:
        parts = shlex.split(extra_args)
    except ValueError:
        return fallback
    for index, part in enumerate(parts):
        if part == "--served-model-name" and index + 1 < len(parts):
            return parts[index + 1]
    return fallback


def print_backend_workflow_bridge_hints() -> None:
    print("  Backend workflow bridge:")
    print("    llama.cpp actions: [L] GGUF model selection / params / preview / run / scripts")
    print("    vLLM actions: [V] profile / materials / command preview / preflight / launch / scripts / status / API smoke")
    print("    Selected vLLM profile actions are under [V]")


def print_llama_cpp_model_summary(models: dict[str, str], draft: dict[str, Any]) -> None:
    selected = draft.get("model_name") or "(none)"
    print(f"  llama.cpp GGUF models: {len(models)} found")
    print(f"    selected: {selected}")
    print("    full list: [L] llama.cpp workspace")


def print_llama_cpp_model_list(models: dict[str, str], draft: dict[str, Any], running: str | None = None) -> None:
    numbered = list(enumerate(models.items(), 1))
    if not numbered:
        print("  llama.cpp GGUF model list: none found")
        return
    print(f"\n  llama.cpp GGUF model list ({len(models)} found)\n")
    for i, (name, _path) in numbered:
        marker = ""
        if name == running:
            marker = " ◀ 실행 중"
        elif name == draft.get("model_name"):
            marker = " ◀ 현재 작업 설정"
        print(f"  [{i:>2}] {name}{marker}")


def print_startup_warnings(messages: list[str]) -> None:
    if not messages:
        return
    print("  Startup warnings:")
    for message in messages:
        print(f"    - {message}")


def registered_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.setdefault("registered_paths", {})
    if not isinstance(value, dict):
        value = {}
        cfg["registered_paths"] = value
    return value


def verify_registered_config_path(path: Any, *, require_writable: bool) -> tuple[bool, list[str]]:
    if not isinstance(path, str) or not path.strip():
        return False, ["config 경로가 아직 등록되지 않았습니다."]
    expanded = expand_path(path)
    messages: list[str] = []
    file_path = Path(expanded)
    if not file_path.exists():
        return False, [f"파일이 없습니다: {expanded}"]
    if not file_path.is_file():
        return False, [f"일반 파일이 아닙니다: {expanded}"]
    if not os.access(expanded, os.R_OK):
        messages.append(f"읽기 권한이 없습니다: {expanded}")
    if require_writable and not os.access(expanded, os.W_OK):
        messages.append(f"쓰기 권한이 없습니다: {expanded}")
    return not messages, messages or [f"확인됨: {expanded}"]


def integration_status_line(cfg: dict[str, Any], key: str, label: str, *, require_writable: bool) -> str:
    path = registered_paths(cfg).get(key)
    ok, messages = verify_registered_config_path(path, require_writable=require_writable)
    state = "활성화 준비됨" if ok else "비활성화"
    target = expand_path(str(path)) if path else "미등록"
    return f"  {label}: {state} ({target}) - {messages[0]}"


def print_integration_status(cfg: dict[str, Any]) -> None:
    print("\n" + color("  연동 등록 상태:", "section"))
    print(integration_status_line(cfg, "hermes_config", "Hermes 설정 변경", require_writable=True))
    print(integration_status_line(cfg, "openclaw_config", "OpenClaw inspection", require_writable=False))
    print("  자동 탐지는 후보일 뿐이고, 등록된 경로만 공식 연결 대상입니다.")


def print_config_candidates(label: str, candidates: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for candidate in candidates:
        expanded = expand_path(candidate)
        if Path(expanded).is_file():
            found.append(expanded)
    print(f"\n  {label} 자동 탐지 후보:")
    if found:
        for index, path in enumerate(found, 1):
            print(f"    [{index}] {path}")
    else:
        print("    후보 파일을 찾지 못했습니다. 직접 경로를 입력하세요.")
    return found


def preview_config_file(path: str, max_lines: int = 12) -> list[str]:
    lines = Path(path).read_text(errors="replace").splitlines()
    return lines[:max_lines]


def register_config_path(
    cfg: dict[str, Any],
    key: str,
    label: str,
    candidates: tuple[str, ...],
    *,
    require_writable: bool,
    read_only: bool,
) -> dict[str, Any]:
    found = print_config_candidates(label, candidates)
    current = registered_paths(cfg).get(key)
    if current:
        print(f"  현재 등록된 경로: {expand_path(str(current))}")
    value = input(f"  {label} config 경로 > ").strip()
    if not value:
        print("  변경하지 않았습니다.")
        return cfg
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(found):
            value = found[index]
        else:
            print("  ⚠️  유효하지 않은 후보 번호입니다.")
            return cfg

    expanded = expand_path(value)
    ok, messages = verify_registered_config_path(expanded, require_writable=require_writable)
    if not ok:
        print("  ⚠️  등록할 수 없습니다.")
        for message in messages:
            print(f"     - {message}")
        return cfg

    updated = dict(cfg)
    paths = dict(registered_paths(updated))
    paths[key] = expanded
    updated["registered_paths"] = paths
    try:
        save_config(updated)
    except Exception as exc:
        print(f"  ⚠️  등록 정보 저장 실패: {exc}")
        return cfg

    print(f"  ✅ {label} config 경로를 등록했습니다.")
    print("  등록된 경로만 공식 연결 대상으로 사용됩니다.")
    if read_only:
        print("  OpenClaw는 현재 읽기 전용 inspection만 수행합니다. 위험한 쓰기 작업은 하지 않습니다.")
    else:
        print("  실제 수정 기능은 diff/백업/사용자 확인/atomic replace 흐름이 준비된 뒤에만 수행됩니다.")
    print("  읽기 전용 미리보기:")
    for line in preview_config_file(expanded):
        print(f"    {line}")
    return updated


def show_hermes_integration_menu(cfg: dict[str, Any]) -> dict[str, Any]:
    print("\n  ── Hermes 연동 ──")
    print("  [1] Hermes config 등록")
    print("  [2] latest vLLM endpoint sync preview")
    print("  [3] latest vLLM endpoint sync write")
    print("  [4] Hermes latest vLLM chat smoke")
    choice = input("  선택 > ").strip()
    if choice == "1":
        return register_config_path(
            cfg,
            "hermes_config",
            "Hermes",
            HERMES_CONFIG_CANDIDATES,
            require_writable=True,
            read_only=False,
        )

    if choice in {"2", "3"}:
        config_path = registered_paths(cfg).get("hermes_config")
        plan = build_hermes_vllm_sync_plan(config_path)
        for line in format_hermes_vllm_sync_plan(plan):
            print(f"  {line}" if line else "")
        if choice == "2":
            print("  preview only. config was not modified.")
            return cfg

        print("  계속하려면 write 또는 WRITE 를 정확히 입력하세요.")
        confirm = input("  confirmation > ").strip()
        result = write_hermes_vllm_sync_plan(plan, confirmed=(confirm.lower() == "write"))
        print("\n  Hermes vLLM sync write result:")
        print(f"  ok: {result.ok}")
        if result.config_path:
            print(f"  config_path: {result.config_path}")
        if result.backup_path:
            print(f"  backup_path: {result.backup_path}")
        for message in result.messages:
            print(f"  - {message}")
        return cfg

    if choice == "4":
        plan = build_hermes_vllm_smoke_plan()
        print("\n  Hermes latest vLLM chat smoke preview:")
        print("  Tool use is not requested. This checks plain Hermes chat against latest vLLM.")
        print(f"  ok: {plan.ok}")
        print(f"  smoke_kind: {plan.smoke_kind}")
        if plan.base_url:
            print(f"  base_url: {plan.base_url}")
        if plan.model_id:
            print(f"  model: {plan.model_id}")
        if plan.command:
            print("  command: " + " ".join(shlex.quote(part) for part in plan.command))
        for message in plan.messages:
            print(f"  - {message}")
        print("  계속하려면 smoke 또는 SMOKE 를 정확히 입력하세요.")
        confirm = input("  confirmation > ").strip()
        result = run_hermes_vllm_smoke(confirmed=(confirm.lower() == "smoke"))
        print_hermes_smoke_result("Hermes latest vLLM chat smoke result", result)
        return cfg

    print("  취소했습니다.")
    return cfg


def show_hermes_vllm_sync_menu(cfg: dict[str, Any]) -> dict[str, Any]:
    print("\n  ── Hermes vLLM sync ──")
    print("  latest vLLM run이 READY일 때만 Hermes endpoint 동기화를 준비합니다.")
    print("  [1] preview")
    print("  [2] write")
    choice = input("  선택 > ").strip()
    if choice not in {"1", "2"}:
        print("  취소했습니다.")
        return cfg

    config_path = registered_paths(cfg).get("hermes_config")
    plan = build_hermes_vllm_sync_plan(config_path)
    for line in format_hermes_vllm_sync_plan(plan):
        print(f"  {line}" if line else "")
    if choice == "1":
        print("  preview only. config was not modified.")
        return cfg

    print("  계속하려면 write 또는 WRITE 를 정확히 입력하세요.")
    confirm = input("  confirmation > ").strip()
    result = write_hermes_vllm_sync_plan(plan, confirmed=(confirm.lower() == "write"))
    print("\n  Hermes vLLM sync write result:")
    print(f"  ok: {result.ok}")
    if result.config_path:
        print(f"  config_path: {result.config_path}")
    if result.backup_path:
        print(f"  backup_path: {result.backup_path}")
    for message in result.messages:
        print(f"  - {message}")
    return cfg


def show_hermes_vllm_chat_smoke() -> None:
    plan = build_hermes_vllm_smoke_plan()
    print("\n  Hermes chat smoke preview:")
    print("  Tool use is not requested. This checks plain Hermes chat against latest vLLM.")
    print(f"  ok: {plan.ok}")
    print(f"  smoke_kind: {plan.smoke_kind}")
    if plan.base_url:
        print(f"  base_url: {plan.base_url}")
    if plan.model_id:
        print(f"  model: {plan.model_id}")
    if plan.command:
        print("  command: " + " ".join(shlex.quote(part) for part in plan.command))
    for message in plan.messages:
        print(f"  - {message}")
    print("  계속하려면 smoke 또는 SMOKE 를 정확히 입력하세요.")
    confirm = input("  confirmation > ").strip()
    result = run_hermes_vllm_smoke(confirmed=(confirm.lower() == "smoke"))
    reason = "raw tool-call markup leaked" if getattr(result, "raw_markup_detected", False) else ""
    print_vllm_readiness_summary(api_status="UNKNOWN", hermes_chat_status=readiness_label(result.status), hermes_tool_status="NOT RUN", reason=reason)
    print_hermes_smoke_result("Hermes chat smoke result", result)


def show_hermes_vllm_tool_agent_smoke() -> None:
    plan = build_hermes_vllm_tool_agent_smoke_plan()
    print("\n  Hermes tool-agent smoke / raw markup check:")
    print("  This patch does not run a dangerous tool-agent task.")
    print(f"  ok: {plan.ok}")
    print(f"  smoke_kind: {plan.smoke_kind}")
    for message in plan.messages:
        print(f"  - {message}")
    result = run_hermes_vllm_tool_agent_smoke(confirmed=False)
    print_vllm_readiness_summary(api_status="UNKNOWN", hermes_chat_status="UNKNOWN", hermes_tool_status=readiness_label(result.status))
    print_hermes_smoke_result("Hermes tool-agent smoke result", result)


def readiness_label(status: str) -> str:
    return str(status or "UNKNOWN").replace("_", " ").upper()


def print_vllm_readiness_summary(*, api_status: str, hermes_chat_status: str, hermes_tool_status: str, reason: str = "") -> None:
    print("\n  ── vLLM readiness ──")
    print(f"  API smoke: {api_status}")
    print(f"  Hermes chat smoke: {hermes_chat_status}")
    print(f"  Hermes tool-agent smoke: {hermes_tool_status}")
    if reason:
        print(f"  reason: {reason}")


def print_hermes_smoke_result(title: str, result: Any) -> None:
    print(f"\n  {title}:")
    print(f"  ok: {result.ok}")
    print(f"  status: {getattr(result, 'status', '-')}")
    print(f"  smoke_kind: {getattr(result, 'smoke_kind', '-')}")
    print(f"  returncode: {result.returncode if result.returncode is not None else '-'}")
    print(f"  raw_markup_detected: {getattr(result, 'raw_markup_detected', False)}")
    patterns = getattr(result, "raw_markup_patterns", None) or []
    if patterns:
        print("  raw_markup_patterns: " + ", ".join(patterns))
    if result.stdout:
        print("  stdout:")
        for line in result.stdout.splitlines()[:20]:
            print(f"    {line}")
    if result.stderr:
        print("  stderr:")
        for line in result.stderr.splitlines()[:20]:
            print(f"    {line}")
    for message in result.messages:
        print(f"  - {message}")


def final_preview_text(draft: dict[str, Any]) -> str:
    model_name = draft.get("model_name")
    model_path = draft.get("model_path")
    if not model_name or not model_path:
        return "모델이 선택되지 않았습니다. 다른 모델을 사용하려면 [M] 모델 변경을 선택하세요."
    try:
        final_command = command_preview(str(model_name), str(model_path), draft)
    except Exception as exc:
        final_command = f"명령 생성 실패: {exc}"

    lines = [
        "[1] 최종 실행 명령",
        final_command,
        "",
        "[2] 실행 요약",
        f"현재 실행할 모델은 {Path(str(model_path)).name} 입니다.",
        "다른 모델을 사용하려면 [M] 모델 변경을 선택하세요.",
        "",
        f"사용될 endpoint는 http://{draft.get('host')}:{draft.get('port')}/v1 입니다.",
        "주소를 바꾸려면 [A] 설정 변경을 선택하세요.",
        "",
        "사용될 주요 파라미터는 다음과 같습니다.",
        f"- Context Size: {draft.get('ctx_size')}",
        f"- KV/기타 추가 파라미터: {' '.join(shlex.quote(x) for x in normalize_extra_args(draft.get('extra_args', []))) or '-'}",
        f"- 사용자 추가 파라미터: {' '.join(shlex.quote(x) for x in normalize_extra_args(draft.get('custom_args', []))) or '-'}",
        f"- reasoning: {draft.get('reasoning')} / budget={draft.get('reasoning_budget')} / enable_thinking={draft.get('enable_thinking')}",
    ]
    conflicts = custom_arg_conflicts(draft)
    if conflicts:
        lines.extend(
            [
                "",
                "주의: 사용자 추가 파라미터가 구조화 설정과 충돌합니다.",
                "충돌 옵션: " + ", ".join(conflicts),
                "충돌을 해결하려면 [K] 파라미터에서 실험 옵션을 변경하세요.",
            ]
        )
    if draft.get("dirty"):
        lines.extend(
            [
                "",
                "이번 실행에는 현재 화면에 보이는 임시 설정이 사용됩니다.",
                "이 값들은 아직 저장된 프로필에 반영되지 않았습니다.",
                "프로그램 종료 시 저장되지 않은 변경값은 사라집니다.",
                f"저장하려면 {SAVE_SETTINGS_ACTION}을 선택하세요.",
            ]
        )
    return "\n".join(lines)


def confirm_final_preview(draft: dict[str, Any], action_label: str) -> bool:
    print()
    print(final_preview_text(draft))
    return input(f"\n  {action_label} 계속할까요? (y/n) > ").strip().lower() == "y"


def choose_script_generation_action(draft: dict[str, Any]) -> str | None:
    print()
    print("  [G] 새 스크립트 생성이 선택되어 있습니다.")
    print()
    print("  현재 설정을 바탕으로 새로운 실행 스크립트가 생성됩니다.")
    print("  기존 스크립트는 삭제되거나 덮어쓰이지 않습니다.")
    print("  기존 스크립트를 정리하려면 [S] 스크립트 관리 메뉴에서 수동으로 삭제하세요.")
    print()
    print(final_preview_text(draft))
    print()
    print("  생성 후 동작을 선택하세요.")
    print("  [1] 생성만")
    print("  [2] 생성 후 실행")
    print("  [R] 작업 화면으로 돌아가기")
    choice = input("  선택 > ").strip().upper()
    if choice == "1":
        return "create"
    if choice == "2":
        return "create_and_run"
    return None


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
        print("\n  📜 저장된 llama.cpp 스크립트가 없습니다.\n")
        return

    running_lines = "\n".join(get_running_servers())

    print(f"\n  📜 저장된 llama.cpp 스크립트 ({len(scripts)}개)\n")
    print("  이 목록은 llama.cpp GGUF 실행 스크립트 전용입니다.")
    print("  vLLM 스크립트 preview/save는 [V] vLLM workspace → vLLM profile 안에 있습니다.\n")
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


def select_script_path() -> str | None:
    scripts = list_scripts()
    if not scripts:
        print("\n  저장된 llama.cpp 스크립트가 없습니다.")
        return None
    show_scripts()
    choice = input("  실행/관리할 llama.cpp 스크립트 번호 > ").strip()
    if not choice.isdigit():
        print("  ⚠️  번호를 입력하세요.")
        return None
    index = int(choice) - 1
    if not 0 <= index < len(scripts):
        print("  ⚠️  유효하지 않은 번호입니다.")
        return None
    return scripts[index][1]


def show_script_readonly(script_path: str) -> None:
    print("\n  이 화면은 읽기 전용입니다.")
    print("  선택한 llama.cpp 스크립트 파일은 여기서 직접 수정되지 않습니다.")
    print()
    print("  스크립트의 설정을 바꾸고 싶다면 [3] 현재 설정으로 불러오기를 선택하세요.")
    print("  불러온 뒤 메인 화면에서 필요한 값을 변경하고, 새 실행 스크립트를 생성할 수 있습니다.")
    print("  기존 스크립트는 삭제되거나 덮어쓰이지 않습니다.")
    print()
    try:
        print(Path(script_path).read_text())
    except Exception as exc:
        print(f"  ⚠️  읽기 실패: {exc}")


def manage_scripts(draft: dict[str, Any]) -> None:
    selected_script: str | None = None
    while True:
        show_scripts()
        scripts = list_scripts()
        if not scripts:
            break

        print(f"  선택된 llama.cpp 스크립트: {Path(selected_script).name if selected_script else '없음'}")
        print("  [1] 실행할 llama.cpp 스크립트 선택")
        print("  [2] llama.cpp 스크립트 내용 보기")
        print("  [3] llama.cpp 현재 설정으로 불러오기")
        print("  [4] 이 llama.cpp 스크립트 그대로 실행")
        print("  [5] llama.cpp 스크립트 삭제")
        print("  [6] 뒤로\n")

        try:
            choice = input("  선택 > ").strip()
        except EOFError:
            print("\n👋 안녕!\n")
            break

        if choice == "6":
            break

        if choice == "1":
            selected_script = select_script_path()
            pause()
            continue

        if choice in {"2", "3", "4", "5"} and not selected_script:
            print("  먼저 [1] 실행할 llama.cpp 스크립트 선택을 선택하세요.")
            pause()
            continue

        if choice == "2":
            show_script_readonly(selected_script)
            pause()
            continue

        if choice == "3":
            ok, message = load_script_into_draft(selected_script, draft)
            print("  " + ("✅ " if ok else "⚠️  ") + message)
            pause()
            if ok:
                break
            continue

        if choice == "4":
            run_existing_script(selected_script)
            pause()
            continue

        if choice == "5":
            confirm = input("  이 스크립트를 삭제할까요? (y/n) > ").strip().lower()
            if confirm == "y":
                try:
                    index = [path for _, path in scripts].index(selected_script)
                    delete_script(index)
                    selected_script = None
                except ValueError:
                    print("  ⚠️  선택된 스크립트를 찾지 못했습니다.")
            pause()


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


def show_vllm_doctor() -> None:
    print("\n  ── vLLM doctor ──")
    print("  모델은 실행하지 않고 wrapper/version/Torch HIP 상태만 확인합니다.")
    report = run_vllm_doctor()
    for line in format_vllm_doctor_report(report).splitlines():
        print(f"  {line}" if line else "")


def format_vllm_profile_section(title: str, profile: Any, port_check: Any = None) -> list[str]:
    return format_vllm_profile_report(title, profile, port_check=port_check)


def vllm_profile_preview_text(port_check: Any = None) -> str:
    presets = builtin_vllm_profile_presets()
    lines = [
        "vLLM profile overview",
        "이 화면은 vLLM 전용 profile/preset 미리보기입니다. llama.cpp 파라미터와 별개입니다.",
        "",
        f"Built-in smoke launch preset: {future_launch_preset_id()}",
        "Custom profile selection/save/load/launch는 아래 vLLM profile menu에서 별도로 처리합니다.",
        "",
        "Launch confirmation guidance:",
    ]
    for guidance in launch_confirmation_guidance_lines():
        lines.append(f"- {guidance}")
    lines.extend(
        [
            "",
            "Available built-in vLLM profiles:",
        ]
    )
    for preset in presets:
        lines.append(f"- {preset.id}: {preset.label} - {preset.description}")
    lines.extend(["", "Built-in presets are read-only templates. Custom drafts can be copied, edited, saved, and launched separately."])
    for preset in presets:
        lines.append("")
        if preset.id == "smoke-qwen-0.5b":
            lines.append("Smoke profile preset (read-only)")
            lines.append("이미 이 시스템에서 성공한 작은 vLLM 확인용 preset입니다. 실행하지 않습니다.")
        if preset.id == "template-local-large-q4":
            lines.append("Local large Q4 template preset (read-only)")
            lines.append("다운로드/launch 없이 /mnt/data_main/downloads/models 아래 HF/safetensors Q4급 디렉터리 형태만 제안합니다.")
        if preset.id == "verified-gemma4-26b-awq-auto":
            lines.append("Verified local Gemma4 26B AWQ profile (read-only)")
            lines.append("이 시스템의 vLLM beta launch / API smoke / Hermes plain chat 기준 profile입니다.")
            lines.append("tool-agent coding 기준 profile로는 아직 검증하지 않았습니다. 실행하려면 custom draft로 복사하세요.")
        lines.extend(format_vllm_profile_section(f"Preset {preset.id}: {preset.label}", preset.profile, port_check=port_check))
    lines.extend(["", "Host guidance:"])
    for guidance in host_guidance_lines():
        lines.append(f"- {guidance}")
    lines.extend(["", "Local / quantized model guidance:"])
    for guidance in large_model_guidance_lines():
        lines.append(f"- {guidance}")
    return "\n".join(lines)


def vllm_custom_profile_text(profile: Any, port_check: Any = None) -> str:
    lines = [
        "vLLM custom profile draft",
        "Preview/preflight first. Save and launch are separate explicit menu actions.",
        "이 draft는 llama.cpp 설정과 별개입니다.",
        "",
    ]
    lines.extend(format_vllm_profile_section("Custom vLLM profile draft", profile, port_check=port_check))
    lines.extend(["", "Editable vLLM fields:"])
    for spec in editable_vllm_profile_field_specs():
        value = getattr(profile, spec.name, "")
        display_value = str(value) if str(value) else "(empty)"
        lines.append(f"- {spec.group} / {spec.name}: {display_value} ({spec.help})")
        lines.append(f"  hint: {spec.input_hint}")
        lines.append(f"  example: {spec.example}")
    lines.extend(["", "Host guidance:"])
    for guidance in host_guidance_lines():
        lines.append(f"- {guidance}")
    lines.extend(["", "Local / quantized model guidance:"])
    for guidance in large_model_guidance_lines():
        lines.append(f"- {guidance}")
    return "\n".join(lines)


def show_vllm_profile_menu(profile: Any, profile_id: str = "custom-draft", *, return_profile_id: bool = False) -> Any:
    print("\n  ── vLLM selected profile workspace ──")
    print(f"  selected profile: {profile_id}")
    print(f"  profile store root: {Path(default_vllm_profile_path()).parent}")
    print(f"  selected draft JSON path: {default_vllm_profile_path(profile_id)}")
    print(f"  selected model: {getattr(profile, 'model', '') or '(empty model)'}")
    print("  Run / verify selected profile")
    print("  [11] launch selected vLLM profile")
    print("  [2] selected profile preview / dry-run / preflight")
    print("  [9] selected profile script preview")
    print("  [10] save selected profile script")
    print("  Choose / import profile")
    print("  [5] list saved custom profiles")
    print("  [6] load saved custom profile from list")
    print("  [8] profile JSON import/validate/preview")
    print("  Edit selected profile")
    print("  [3] edit selected profile draft")
    print("  [4] save selected profile draft")
    print("  [7] delete saved custom profile from list")
    print("  Reference")
    print("  [1] built-in profile preview")
    choice = input("  선택 > ").strip()

    if choice == "1":
        for line in vllm_profile_preview_text().splitlines():
            print(f"  {line}" if line else "")
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "2":
        for line in vllm_custom_profile_text(profile).splitlines():
            print(f"  {line}" if line else "")
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "3":
        return _vllm_profile_menu_return(edit_vllm_custom_profile(profile), profile_id, return_profile_id)
    if choice == "4":
        updated_profile_id = prompt_vllm_profile_id(profile_id)
        result = save_vllm_profile_draft(profile, profile_id=updated_profile_id)
        print_vllm_profile_store_result(result)
        if result.ok:
            profile_id = updated_profile_id
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "5":
        print_vllm_profile_list_result(list_vllm_profile_drafts(), selected_profile_id=profile_id)
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "6":
        loaded_profile, loaded_profile_id = load_vllm_profile_from_list(profile, profile_id)
        return _vllm_profile_menu_return(loaded_profile, loaded_profile_id, return_profile_id)
    if choice == "7":
        deleted_profile_id = delete_vllm_profile_from_list(profile_id)
        if deleted_profile_id == profile_id:
            profile_id = "custom-draft"
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "8":
        profile, profile_id = show_vllm_profile_json_menu(profile, profile_id)
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "9":
        preview = build_vllm_script_preview(profile)
        print_vllm_script_preview(preview)
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "10":
        result = save_vllm_script(profile)
        print_vllm_script_save_result(result)
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)
    if choice == "11":
        profile, profile_id = show_vllm_custom_launch(profile, profile_id)
        return _vllm_profile_menu_return(profile, profile_id, return_profile_id)

    print("  취소했습니다.")
    return _vllm_profile_menu_return(profile, profile_id, return_profile_id)


def _vllm_profile_menu_return(profile: Any, profile_id: str, return_profile_id: bool) -> Any:
    if return_profile_id:
        return profile, profile_id
    return profile


def show_vllm_profile_json_menu(profile: Any, profile_id: str) -> tuple[Any, str]:
    print("\n  ── vLLM profile JSON / preset ──")
    print(f"  selected draft JSON path: {default_vllm_profile_path(profile_id)}")
    print(f"  small example JSON path: {Path('examples/vllm-profile.example.json')}")
    print(f"  local large example JSON path: {Path('examples/vllm-profile.local-large.example.json')}")
    print("  These files are safe to open in an editor; this menu does not launch an editor.")
    print("  [1] profile JSON preview")
    print("  [2] import profile JSON file")
    print("  [3] validate profile JSON file")
    print("  [4] copy built-in preset to custom draft")
    choice = input("  선택 > ").strip()
    if choice == "1":
        print_vllm_profile_json_preview(profile, profile_id)
        return profile, profile_id
    if choice == "2":
        return import_vllm_profile_json_file(profile, profile_id)
    if choice == "3":
        validate_vllm_profile_json_file_from_menu()
        return profile, profile_id
    if choice == "4":
        return copy_vllm_builtin_preset_to_draft(profile, profile_id)
    print("  취소했습니다.")
    return profile, profile_id


def edit_vllm_custom_profile(profile: Any) -> Any:
    specs = editable_vllm_profile_field_specs()
    fields = [spec.name for spec in specs]
    print("\n  ── vLLM custom profile edit ──")
    print("  profile 값을 한 개씩 수정합니다. 저장/launch는 별도 메뉴에서만 실행합니다.")
    current_group = ""
    for index, spec in enumerate(specs, 1):
        if spec.group != current_group:
            current_group = spec.group
            print(f"  {current_group}")
        current = getattr(profile, spec.name, "")
        display_value = str(current) if str(current) else "(empty)"
        print(f"  [{index}] {spec.name}: {display_value}")
        print(f"      {spec.label} - {spec.help}")
        print(f"      hint: {spec.input_hint}")
        print(f"      example: {spec.example}")
    selected = input("  field 번호 또는 이름 > ").strip()
    field_name = selected
    if selected.isdigit():
        index = int(selected)
        if 1 <= index <= len(fields):
            field_name = fields[index - 1]
    if field_name not in fields:
        updated, messages = update_vllm_profile_field(profile, field_name, "")
        for message in messages:
            print(f"  - {message}")
        return updated
    current = getattr(profile, field_name, "")
    selected_spec = next((spec for spec in specs if spec.name == field_name), None)
    if selected_spec:
        print(f"  selected: {selected_spec.group} / {selected_spec.name}")
        print(f"  help: {selected_spec.help}")
        print(f"  hint: {selected_spec.input_hint}")
        print(f"  example: {selected_spec.example}")
    print(f"  new value for {field_name} [{current}]")
    raw_value = input("  > ").strip()
    if not raw_value:
        raw_value = str(current)
    updated, messages = update_vllm_profile_field(profile, field_name, raw_value)
    for message in messages:
        print(f"  - {message}")
    print()
    for line in vllm_custom_profile_text(updated).splitlines():
        print(f"  {line}" if line else "")
    return updated


def prompt_vllm_profile_id(default: str = "custom-draft") -> str:
    raw = input(f"  profile id [{default}] > ").strip()
    return raw or default


def load_vllm_profile_from_list(profile: Any, profile_id: str) -> tuple[Any, str]:
    result = list_vllm_profile_drafts()
    print_vllm_profile_list_result(result, selected_profile_id=profile_id)
    if not result.profiles:
        return profile, profile_id

    raw = input("  load profile number > ").strip()
    try:
        selected_index = int(raw)
    except ValueError:
        print("  취소했습니다.")
        return profile, profile_id
    if not 1 <= selected_index <= len(result.profiles):
        print("  취소했습니다.")
        return profile, profile_id

    selected = result.profiles[selected_index - 1]
    load_result = load_vllm_profile_draft(profile_id=selected.profile_id)
    print_vllm_profile_store_result(load_result)
    if load_result.ok and load_result.profile:
        return load_result.profile, selected.profile_id
    return profile, profile_id


def delete_vllm_profile_from_list(selected_profile_id: str = "custom-draft") -> str | None:
    result = list_vllm_profile_drafts()
    print_vllm_profile_list_result(result, selected_profile_id=selected_profile_id)
    if not result.profiles:
        return None

    raw = input("  delete profile number > ").strip()
    try:
        selected_index = int(raw)
    except ValueError:
        print("  취소했습니다.")
        return None
    if not 1 <= selected_index <= len(result.profiles):
        print("  취소했습니다.")
        return None

    selected = result.profiles[selected_index - 1]
    print(f"  삭제하려면 delete 또는 DELETE 를 정확히 입력하세요: {selected.profile_id}")
    confirm = input("  confirmation > ").strip()
    if confirm.lower() != "delete":
        print("  취소했습니다.")
        return None
    delete_result = delete_vllm_profile_draft(profile_id=selected.profile_id, confirmed=True)
    print_vllm_profile_store_result(delete_result)
    if delete_result.ok:
        return selected.profile_id
    return None


def print_vllm_profile_store_result(result: Any) -> None:
    print("\n  vLLM profile draft store:")
    print(f"  ok: {result.ok}")
    print(f"  profile_path: {result.profile_path or '-'}")
    for message in result.messages:
        print(f"  - {message}")


def print_vllm_profile_selection_result(result: Any) -> None:
    print("\n  vLLM beta profile selection:")
    print(f"  ok: {result.ok}")
    print(f"  profile_id: {result.profile_id}")
    print(f"  profile_path: {result.profile_path or '-'}")
    print(f"  selected_state_path: {result.selected_state_path or '-'}")
    if result.profile:
        print(selected_vllm_profile_summary_line(result.profile, result.profile_id))
        print(selected_vllm_profile_path_line(result.profile_id))
    for message in result.messages:
        print(f"  - {message}")


def print_vllm_selected_profile_result(result: Any) -> None:
    print("\n  vLLM selected profile state:")
    print(f"  ok: {result.ok}")
    print(f"  state_path: {result.state_path}")
    for message in result.messages:
        print(f"  - {message}")


def print_vllm_profile_list_result(result: Any, selected_profile_id: str | None = None) -> None:
    print("\n  vLLM saved custom profiles:")
    print(f"  ok: {result.ok}")
    print(f"  store_root: {result.store_root}")
    for index, profile in enumerate(result.profiles, 1):
        state = "valid" if not profile.validation_messages else "needs attention"
        marker = " *selected*" if selected_profile_id and profile.profile_id == selected_profile_id else ""
        print(f"  [{index}] {profile.profile_id}: {profile.model or '(empty model)'} [{state}]{marker}")
        print(f"    path: {profile.profile_path}")
        for message in profile.validation_messages:
            print(f"    validation: {message}")
    for message in result.messages:
        print(f"  - {message}")


def print_vllm_profile_json_preview(profile: Any, profile_id: str = "custom-draft") -> None:
    print("\n  vLLM profile JSON preview:")
    for line in format_vllm_profile_draft_json(profile, profile_id=profile_id).splitlines():
        print(f"  {line}" if line else "")


def import_vllm_profile_json_file(profile: Any, profile_id: str) -> tuple[Any, str]:
    raw_path = input("  profile JSON path > ").strip()
    if not raw_path:
        print("  취소했습니다.")
        return profile, profile_id
    result = load_vllm_profile_json_file(raw_path)
    print_vllm_profile_store_result(result)
    if result.ok and result.profile:
        return result.profile, result.profile_id or Path(result.profile_path or raw_path).stem
    return profile, profile_id


def validate_vllm_profile_json_file_from_menu() -> None:
    raw_path = input("  profile JSON path > ").strip()
    if not raw_path:
        print("  취소했습니다.")
        return
    result = validate_vllm_profile_json_file(raw_path)
    print_vllm_profile_store_result(result)


def copy_vllm_builtin_preset_to_draft(profile: Any, profile_id: str) -> tuple[Any, str]:
    presets = builtin_vllm_profile_presets()
    print("\n  vLLM built-in presets:")
    for index, preset in enumerate(presets, 1):
        print(f"  [{index}] {preset.id}: {preset.label}")
        print(f"      {preset.description}")
    raw = input("  copy preset number > ").strip()
    try:
        selected_index = int(raw)
    except ValueError:
        print("  취소했습니다.")
        return profile, profile_id
    if not 1 <= selected_index <= len(presets):
        print("  취소했습니다.")
        return profile, profile_id

    selected = presets[selected_index - 1]
    copied_profile_id = f"draft-from-{selected.id}"
    print(f"  copied built-in preset to in-memory custom draft: {copied_profile_id}")
    print(f"  next [4] save default profile id: {copied_profile_id}")
    print(f"  next [4] save path: {default_vllm_profile_path(copied_profile_id)}")
    print("  저장/launch는 하지 않았습니다. 필요하면 [4] save custom profile draft를 사용하세요.")
    return selected.profile, copied_profile_id


def print_vllm_script_preview(preview: Any) -> None:
    print("\n  vLLM custom script preview:")
    print(f"  ok: {preview.ok}")
    for message in preview.messages:
        print(f"  - {message}")
    if preview.script_text:
        print()
        for line in preview.script_text.splitlines():
            print(f"  {line}" if line else "")


def print_vllm_script_save_result(result: Any) -> None:
    print("\n  vLLM custom script save:")
    print(f"  ok: {result.ok}")
    print(f"  script_path: {result.script_path or '-'}")
    for message in result.messages:
        print(f"  - {message}")


def show_vllm_custom_launch(profile: Any, profile_id: str = "custom-draft") -> tuple[Any, str]:
    print("\n  ── vLLM custom profile launch ──")
    print(f"  custom draft를 실제 vLLM launch 대상으로 사용합니다: {profile_id}")
    print("  launch 전 profile preview, command dry-run, preflight를 다시 표시합니다.\n")
    for line in vllm_custom_profile_text(profile).splitlines():
        print(f"  {line}" if line else "")
    preflight = run_vllm_preflight(profile)
    if vllm_preflight_has_port_conflict(preflight):
        return handle_vllm_port_conflict(profile, profile_id)
    print("\n  계속하려면 launch 또는 LAUNCH 를 정확히 입력하세요.")
    confirm = input("  confirmation > ").strip()
    result = launch_vllm_profile_once(
        profile,
        confirmed=(confirm.lower() == "launch"),
        preset_id=profile_id,
        profile_path=default_vllm_profile_path(profile_id),
    )
    print_vllm_launch_result(result)
    return profile, profile_id


def vllm_preflight_has_port_conflict(preflight: Any) -> bool:
    for check in getattr(preflight, "checks", []):
        if getattr(check, "name", "") != "port availability" or getattr(check, "ok", True):
            continue
        message = str(getattr(check, "message", ""))
        return "already in use" in message or "not available" in message or "Address already in use" in message
    return False


def handle_vllm_port_conflict(profile: Any, profile_id: str) -> tuple[Any, str]:
    print("\n  ── vLLM port conflict ──")
    for line in vllm_port_conflict_guidance_lines(profile):
        print(f"  {line}" if line else "")
    choice = input("  선택 > ").strip().upper()
    if choice == "1":
        print("  기존 서버 재사용을 선택했습니다. READY 여부는 latest status 또는 API smoke로 확인하세요.")
        return profile, profile_id
    if choice == "2":
        show_vllm_smoke_manage()
        return profile, profile_id
    if choice == "3":
        return prompt_vllm_selected_profile_fields(profile, profile_id, ["host", "port"])
    if choice == "4":
        print("  기존 프로세스 종료는 latest run status/log/stop에서 명시 확인 후 진행합니다.")
        show_vllm_smoke_manage()
        return profile, profile_id
    print("  취소했습니다.")
    return profile, profile_id


def show_vllm_selected_profile_preview(profile: Any, profile_id: str) -> None:
    print("\n  ── selected vLLM profile preview / preflight ──")
    print(selected_vllm_profile_summary_line(profile, profile_id))
    print(selected_vllm_profile_path_line(profile_id))
    for line in vllm_custom_profile_text(profile).splitlines():
        print(f"  {line}" if line else "")


def show_vllm_selected_profile_settings(profile: Any, profile_id: str = "custom-draft") -> tuple[Any, str]:
    print("\n  ── vLLM selected profile settings ──")
    print(f"  selected profile id: {profile_id}")
    print(f"  profile path: {default_vllm_profile_path(profile_id)}")
    for field_name in selected_vllm_profile_setting_fields():
        value = getattr(profile, field_name, "")
        print(f"  {field_name}: {value if str(value) else 'auto' if field_name == 'max_model_len' else '-'}")
    print("  extra_args tokens:")
    print_vllm_extra_arg_tokens(profile)
    print("\n  [1] model path 변경")
    print("  [2] host / port 변경")
    print("  [3] memory / context 변경")
    print("  [4] extra_args raw edit (advanced)")
    print("  [5] common vLLM option 추가")
    print("  [6] option token 제거")
    print("  [7] 저장")
    print("  [P] preflight")
    print("  [L] launch")
    print("  [R] return")
    choice = input("  선택 > ").strip()
    upper = choice.upper()
    if choice == "1":
        return prompt_vllm_selected_profile_fields(profile, profile_id, ["model"])
    if choice == "2":
        return prompt_vllm_selected_profile_fields(profile, profile_id, ["host", "port"])
    if choice == "3":
        return prompt_vllm_selected_profile_fields(
            profile,
            profile_id,
            ["dtype", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens"],
        )
    if choice == "4":
        updated = edit_vllm_extra_args_raw(profile)
        return updated, profile_id
    if choice == "5":
        updated = add_common_vllm_extra_arg_from_menu(profile)
        return updated, profile_id
    if choice == "6":
        updated = remove_vllm_extra_arg_from_menu(profile)
        return updated, profile_id
    if choice == "7":
        saved_profile = save_vllm_selected_profile_settings(profile, profile_id)
        return saved_profile, profile_id
    if upper == "P":
        show_vllm_selected_profile_preview(profile, profile_id)
        return profile, profile_id
    if upper == "L":
        return show_vllm_custom_launch(profile, profile_id)
    print("  취소했습니다.")
    return profile, profile_id


def selected_vllm_profile_setting_fields() -> list[str]:
    return [
        "model",
        "host",
        "port",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "kv_cache_dtype",
        "max_num_seqs",
        "max_num_batched_tokens",
        "extra_args",
    ]


def prompt_vllm_selected_profile_fields(profile: Any, profile_id: str, fields: list[str]) -> tuple[Any, str]:
    updated = profile
    for field_name in fields:
        current = getattr(updated, field_name, "")
        raw_value = input(f"  {field_name} [{current}] > ").strip()
        if not raw_value:
            continue
        updated, messages = update_vllm_profile_field(updated, field_name, raw_value)
        for message in messages:
            print(f"  - {message}")
    print_selected_profile_validation_and_preview(updated)
    return updated, profile_id


def print_vllm_extra_arg_tokens(profile: Any) -> None:
    tokens, messages = tokenize_vllm_extra_args(profile)
    if messages:
        for message in messages:
            print(f"    - {message}")
        return
    if not tokens:
        print("    - none")
        return
    for index, token in enumerate(tokens, 1):
        print(f"    [{index}] {token}")


def edit_vllm_extra_args_raw(profile: Any) -> Any:
    print("  raw extra_args 입력. shlex parsing에 실패하면 반영하지 않습니다.")
    print(f"  current: {getattr(profile, 'extra_args', '')}")
    raw_value = input("  extra_args > ")
    updated, messages = update_vllm_profile_field(profile, "extra_args", raw_value)
    _tokens, parse_messages = tokenize_vllm_extra_args(updated)
    if parse_messages:
        for message in parse_messages:
            print(f"  - {message}")
        print("  저장하지 않았습니다.")
        return profile
    for message in messages:
        print(f"  - {message}")
    print_selected_profile_validation_and_preview(updated)
    return updated


def add_common_vllm_extra_arg_from_menu(profile: Any) -> Any:
    options = list(common_vllm_extra_arg_options().items())
    print("\n  common vLLM options:")
    for index, (option, requires_value) in enumerate(options, 1):
        suffix = " VALUE" if requires_value else ""
        print(f"  [{index}] {option}{suffix}")
    raw = input("  option number > ").strip()
    try:
        selected_index = int(raw)
    except ValueError:
        print("  취소했습니다.")
        return profile
    if not 1 <= selected_index <= len(options):
        print("  취소했습니다.")
        return profile
    option, requires_value = options[selected_index - 1]
    value = input(f"  value for {option} > ").strip() if requires_value else ""
    updated, messages = add_vllm_extra_arg(profile, option, value)
    for message in messages:
        print(f"  - {message}")
    print_vllm_extra_arg_tokens(updated)
    print_selected_profile_validation_and_preview(updated)
    return updated


def remove_vllm_extra_arg_from_menu(profile: Any) -> Any:
    print_vllm_extra_arg_tokens(profile)
    raw = input("  remove token number > ").strip()
    try:
        token_index = int(raw)
    except ValueError:
        print("  취소했습니다.")
        return profile
    updated, messages = remove_vllm_extra_arg_token(profile, token_index)
    for message in messages:
        print(f"  - {message}")
    print_vllm_extra_arg_tokens(updated)
    print_selected_profile_validation_and_preview(updated)
    return updated


def print_selected_profile_validation_and_preview(profile: Any) -> None:
    validation_messages = validate_vllm_profile(profile)
    print("  Validation messages:")
    if validation_messages:
        for message in validation_messages:
            print(f"  - {message}")
    else:
        print("  - none")
    command, command_messages = build_vllm_command(profile)
    print("  Command preview / dry-run:")
    if command:
        print("  " + " ".join(shlex.quote(part) for part in command))
    else:
        print("  No runnable command preview:")
        for message in command_messages:
            print(f"  - {message}")


def save_vllm_selected_profile_settings(profile: Any, profile_id: str) -> Any:
    print_selected_profile_validation_and_preview(profile)
    command, command_messages = build_vllm_command(profile)
    if command is None:
        print("  저장하지 않았습니다. command preview를 먼저 고쳐야 합니다.")
        for message in command_messages:
            print(f"  - {message}")
        return profile
    backup = backup_vllm_profile_draft(profile_id=profile_id)
    for message in backup.messages:
        print(f"  - {message}")
    if not backup.ok:
        print("  저장하지 않았습니다. backup 실패를 먼저 확인하세요.")
        return profile
    result = save_vllm_profile_draft(profile, profile_id=profile_id)
    print_vllm_profile_store_result(result)
    if result.ok:
        print_selected_vllm_profile_summary(profile, profile_id)
    return profile


def vllm_smoke_launch_preview_text(port_check: Any = None) -> str:
    presets = {preset.id: preset for preset in builtin_vllm_profile_presets()}
    preset = presets[future_launch_preset_id()]
    lines = [
        "vLLM smoke launch preview",
        f"Launch target preset: {preset.id}: {preset.label}",
        "This action is smoke-preset-only. Custom vLLM profile launch is handled separately from [B].",
        "",
        "Launch confirmation guidance:",
    ]
    for guidance in launch_confirmation_guidance_lines():
        if "no launch button" in guidance:
            continue
        lines.append(f"- {guidance}")
    lines.append("")
    lines.extend(format_vllm_profile_section(f"Preset {preset.id}: {preset.label}", preset.profile, port_check=port_check))
    lines.extend(["", "Host guidance:"])
    for guidance in host_guidance_lines():
        lines.append(f"- {guidance}")
    return "\n".join(lines)


def show_vllm_smoke_launch_once() -> None:
    print("\n  ── vLLM smoke launch ──")
    print("  smoke-qwen-0.5b preset만 1회 launch 대상으로 사용합니다.")
    print("  launch 전 command preview, preflight, confirmation guidance를 다시 표시합니다.\n")
    for line in vllm_smoke_launch_preview_text().splitlines():
        print(f"  {line}" if line else "")
    print("\n  계속하려면 launch 또는 LAUNCH 를 정확히 입력하세요.")
    confirm = input("  confirmation > ").strip()
    if confirm.lower() != "launch":
        result = launch_vllm_smoke_once(confirmed=False)
        print_vllm_launch_result(result)
        return
    result = launch_vllm_smoke_once(confirmed=True)
    print_vllm_launch_result(result)


def print_vllm_launch_result(result: Any) -> None:
    status = "started" if result.ok else "not started"
    print(f"\n  vLLM launch result: {status}")
    print(f"  ok: {result.ok}")
    print(f"  preset_id: {result.preset_id}")
    print(f"  pid: {result.pid if result.pid is not None else '-'}")
    print(f"  run_id: {result.run_id or '-'}")
    print(f"  log_path: {result.log_path or '-'}")
    print(f"  record_path: {result.record_path or '-'}")
    print(f"  profile_path: {getattr(result, 'profile_path', None) or '-'}")
    print(f"  host: {result.host or '-'}")
    print(f"  port: {result.port if result.port is not None else '-'}")
    if result.command:
        print("  command: " + " ".join(shlex.quote(part) for part in result.command))
    if result.messages:
        print("  messages:")
        for message in result.messages:
            print(f"    - {message}")


def show_vllm_smoke_manage() -> None:
    print("\n  ── vLLM latest run status/log/stop ──")
    latest_result = latest_vllm_run_record()
    latest_record = latest_result.record if latest_result.ok else None
    if latest_record:
        print("  latest run record:")
        print_vllm_record_summary(latest_record)
    else:
        for message in latest_result.messages:
            print(f"  - {message}")
        print("  latest.json이 없거나 유효하지 않으면 수동 입력으로 진행합니다.")
    print("  [1] status")
    print("  [2] log")
    print("  [3] stop")
    choice = input("  선택 > ").strip()

    if choice == "1":
        record = prompt_vllm_run_record(latest_record)
        if record:
            pid = str(record.pid)
            run_id = record.run_id
            log_path = record.log_path
            preset_id = record.preset_id
            host = record.host
            port = record.port
        else:
            pid = input("  pid > ").strip()
            run_id = input("  run_id > ").strip()
            log_path = input("  log_path > ").strip()
            preset_id = future_launch_preset_id()
            host = input("  host [127.0.0.1] > ").strip() or "127.0.0.1"
            port = input("  port [8000] > ").strip() or "8000"
        result = check_vllm_run_status(pid=pid, run_id=run_id, log_path=log_path, preset_id=preset_id, host=host, port=port)
        print_vllm_status_result(result)
        return

    if choice == "2":
        record = prompt_vllm_run_record(latest_record)
        log_path = record.log_path if record else input("  log_path > ").strip()
        raw_lines = input("  last N lines [80] > ").strip()
        try:
            last_lines = int(raw_lines) if raw_lines else 80
        except ValueError:
            last_lines = 80
        result = read_vllm_run_log(log_path, last_lines=last_lines)
        print_vllm_log_result(result)
        return

    if choice == "3":
        record = prompt_vllm_run_record(latest_record)
        if record:
            pid = str(record.pid)
            run_id = record.run_id
            preset_id = record.preset_id
        else:
            pid = input("  pid > ").strip()
            run_id = input("  run_id > ").strip()
            preset_id = future_launch_preset_id()
        print("  계속하려면 stop 또는 STOP 를 정확히 입력하세요.")
        confirm = input("  confirmation > ").strip()
        result = stop_vllm_run(pid=pid, run_id=run_id, preset_id=preset_id, confirmed=(confirm.lower() == "stop"))
        print_vllm_stop_result(result)
        return

    print("  취소했습니다.")


def prompt_vllm_run_record(default_record: Any = None) -> Any:
    if default_record:
        record_path = input("  record_path [Enter=latest, manual=빈 값 대신 -] > ").strip()
        if not record_path:
            return default_record
        if record_path == "-":
            return None
    else:
        record_path = input("  record_path [manual 입력은 빈 값] > ").strip()
    if not record_path:
        return None
    result = read_vllm_run_record(record_path)
    if result.ok and result.record:
        return result.record
    for message in result.messages:
        print(f"  - {message}")
    return None


def print_vllm_record_summary(record: Any) -> None:
    print(f"    run_id: {record.run_id}")
    print(f"    pid: {record.pid}")
    print(f"    log_path: {record.log_path}")
    print(f"    host: {record.host}")
    print(f"    port: {record.port}")


def print_vllm_status_result(result: Any) -> None:
    print("\n  vLLM latest run status:")
    print(f"  ok: {result.ok}")
    print(f"  preset_id: {result.preset_id}")
    print(f"  pid: {result.pid if result.pid is not None else '-'}")
    print(f"  run_id: {result.run_id or '-'}")
    print(f"  log_path: {result.log_path or '-'}")
    print(f"  alive: {result.alive if result.alive is not None else '-'}")
    print(f"  log_exists: {result.log_exists}")
    print(f"  port_listening: {result.port_listening if result.port_listening is not None else '-'}")
    for message in result.messages:
        print(f"  - {message}")


def print_vllm_log_result(result: Any) -> None:
    print("\n  vLLM latest run log:")
    print(f"  ok: {result.ok}")
    print(f"  log_path: {result.log_path or '-'}")
    for message in result.messages:
        print(f"  - {message}")
    for line in result.lines:
        print(f"  {line}")


def print_vllm_stop_result(result: Any) -> None:
    print("\n  vLLM latest run stop:")
    print(f"  ok: {result.ok}")
    print(f"  preset_id: {result.preset_id}")
    print(f"  pid: {result.pid if result.pid is not None else '-'}")
    print(f"  run_id: {result.run_id or '-'}")
    for message in result.messages:
        print(f"  - {message}")


def show_vllm_api_smoke() -> None:
    print("\n  ── vLLM API 연결 테스트 ──")
    print("  Hermes와 무관한 OpenAI-compatible endpoint를 read-only로 확인합니다.")
    result = run_vllm_api_smoke()
    print_vllm_readiness_summary(
        api_status="PASS" if result.ok else "FAIL",
        hermes_chat_status="UNKNOWN",
        hermes_tool_status="NOT RUN",
    )
    print_vllm_api_smoke_result(result)


def print_vllm_api_smoke_result(result: Any) -> None:
    print("\n  vLLM API smoke result:")
    print(f"  ok: {result.ok}")
    print(f"  base_url: {result.base_url or '-'}")
    print(f"  model_id: {result.model_id or '-'}")
    for message in result.messages:
        print(f"  - {message}")
    for check in result.checks:
        status = "PASS" if check.ok else "FAIL"
        code = f" HTTP {check.status_code}" if check.status_code is not None else ""
        print(f"  [{status}] {check.name}{code}: {check.message}")


def initial_vllm_profile_selection_with_messages() -> tuple[Any, str, list[str]]:
    result = load_selected_vllm_profile_draft()
    if result.ok and result.profile and result.profile_id:
        return result.profile, result.profile_id, []
    if result.messages == ["no selected vLLM profile saved yet"]:
        return default_vllm_profile(), "custom-draft", []
    messages = ["selected vLLM profile could not be loaded; using custom-draft defaults"]
    messages.extend(result.messages)
    return default_vllm_profile(), "custom-draft", messages


def initial_vllm_profile_selection() -> tuple[Any, str]:
    profile, profile_id, _messages = initial_vllm_profile_selection_with_messages()
    return profile, profile_id


def choose_llama_cpp_menu_action(
    models: dict[str, str] | None = None,
    draft: dict[str, Any] | None = None,
    running: str | None = None,
) -> str:
    print("\n  ── llama.cpp workspace ──")
    print("  이 메뉴는 GGUF 모델과 llama.cpp 실행 흐름 전용입니다.")
    if models is not None and draft is not None:
        print_llama_cpp_model_list(models, draft, running)
        print()
    print("  [1] llama.cpp 설정/스크립트 불러오기")
    print("  [2] llama.cpp GGUF 모델 변경")
    print("  [3] llama.cpp 설정 변경 / 현재 설정 저장")
    print("  [4] llama.cpp 파라미터")
    print("  [5] llama.cpp 최종 미리보기")
    print("  [6] llama.cpp 1회 실행")
    print("  [7] llama.cpp 새 스크립트 생성")
    print("  [8] llama.cpp 스크립트 관리")
    choice = input("  선택 > ").strip()
    return {
        "1": "LOAD",
        "2": "M",
        "3": "A",
        "4": "K",
        "5": "P",
        "6": "O",
        "7": "G",
        "8": "S",
    }.get(choice, "")


def choose_vllm_menu_action(profile: Any = None, profile_id: str = "custom-draft", run_summary: Any = None) -> str:
    print("\n  ── vLLM workspace ──")
    print("  이 메뉴는 vLLM beta launch path와 OpenAI-compatible server 흐름 전용입니다.")
    if profile is not None:
        print("\n  Selected vLLM profile:")
        print(selected_vllm_profile_summary_line(profile, profile_id))
        print(selected_vllm_profile_path_line(profile_id))
    print("\n  Recent vLLM run:")
    print(recent_vllm_run_summary_line(run_summary))
    print("\n  [1] vLLM 검증 기준 profile 불러오기: Gemma4 26B AWQ")
    print("  [2] selected profile preview / preflight")
    print("  [3] launch selected vLLM profile")
    print("  [4] latest run status / log / stop")
    print("  [5] vLLM API 연결 테스트")
    print("  [6] Hermes 설정 동기화 preview/write")
    print("  [7] Hermes 단순 chat 테스트")
    print("  [8] Hermes tool-agent 테스트 / raw markup 검사")
    print("  [9] vLLM doctor")
    print("  [10] selected profile settings")
    print("  [A] advanced profile workspace / scripts / JSON")
    print("  [R] return")
    choice = input("  선택 > ").strip()
    return {
        "1": "VLLM_SELECT_GEMMA4_BETA",
        "2": "VLLM_SELECTED_PREVIEW",
        "3": "VLLM_SELECTED_LAUNCH",
        "4": "Z",
        "5": "W",
        "6": "HERMES_VLLM_SYNC",
        "7": "HERMES_VLLM_CHAT_SMOKE",
        "8": "HERMES_VLLM_TOOL_AGENT_SMOKE",
        "9": "VLLM_DOCTOR",
        "10": "VLLM_SELECTED_SETTINGS",
        "A": "B",
        "a": "B",
    }.get(choice, "")


# ─── 메인 루프 ─────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    models = get_model_list(MODELS_DIR)
    draft = draft_from_config(cfg, models)
    vllm_profile_draft, vllm_profile_draft_id, startup_warnings = initial_vllm_profile_selection_with_messages()

    while True:
        print_header()
        print("  llama.cpp / vLLM 로컬 AI 엔진 관제판")
        print(f"  모델 디렉터리: {MODELS_DIR}")
        print_working_draft_status(draft)
        print_integration_status(cfg)
        running = get_running_model()
        print_planned_run_summary(draft, running)
        vllm_run_summary = latest_vllm_run_summary()
        print_recent_vllm_run_summary(vllm_run_summary)
        print_selected_vllm_profile_summary(vllm_profile_draft, vllm_profile_draft_id)
        print_backend_workflow_bridge_hints()
        print_llama_cpp_model_summary(models, draft)
        print_startup_warnings(startup_warnings + recent_vllm_run_startup_warnings(vllm_run_summary))
        if not models:
            print(f"\n  ⚠️  {MODELS_DIR} 에서 GGUF 파일을 찾을 수 없습니다.")
            print("     그래도 [L] llama.cpp workspace, [I] 시스템 정보, [E] Hermes 등록, [C] OpenClaw 등록은 사용할 수 있습니다.")

        if running:
            print(f"  🔴 실행 중: {running}\n")

        numbered = list(enumerate(models.items(), 1))

        existing_scripts = list_scripts()
        script_info = f" ({len(existing_scripts)}개)" if existing_scripts else ""

        print("\n  [L] llama.cpp workspace")
        print("  [V] vLLM workspace")
        print(f"  [S] llama.cpp 스크립트 관리{script_info}")
        print("  [E] Hermes 등록/연동")
        print("  [C] OpenClaw 등록")
        print("  [H] 서버 상태 확인")
        print("  [I] 시스템 정보")
        print("  [T] no-thinking 채팅 테스트")
        print("  [R] 모델 목록 새로고침")
        print("  [Q] 취소\n")

        try:
            choice = input("  선택 > ").strip()
        except EOFError:
            print("\n👋 안녕!\n")
            break

        if not choice:
            continue

        upper = choice.upper()

        if upper == "Q":
            print("\n👋 안녕!\n")
            break

        if upper == "L":
            upper = choose_llama_cpp_menu_action(models, draft, running)
            if not upper:
                print("  취소했습니다.")
                pause()
                continue

        if upper == "V":
            upper = choose_vllm_menu_action(vllm_profile_draft, vllm_profile_draft_id, vllm_run_summary)
            if not upper:
                print("  취소했습니다.")
                pause()
                continue

        if upper == "LOAD":
            print("\n  [L] 불러오기")
            print("  [1] saved profile/config")
            print("  [2] existing generated script")
            print("  [3] last run record")
            print("  [4] defaults")
            sub = input("  선택 > ").strip()
            if sub == "1":
                draft = draft_from_config(cfg, models)
                print("  ✅ 저장된 profile/config에서 현재 작업 설정을 불러왔습니다.")
            elif sub == "2":
                selected = select_script_path()
                if selected:
                    ok, message = load_script_into_draft(selected, draft)
                    print("  " + ("✅ " if ok else "⚠️  ") + message)
            elif sub == "3":
                ok, message = load_last_run_record(models, draft)
                print("  " + ("✅ " if ok else "⚠️  ") + message)
            elif sub == "4":
                draft = draft_from_config(load_config(), models)
                draft["dirty"] = True
                draft["loaded_from"] = "defaults"
                draft["status"] = "기본값에서 불러온 임시 작업 설정입니다."
                print("  ✅ 기본값을 현재 작업 설정으로 불러왔습니다.")
            pause()
            continue

        if upper == "M":
            if not numbered:
                print("  ⚠️  선택할 모델이 없습니다. 모델 디렉터리를 확인한 뒤 [R] 모델 목록 새로고침을 선택하세요.")
                pause()
                continue
            choice = input("  모델 번호 또는 검색어 > ").strip()
            upper = ""

        if upper == "A":
            cfg, draft = settings_menu(cfg, draft)
            continue

        if upper == "K":
            before = copy.deepcopy(draft)
            draft = edit_parameters(draft)
            if draft != before:
                draft["dirty"] = True
                draft["status"] = "파라미터 변경으로 생긴 임시 작업 설정입니다."
            pause()
            continue

        if upper == "B":
            before_vllm_profile_id = vllm_profile_draft_id
            vllm_profile_draft, vllm_profile_draft_id = show_vllm_profile_menu(
                vllm_profile_draft,
                vllm_profile_draft_id,
                return_profile_id=True,
            )
            if vllm_profile_draft_id != before_vllm_profile_id:
                result = save_selected_vllm_profile_id(vllm_profile_draft_id)
                if not result.ok:
                    print_vllm_selected_profile_result(result)
            pause()
            continue

        if upper == "VLLM_SELECT_GEMMA4_BETA":
            result = save_verified_gemma4_26b_awq_beta_profile()
            print_vllm_profile_selection_result(result)
            if result.ok and result.profile:
                vllm_profile_draft = result.profile
                vllm_profile_draft_id = result.profile_id
            pause()
            continue

        if upper == "VLLM_SELECTED_PREVIEW":
            selected = load_selected_vllm_profile_draft()
            if selected.ok and selected.profile and selected.profile_id:
                vllm_profile_draft = selected.profile
                vllm_profile_draft_id = selected.profile_id
            else:
                print_vllm_profile_store_result(selected)
            show_vllm_selected_profile_preview(vllm_profile_draft, vllm_profile_draft_id)
            pause()
            continue

        if upper == "VLLM_SELECTED_LAUNCH":
            selected = load_selected_vllm_profile_draft()
            if selected.ok and selected.profile and selected.profile_id:
                vllm_profile_draft = selected.profile
                vllm_profile_draft_id = selected.profile_id
            else:
                print_vllm_profile_store_result(selected)
                pause()
                continue
            vllm_profile_draft, vllm_profile_draft_id = show_vllm_custom_launch(vllm_profile_draft, vllm_profile_draft_id)
            pause()
            continue

        if upper == "VLLM_SELECTED_SETTINGS":
            selected = load_selected_vllm_profile_draft()
            if selected.ok and selected.profile and selected.profile_id:
                vllm_profile_draft = selected.profile
                vllm_profile_draft_id = selected.profile_id
            else:
                print_vllm_profile_store_result(selected)
            vllm_profile_draft, vllm_profile_draft_id = show_vllm_selected_profile_settings(
                vllm_profile_draft,
                vllm_profile_draft_id,
            )
            pause()
            continue

        if upper == "W":
            show_vllm_api_smoke()
            pause()
            continue

        if upper == "Y":
            show_vllm_smoke_launch_once()
            pause()
            continue

        if upper == "Z":
            show_vllm_smoke_manage()
            pause()
            continue

        if upper == "HERMES_VLLM_SYNC":
            cfg = show_hermes_vllm_sync_menu(cfg)
            pause()
            continue

        if upper == "HERMES_VLLM_CHAT_SMOKE":
            show_hermes_vllm_chat_smoke()
            pause()
            continue

        if upper == "HERMES_VLLM_TOOL_AGENT_SMOKE":
            show_hermes_vllm_tool_agent_smoke()
            pause()
            continue

        if upper == "P":
            print()
            print(final_preview_text(draft))
            pause()
            continue

        if upper == "O":
            if not draft.get("model_name") or not draft.get("model_path"):
                print("  ⚠️  모델이 선택되지 않았습니다. [M] 모델 변경을 먼저 선택하세요.")
                pause()
                continue
            if not confirm_final_preview(draft, "[O] 1회 실행"):
                continue
            ok, message = write_last_run_record(draft, "one_time_run")
            print("  " + ("✅ " if ok else "⚠️  ") + message)
            with tempfile.TemporaryDirectory(prefix="llama-suite-once-") as directory:
                _script_name, script_path = generate_script(str(draft["model_name"]), str(draft["model_path"]), draft, scripts_dir=directory)
                run_script(script_path, model_name=str(draft["model_name"]))
            pause()
            continue

        if upper == "G":
            if not draft.get("model_name") or not draft.get("model_path"):
                print("  ⚠️  모델이 선택되지 않았습니다. [M] 모델 변경을 먼저 선택하세요.")
                pause()
                continue
            action = choose_script_generation_action(draft)
            if not action:
                continue
            script_name, script_path = generate_script(str(draft["model_name"]), str(draft["model_path"]), draft)
            print(f"  📝 새 실행 스냅샷 생성됨: {script_name}")
            print(f"     {script_path}")
            if action == "create_and_run":
                run_script(script_path, model_name=str(draft["model_name"]))
            pause()
            continue

        if upper == "S":
            manage_scripts(draft)
            continue

        if upper == "E":
            cfg = show_hermes_integration_menu(cfg)
            pause()
            continue

        if upper == "C":
            cfg = register_config_path(
                cfg,
                "openclaw_config",
                "OpenClaw",
                OPENCLAW_CONFIG_CANDIDATES,
                require_writable=False,
                read_only=True,
            )
            pause()
            continue

        if upper == "H":
            show_status(draft, get_running_servers())
            pause()
            continue

        if upper == "I":
            show_system_info()
            pause()
            continue

        if upper == "VLLM_DOCTOR":
            show_vllm_doctor()
            pause()
            continue

        if upper == "T":
            quick_no_think_test(draft)
            pause()
            continue

        if upper == "R":
            models = get_model_list(MODELS_DIR)
            if draft.get("model_name") in models:
                draft["model_path"] = models[draft["model_name"]]
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

        draft["model_name"] = model_name
        draft["model_path"] = model_path
        draft["dirty"] = True
        draft["loaded_from"] = "model selection"
        draft["status"] = "모델 변경으로 생긴 임시 작업 설정입니다."

        profiles = load_profiles()
        profile = profiles.get("models", {}).get(model_name) or default_model_profile(model_name, model_path)

        result = get_latest_script(model_name)
        existing_script, existing_name = (result if result else (None, None))
        effective_ctx_size = resolve_ctx_size(model_name, model_path, draft)
        draft["ctx_size"] = effective_ctx_size

        print(f"\n  📦 모델 : {model_name}")
        print(f"  📄 경로 : {model_path}")
        print(f"  ⚙️  설정 : ctx={effective_ctx_size}, {draft['host']}:{draft['port']}")
        print(
            f"  🧾 profile: ctx={profile.get('stable_ctx_size') or 'unknown'}, "
            f"backend={profile.get('recommended_backend') or 'unknown'}, "
            f"tool={profile.get('hermes_tool_call', {}).get('status') or 'unknown'}"
        )
        print(
            f"  🧠 reasoning={draft.get('reasoning')}, "
            f"budget={draft.get('reasoning_budget')}, "
            f"enable_thinking={draft.get('enable_thinking')}"
        )

        if existing_script:
            status = "modern" if script_is_modern(existing_script) else "old"
            print(f"  📝 기존 스크립트: {existing_name} ({status})")
            print("  기존 스크립트를 수정하려면 [S] 스크립트 관리 → [3] 현재 설정으로 불러오기를 사용하세요.")
        print("  실행하려면 메인 화면에서 [O] 1회 실행 또는 [G] 새 스크립트 생성 → [2] 생성 후 실행을 선택하세요.")
        pause()


if __name__ == "__main__":
    main()
