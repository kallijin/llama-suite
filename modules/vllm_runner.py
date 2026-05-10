from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.vllm_profiles import (
    FUTURE_LAUNCH_PRESET_ID,
    VllmProfile,
    build_vllm_command,
    run_vllm_preflight,
    validate_vllm_profile,
)


DEFAULT_VLLM_RUN_ROOT = "~/.local/state/llama-suite/runs/vllm"


@dataclass(frozen=True)
class VllmLaunchPlan:
    preset_id: str
    run_id: str
    command: list[str]
    env_preview: dict[str, str]
    log_path: str
    host: str
    port: int


@dataclass(frozen=True)
class VllmLaunchReadiness:
    ok: bool
    plan: VllmLaunchPlan | None
    messages: list[str]


def build_vllm_launch_environment_preview(profile: VllmProfile) -> dict[str, str]:
    return {
        "VLLM_CACHE_ROOT": str(profile.vllm_cache_root),
        "HF_HOME": str(profile.hf_home),
        "TRANSFORMERS_CACHE": str(profile.transformers_cache),
    }


def build_vllm_launch_plan(
    profile: VllmProfile,
    *,
    preset_id: str = FUTURE_LAUNCH_PRESET_ID,
    timestamp: str | None = None,
    state_root: str | Path | None = None,
    port_check: Any = None,
) -> VllmLaunchReadiness:
    validation_messages = validate_vllm_profile(profile)
    if validation_messages:
        return VllmLaunchReadiness(False, None, validation_messages)

    command, command_messages = build_vllm_command(profile)
    if command is None:
        return VllmLaunchReadiness(False, None, command_messages)

    preflight = run_vllm_preflight(profile, port_check=port_check)
    if not preflight.ok:
        messages = [f"{check.name}: {check.message}" for check in preflight.checks if not check.ok]
        return VllmLaunchReadiness(False, None, messages)

    launch_timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_preset_id = sanitize_run_id_part(preset_id)
    run_id = f"vllm-{safe_preset_id}-{launch_timestamp}"
    root = Path(state_root or DEFAULT_VLLM_RUN_ROOT).expanduser()
    log_path = str(root / f"{run_id}.log")

    plan = VllmLaunchPlan(
        preset_id=preset_id,
        run_id=run_id,
        command=command,
        env_preview=build_vllm_launch_environment_preview(profile),
        log_path=log_path,
        host=str(profile.host),
        port=int(profile.port),
    )
    return VllmLaunchReadiness(True, plan, [])


def sanitize_run_id_part(value: Any) -> str:
    text = str(value or "").strip()
    safe_chars = []
    for char in text:
        if char.isalnum() or char in "._-":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._-")
    return sanitized or "vllm"
