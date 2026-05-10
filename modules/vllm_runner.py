from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.vllm_profiles import (
    FUTURE_LAUNCH_PRESET_ID,
    VllmProfile,
    build_vllm_command,
    smoke_vllm_profile,
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


@dataclass(frozen=True)
class VllmLaunchResult:
    ok: bool
    pid: int | None
    run_id: str | None
    log_path: str | None
    host: str | None
    port: int | None
    preset_id: str
    command: list[str]
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


def launch_vllm_smoke_once(
    *,
    confirmed: bool,
    timestamp: str | None = None,
    state_root: str | Path | None = None,
    port_check: Any = None,
    popen_factory: Any = None,
) -> VllmLaunchResult:
    preset_id = FUTURE_LAUNCH_PRESET_ID
    if not confirmed:
        return VllmLaunchResult(
            ok=False,
            pid=None,
            run_id=None,
            log_path=None,
            host=None,
            port=None,
            preset_id=preset_id,
            command=[],
            messages=["launch cancelled: explicit confirmation is required"],
        )

    readiness = build_vllm_launch_plan(
        smoke_vllm_profile(),
        preset_id=preset_id,
        timestamp=timestamp,
        state_root=state_root,
        port_check=port_check,
    )
    if not readiness.ok or readiness.plan is None:
        return VllmLaunchResult(
            ok=False,
            pid=None,
            run_id=None,
            log_path=None,
            host=None,
            port=None,
            preset_id=preset_id,
            command=[],
            messages=readiness.messages,
        )

    plan = readiness.plan
    log_path = Path(plan.log_path).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(plan.env_preview)
        popen = popen_factory or subprocess.Popen
        with log_path.open("ab") as log_file:
            process = popen(
                plan.command,
                stdout=log_file,
                stderr=log_file,
                env=env,
                start_new_session=True,
            )
    except Exception as exc:
        return _failed_launch_result(plan, f"launch failed: {exc}")

    return VllmLaunchResult(
        ok=True,
        pid=int(process.pid),
        run_id=plan.run_id,
        log_path=plan.log_path,
        host=plan.host,
        port=plan.port,
        preset_id=plan.preset_id,
        command=plan.command,
        messages=["vLLM smoke preset launch started"],
    )


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


def _failed_launch_result(plan: VllmLaunchPlan, message: str) -> VllmLaunchResult:
    return VllmLaunchResult(
        ok=False,
        pid=None,
        run_id=plan.run_id,
        log_path=plan.log_path,
        host=plan.host,
        port=plan.port,
        preset_id=plan.preset_id,
        command=plan.command,
        messages=[message],
    )
