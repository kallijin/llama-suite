from __future__ import annotations

import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.hermes_integration import served_model_id_from_vllm_record
from modules.vllm_runner import latest_vllm_run_record, latest_vllm_run_summary


DEFAULT_HERMES_BIN = "~/.local/bin/hermes"
DEFAULT_HERMES_SMOKE_PROMPT = "Reply with exactly: llama-suite-ok"


@dataclass(frozen=True)
class HermesSmokePlan:
    ok: bool
    command: list[str]
    base_url: str | None
    model_id: str | None
    messages: list[str]


@dataclass(frozen=True)
class HermesSmokeResult:
    ok: bool
    command: list[str]
    stdout: str
    stderr: str
    returncode: int | None
    messages: list[str]


def build_hermes_vllm_smoke_plan(
    *,
    hermes_bin: str = DEFAULT_HERMES_BIN,
    latest_record: Any = None,
    status_check: Any = None,
    prompt: str = DEFAULT_HERMES_SMOKE_PROMPT,
) -> HermesSmokePlan:
    hermes_path = str(Path(hermes_bin).expanduser())
    if not Path(hermes_path).is_file():
        return HermesSmokePlan(False, [], None, None, [f"Hermes executable does not exist: {hermes_path}"])
    if not os.access(hermes_path, os.X_OK):
        return HermesSmokePlan(False, [], None, None, [f"Hermes executable is not executable: {hermes_path}"])

    latest = latest_record or latest_vllm_run_record()
    if not getattr(latest, "ok", False) or latest.record is None:
        return HermesSmokePlan(False, [], None, None, ["latest vLLM run record is not available"] + list(getattr(latest, "messages", [])))

    summary = latest_vllm_run_summary(latest_record=latest, status_check=status_check)
    if summary.status != "READY":
        return HermesSmokePlan(False, [], summary.endpoint, None, [f"latest vLLM run is not READY: {summary.status}"] + summary.messages)

    model_id = served_model_id_from_vllm_record(latest.record)
    command = [
        hermes_path,
        "chat",
        "-q",
        prompt,
        "-Q",
        "--provider",
        "custom",
        "--model",
        model_id,
        "--max-turns",
        "1",
        "--source",
        "llama-suite",
    ]
    return HermesSmokePlan(True, command, summary.endpoint, model_id, ["Hermes vLLM smoke plan built"])


def run_hermes_vllm_smoke(
    *,
    confirmed: bool,
    hermes_bin: str = DEFAULT_HERMES_BIN,
    latest_record: Any = None,
    status_check: Any = None,
    runner: Any = None,
    timeout: float = 120.0,
) -> HermesSmokeResult:
    plan = build_hermes_vllm_smoke_plan(hermes_bin=hermes_bin, latest_record=latest_record, status_check=status_check)
    if not plan.ok:
        return HermesSmokeResult(False, plan.command, "", "", None, plan.messages)
    if not confirmed:
        return HermesSmokeResult(False, plan.command, "", "", None, ["Hermes smoke cancelled: explicit confirmation is required"])

    run = runner or subprocess.run
    try:
        completed = run(plan.command, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return HermesSmokeResult(False, plan.command, "", "", None, [f"Hermes smoke failed: {exc}"])

    stdout = str(getattr(completed, "stdout", ""))
    stderr = str(getattr(completed, "stderr", ""))
    returncode = int(getattr(completed, "returncode", 1))
    ok = returncode == 0 and "llama-suite-ok" in stdout.lower()
    messages = ["Hermes smoke completed" if ok else "Hermes smoke did not return expected marker"]
    return HermesSmokeResult(ok, plan.command, stdout, stderr, returncode, messages)
