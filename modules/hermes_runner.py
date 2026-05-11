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
HERMES_SMOKE_KIND_CHAT = "chat"
HERMES_SMOKE_KIND_TOOL_AGENT = "tool_agent"
RAW_MARKUP_PATTERNS = (
    "<|channel>",
    "<|tool_call>",
    "<tool_call|>",
    "call:terminal",
    "call:execute_code",
    "hidden reasoning",
    "chain of thought",
)


@dataclass(frozen=True)
class HermesSmokePlan:
    ok: bool
    command: list[str]
    base_url: str | None
    model_id: str | None
    messages: list[str]
    smoke_kind: str = HERMES_SMOKE_KIND_CHAT


@dataclass(frozen=True)
class HermesSmokeResult:
    ok: bool
    command: list[str]
    stdout: str
    stderr: str
    returncode: int | None
    messages: list[str]
    smoke_kind: str = HERMES_SMOKE_KIND_CHAT
    status: str = "not_run"
    raw_markup_detected: bool = False
    raw_markup_patterns: list[str] | None = None


def detect_agent_raw_markup(text: str) -> list[str]:
    lowered = str(text or "").lower()
    found: list[str] = []
    for pattern in RAW_MARKUP_PATTERNS:
        if pattern.lower() in lowered:
            found.append(pattern)
    return found


def has_agent_raw_markup(text: str) -> bool:
    return bool(detect_agent_raw_markup(text))


def build_hermes_vllm_smoke_plan(
    *,
    hermes_bin: str = DEFAULT_HERMES_BIN,
    latest_record: Any = None,
    status_check: Any = None,
    prompt: str = DEFAULT_HERMES_SMOKE_PROMPT,
) -> HermesSmokePlan:
    hermes_path = str(Path(hermes_bin).expanduser())
    if not Path(hermes_path).is_file():
        return HermesSmokePlan(False, [], None, None, [f"Hermes executable does not exist: {hermes_path}"], HERMES_SMOKE_KIND_CHAT)
    if not os.access(hermes_path, os.X_OK):
        return HermesSmokePlan(False, [], None, None, [f"Hermes executable is not executable: {hermes_path}"], HERMES_SMOKE_KIND_CHAT)

    latest = latest_record or latest_vllm_run_record()
    if not getattr(latest, "ok", False) or latest.record is None:
        return HermesSmokePlan(False, [], None, None, ["latest vLLM run record is not available"] + list(getattr(latest, "messages", [])), HERMES_SMOKE_KIND_CHAT)

    summary = latest_vllm_run_summary(latest_record=latest, status_check=status_check)
    if summary.status != "READY":
        return HermesSmokePlan(False, [], summary.endpoint, None, [f"latest vLLM run is not READY: {summary.status}"] + summary.messages, HERMES_SMOKE_KIND_CHAT)

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
    return HermesSmokePlan(True, command, summary.endpoint, model_id, ["Hermes chat smoke plan built"], HERMES_SMOKE_KIND_CHAT)


def build_hermes_vllm_tool_agent_smoke_plan() -> HermesSmokePlan:
    return HermesSmokePlan(
        False,
        [],
        None,
        None,
        ["Hermes tool-agent smoke is unsupported: safe tool-agent probe is not implemented yet"],
        HERMES_SMOKE_KIND_TOOL_AGENT,
    )


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
        return HermesSmokeResult(False, plan.command, "", "", None, plan.messages, HERMES_SMOKE_KIND_CHAT, "not_run", False, [])
    if not confirmed:
        return HermesSmokeResult(False, plan.command, "", "", None, ["Hermes chat smoke cancelled: explicit confirmation is required"], HERMES_SMOKE_KIND_CHAT, "not_run", False, [])

    run = runner or subprocess.run
    try:
        completed = run(plan.command, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return HermesSmokeResult(False, plan.command, "", "", None, [f"Hermes chat smoke failed: {exc}"], HERMES_SMOKE_KIND_CHAT, "fail", False, [])

    stdout = str(getattr(completed, "stdout", ""))
    stderr = str(getattr(completed, "stderr", ""))
    returncode = int(getattr(completed, "returncode", 1))
    raw_patterns = detect_agent_raw_markup(stdout + "\n" + stderr)
    raw_markup_detected = bool(raw_patterns)
    ok = returncode == 0 and "llama-suite-ok" in stdout.lower() and not raw_markup_detected
    status = "pass" if ok else "fail"
    messages = ["Hermes chat smoke completed" if ok else "Hermes chat smoke did not return expected marker"]
    if raw_markup_detected:
        messages.append("raw tool-call markup leaked")
    return HermesSmokeResult(ok, plan.command, stdout, stderr, returncode, messages, HERMES_SMOKE_KIND_CHAT, status, raw_markup_detected, raw_patterns)


def run_hermes_vllm_tool_agent_smoke(*, confirmed: bool = False) -> HermesSmokeResult:
    plan = build_hermes_vllm_tool_agent_smoke_plan()
    return HermesSmokeResult(
        False,
        plan.command,
        "",
        "",
        None,
        plan.messages,
        HERMES_SMOKE_KIND_TOOL_AGENT,
        "unsupported",
        False,
        [],
    )
