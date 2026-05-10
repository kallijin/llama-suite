from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
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
    record_path: str | None = None


@dataclass(frozen=True)
class VllmSmokeStatusResult:
    ok: bool
    pid: int | None
    run_id: str
    log_path: str
    preset_id: str
    alive: bool | None
    log_exists: bool
    port_listening: bool | None
    messages: list[str]


@dataclass(frozen=True)
class VllmSmokeLogResult:
    ok: bool
    log_path: str
    lines: list[str]
    messages: list[str]


@dataclass(frozen=True)
class VllmSmokeStopResult:
    ok: bool
    pid: int | None
    run_id: str
    preset_id: str
    messages: list[str]


@dataclass(frozen=True)
class VllmRunRecord:
    backend: str
    preset_id: str
    run_id: str
    pid: int
    log_path: str
    host: str
    port: int
    command: list[str]
    started_at: str


@dataclass(frozen=True)
class VllmRunRecordResult:
    ok: bool
    record: VllmRunRecord | None
    record_path: str | None
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

    record_result = write_vllm_run_record(
        VllmRunRecord(
            backend="vllm",
            preset_id=plan.preset_id,
            run_id=plan.run_id,
            pid=int(process.pid),
            log_path=plan.log_path,
            host=plan.host,
            port=plan.port,
            command=plan.command,
            started_at=datetime.now().isoformat(timespec="seconds"),
        ),
        state_root=log_path.parent,
    )
    messages = ["vLLM smoke preset launch started"]
    messages.extend(record_result.messages)

    return VllmLaunchResult(
        ok=True,
        pid=int(process.pid),
        run_id=plan.run_id,
        log_path=plan.log_path,
        host=plan.host,
        port=plan.port,
        preset_id=plan.preset_id,
        command=plan.command,
        messages=messages,
        record_path=record_result.record_path,
    )


def vllm_run_record_path(run_id: str, *, state_root: str | Path | None = None) -> str:
    root = Path(state_root or DEFAULT_VLLM_RUN_ROOT).expanduser()
    return str(root / f"{sanitize_run_id_part(run_id)}.json")


def write_vllm_run_record(record: VllmRunRecord, *, state_root: str | Path | None = None) -> VllmRunRecordResult:
    record_path = vllm_run_record_path(record.run_id, state_root=state_root)
    try:
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        return VllmRunRecordResult(False, record, record_path, [f"run record write failed: {exc}"])
    return VllmRunRecordResult(True, record, record_path, [f"run record saved: {record_path}"])


def read_vllm_run_record(record_path: str) -> VllmRunRecordResult:
    expanded_path = str(Path(record_path).expanduser()) if record_path else ""
    if not expanded_path:
        return VllmRunRecordResult(False, None, None, ["run record path is required"])
    try:
        data = json.loads(Path(expanded_path).read_text())
        record = VllmRunRecord(
            backend=str(data.get("backend", "")),
            preset_id=str(data.get("preset_id", "")),
            run_id=str(data.get("run_id", "")),
            pid=int(data.get("pid", 0)),
            log_path=str(data.get("log_path", "")),
            host=str(data.get("host", "")),
            port=int(data.get("port", 0)),
            command=[str(part) for part in data.get("command", [])],
            started_at=str(data.get("started_at", "")),
        )
    except Exception as exc:
        return VllmRunRecordResult(False, None, expanded_path, [f"run record read failed: {exc}"])
    return VllmRunRecordResult(True, record, expanded_path, [])


def latest_vllm_run_record(*, state_root: str | Path | None = None) -> VllmRunRecordResult:
    root = Path(state_root or DEFAULT_VLLM_RUN_ROOT).expanduser()
    try:
        records = sorted(root.glob("vllm-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except Exception as exc:
        return VllmRunRecordResult(False, None, None, [f"run record lookup failed: {exc}"])
    if not records:
        return VllmRunRecordResult(False, None, None, [f"no vLLM run records found under {root}"])
    return read_vllm_run_record(str(records[0]))


def check_vllm_smoke_status(
    *,
    pid: int | str | None,
    run_id: str = "",
    log_path: str = "",
    alive_check: Any = None,
    port_check: Any = None,
) -> VllmSmokeStatusResult:
    pid_number = _coerce_pid(pid)
    messages: list[str] = []
    alive: bool | None = None
    if pid_number is None:
        messages.append("pid is required")
    else:
        try:
            alive = bool((alive_check or _pid_is_alive)(pid_number))
            messages.append("process is alive" if alive else "process is not alive")
        except Exception as exc:
            messages.append(f"process status check failed: {exc}")

    expanded_log_path = str(Path(log_path).expanduser()) if log_path else ""
    log_exists = bool(expanded_log_path and Path(expanded_log_path).is_file())
    messages.append(f"log file {'exists' if log_exists else 'does not exist'}: {expanded_log_path or '-'}")

    port_listening: bool | None = None
    if port_check is not None:
        try:
            port_result = port_check("127.0.0.1", 8000)
            port_listening = bool(getattr(port_result, "ok", port_result))
            detail = getattr(port_result, "message", "")
            messages.append(f"port 8000 listening: {port_listening}" + (f" ({detail})" if detail else ""))
        except Exception as exc:
            messages.append(f"port check failed: {exc}")

    ok = pid_number is not None and alive is not None
    return VllmSmokeStatusResult(
        ok=ok,
        pid=pid_number,
        run_id=str(run_id or ""),
        log_path=expanded_log_path,
        preset_id=FUTURE_LAUNCH_PRESET_ID,
        alive=alive,
        log_exists=log_exists,
        port_listening=port_listening,
        messages=messages,
    )


def read_vllm_smoke_log(log_path: str, *, last_lines: int = 80) -> VllmSmokeLogResult:
    expanded_log_path = str(Path(log_path).expanduser()) if log_path else ""
    if not expanded_log_path:
        return VllmSmokeLogResult(False, "", [], ["log path is required"])

    path = Path(expanded_log_path)
    if not path.is_file():
        return VllmSmokeLogResult(False, expanded_log_path, [], [f"log file does not exist: {expanded_log_path}"])

    try:
        line_count = max(0, int(last_lines))
        lines = path.read_text(errors="replace").splitlines()
    except Exception as exc:
        return VllmSmokeLogResult(False, expanded_log_path, [], [f"log read failed: {exc}"])

    return VllmSmokeLogResult(True, expanded_log_path, lines[-line_count:] if line_count else [], [])


def stop_vllm_smoke(
    *,
    pid: int | str | None,
    confirmed: bool,
    run_id: str = "",
    getpgid_func: Any = None,
    killpg_func: Any = None,
) -> VllmSmokeStopResult:
    pid_number = _coerce_pid(pid)
    if not confirmed:
        return VllmSmokeStopResult(
            False,
            pid_number,
            str(run_id or ""),
            FUTURE_LAUNCH_PRESET_ID,
            ["stop cancelled: explicit confirmation is required"],
        )
    if pid_number is None:
        return VllmSmokeStopResult(False, None, str(run_id or ""), FUTURE_LAUNCH_PRESET_ID, ["pid is required"])

    getpgid = getpgid_func or os.getpgid
    killpg = killpg_func or os.killpg
    try:
        pgid = getpgid(pid_number)
        killpg(pgid, signal.SIGTERM)
    except Exception as exc:
        return VllmSmokeStopResult(
            False,
            pid_number,
            str(run_id or ""),
            FUTURE_LAUNCH_PRESET_ID,
            [f"stop failed: {exc}"],
        )

    return VllmSmokeStopResult(
        True,
        pid_number,
        str(run_id or ""),
        FUTURE_LAUNCH_PRESET_ID,
        [f"SIGTERM sent to process group for pid {pid_number}"],
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


def _coerce_pid(value: int | str | None) -> int | None:
    try:
        pid = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
