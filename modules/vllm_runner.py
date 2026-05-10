from __future__ import annotations

import json
import os
import signal
import socket
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
from modules.vllm_profile_store import default_vllm_profile_path


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
    profile_id: str | None = None
    profile_path: str | None = None
    profile_snapshot: dict[str, Any] | None = None


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
    profile_path: str | None = None


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
    command: list[str]
    env_preview: dict[str, str]
    log_path: str
    host: str
    port: int
    started_at: str
    status_hint: str
    profile_id: str | None = None
    profile_path: str | None = None
    profile_snapshot: dict[str, Any] | None = None
    schema: str = "llama-suite.run.v1"


@dataclass(frozen=True)
class VllmRunRecordResult:
    ok: bool
    record: VllmRunRecord | None
    record_path: str | None
    messages: list[str]


@dataclass(frozen=True)
class VllmLatestRunSummary:
    ok: bool
    preset_id: str | None
    model: str | None
    endpoint: str | None
    status: str
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
    profile_path: str | Path | None = None,
    include_profile_snapshot: bool = False,
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
        profile_id=preset_id,
        profile_path=str(Path(profile_path).expanduser()) if profile_path else None,
        profile_snapshot=profile.to_dict() if include_profile_snapshot else None,
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

    return _launch_vllm_plan(readiness.plan, popen_factory=popen_factory, started_message="vLLM smoke preset launch started")


def launch_vllm_profile_once(
    profile: VllmProfile,
    *,
    confirmed: bool,
    preset_id: str = "custom-draft",
    timestamp: str | None = None,
    state_root: str | Path | None = None,
    port_check: Any = None,
    popen_factory: Any = None,
    profile_path: str | Path | None = None,
) -> VllmLaunchResult:
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
        profile,
        preset_id=preset_id,
        timestamp=timestamp,
        state_root=state_root,
        port_check=port_check,
        profile_path=profile_path or default_vllm_profile_path(preset_id),
        include_profile_snapshot=True,
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

    return _launch_vllm_plan(readiness.plan, popen_factory=popen_factory, started_message="vLLM custom profile launch started")


def _launch_vllm_plan(
    plan: VllmLaunchPlan,
    *,
    popen_factory: Any = None,
    started_message: str,
) -> VllmLaunchResult:
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
            command=plan.command,
            env_preview=plan.env_preview,
            log_path=plan.log_path,
            host=plan.host,
            port=plan.port,
            started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            status_hint="started",
            profile_id=plan.profile_id,
            profile_path=plan.profile_path,
            profile_snapshot=plan.profile_snapshot,
        ),
        state_root=log_path.parent,
    )
    messages = [started_message]
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
        profile_path=plan.profile_path,
    )


def vllm_run_record_path(run_id: str, *, state_root: str | Path | None = None) -> str:
    root = Path(state_root or DEFAULT_VLLM_RUN_ROOT).expanduser()
    return str(root / f"{sanitize_run_id_part(run_id)}.json")


def write_vllm_run_record(record: VllmRunRecord, *, state_root: str | Path | None = None) -> VllmRunRecordResult:
    record_path = vllm_run_record_path(record.run_id, state_root=state_root)
    try:
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(record), indent=2, sort_keys=True) + "\n"
        _atomic_write_text(path, payload)
        _atomic_write_text(path.parent / "latest.json", payload)
    except Exception as exc:
        return VllmRunRecordResult(False, record, record_path, [f"run record write failed: {exc}"])
    return VllmRunRecordResult(True, record, record_path, [f"run record saved: {record_path}", f"latest run record updated: {path.parent / 'latest.json'}"])


def read_vllm_run_record(record_path: str) -> VllmRunRecordResult:
    expanded_path = str(Path(record_path).expanduser()) if record_path else ""
    if not expanded_path:
        return VllmRunRecordResult(False, None, None, ["run record path is required"])
    try:
        data = json.loads(Path(expanded_path).read_text())
        record = VllmRunRecord(
            schema=str(data.get("schema", "")),
            backend=str(data.get("backend", "")),
            preset_id=str(data.get("preset_id", "")),
            run_id=str(data.get("run_id", "")),
            pid=int(data.get("pid", 0)),
            command=[str(part) for part in data.get("command", [])],
            env_preview={str(key): str(value) for key, value in dict(data.get("env_preview", {})).items()},
            log_path=str(data.get("log_path", "")),
            host=str(data.get("host", "")),
            port=int(data.get("port", 0)),
            started_at=str(data.get("started_at", "")),
            status_hint=str(data.get("status_hint", "")),
            profile_id=str(data.get("profile_id")) if data.get("profile_id") is not None else None,
            profile_path=str(data.get("profile_path")) if data.get("profile_path") is not None else None,
            profile_snapshot=dict(data.get("profile_snapshot")) if isinstance(data.get("profile_snapshot"), dict) else None,
        )
    except Exception as exc:
        return VllmRunRecordResult(False, None, expanded_path, [f"run record read failed: {exc}"])
    validation_messages = validate_vllm_run_record(record)
    if validation_messages:
        return VllmRunRecordResult(False, record, expanded_path, validation_messages)
    return VllmRunRecordResult(True, record, expanded_path, [])


def validate_vllm_run_record(record: VllmRunRecord) -> list[str]:
    messages: list[str] = []
    if record.schema != "llama-suite.run.v1":
        messages.append("invalid run record schema")
    if record.backend != "vllm":
        messages.append("invalid run record backend")
    if not record.run_id:
        messages.append("run_id is required")
    if record.pid <= 0:
        messages.append("pid should be a positive integer")
    if not record.log_path:
        messages.append("log_path is required")
    if not record.host:
        messages.append("host is required")
    if record.port <= 0:
        messages.append("port should be a positive integer")
    return messages


def latest_vllm_run_record(*, state_root: str | Path | None = None) -> VllmRunRecordResult:
    root = Path(state_root or DEFAULT_VLLM_RUN_ROOT).expanduser()
    latest_path = root / "latest.json"
    if latest_path.is_file():
        return read_vllm_run_record(str(latest_path))
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
    host: str = "127.0.0.1",
    port: int | str = 8000,
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
    check_port = port_check or check_tcp_listening
    if host and port:
        try:
            port_result = check_port(host, port)
            port_listening = bool(getattr(port_result, "ok", port_result))
            detail = getattr(port_result, "message", "")
            messages.append(f"port {port} listening on {host}: {port_listening}" + (f" ({detail})" if detail else ""))
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


def check_tcp_listening(host: str, port: int | str, timeout: float = 1.0) -> Any:
    port_number = _coerce_port(port)
    if port_number is None:
        return VllmSmokeStatusPortCheck(False, "port should be a positive integer")
    host_text = str(host or "").strip()
    if not host_text:
        return VllmSmokeStatusPortCheck(False, "host is required")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            code = sock.connect_ex((host_text, port_number))
    except Exception as exc:
        return VllmSmokeStatusPortCheck(False, f"listening check failed: {exc}")

    if code == 0:
        return VllmSmokeStatusPortCheck(True, f"{host_text}:{port_number} is listening")
    return VllmSmokeStatusPortCheck(False, f"{host_text}:{port_number} is not listening")


def check_vllm_run_status(**kwargs: Any) -> VllmSmokeStatusResult:
    return check_vllm_smoke_status(**kwargs)


def latest_vllm_run_summary(
    *,
    latest_record: Any = None,
    status_check: Any = None,
) -> VllmLatestRunSummary:
    latest = latest_record or latest_vllm_run_record()
    if not latest.ok or latest.record is None:
        return VllmLatestRunSummary(False, None, None, None, "UNKNOWN", latest.messages)

    record = latest.record
    check_status = status_check or check_vllm_run_status
    status = check_status(
        pid=record.pid,
        run_id=record.run_id,
        log_path=record.log_path,
        host=record.host,
        port=record.port,
    )
    if status.alive is False:
        state = "STOPPED"
    elif status.alive and status.port_listening:
        state = "READY"
    elif status.alive and status.port_listening is False:
        state = "STARTING"
    else:
        state = "UNKNOWN"

    model = _model_from_record(record)
    endpoint = f"http://{record.host}:{record.port}/v1"
    return VllmLatestRunSummary(True, record.preset_id, model, endpoint, state, status.messages)


@dataclass(frozen=True)
class VllmSmokeStatusPortCheck:
    ok: bool
    message: str


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


def read_vllm_run_log(log_path: str, *, last_lines: int = 80) -> VllmSmokeLogResult:
    return read_vllm_smoke_log(log_path, last_lines=last_lines)


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


def stop_vllm_run(**kwargs: Any) -> VllmSmokeStopResult:
    return stop_vllm_smoke(**kwargs)


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


def _coerce_port(value: int | str) -> int | None:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None
    return port if port > 0 else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _atomic_write_text(path: Path, payload: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(payload)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _model_from_command(command: list[str]) -> str | None:
    if "serve" not in command:
        return None
    index = command.index("serve")
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _model_from_record(record: VllmRunRecord) -> str | None:
    if isinstance(record.profile_snapshot, dict):
        model = str(record.profile_snapshot.get("model") or "").strip()
        if model:
            return model
    return _model_from_command(record.command)
