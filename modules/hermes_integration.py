from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.vllm_runner import VllmRunRecord, latest_vllm_run_record, latest_vllm_run_summary


@dataclass(frozen=True)
class HermesVllmSyncPlan:
    ok: bool
    config_path: str | None
    base_url: str | None
    model_id: str | None
    run_id: str | None
    original_text: str
    updated_text: str
    messages: list[str]


@dataclass(frozen=True)
class HermesVllmSyncResult:
    ok: bool
    config_path: str | None
    backup_path: str | None
    messages: list[str]


def build_hermes_vllm_sync_plan(
    config_path: Any,
    *,
    latest_record: Any = None,
    status_check: Any = None,
) -> HermesVllmSyncPlan:
    path = _expand_config_path(config_path)
    if path is None:
        return _empty_plan(["Hermes config path is not registered"])
    if not path.is_file():
        return _empty_plan([f"Hermes config file does not exist: {path}"], config_path=str(path))

    latest = latest_record or latest_vllm_run_record()
    if not getattr(latest, "ok", False) or latest.record is None:
        return _empty_plan(["latest vLLM run record is not available"] + list(getattr(latest, "messages", [])), config_path=str(path))

    summary = latest_vllm_run_summary(latest_record=latest, status_check=status_check)
    if summary.status != "READY":
        return _empty_plan([f"latest vLLM run is not READY: {summary.status}"] + summary.messages, config_path=str(path))

    try:
        original = path.read_text(errors="replace")
    except Exception as exc:
        return _empty_plan([f"Hermes config read failed: {exc}"], config_path=str(path))

    record = latest.record
    base_url = f"http://{record.host}:{record.port}/v1"
    model_id = served_model_id_from_vllm_record(record)
    updated = update_hermes_config_text(original, base_url=base_url, model_id=model_id, config_path=str(path))

    return HermesVllmSyncPlan(
        ok=True,
        config_path=str(path),
        base_url=base_url,
        model_id=model_id,
        run_id=record.run_id,
        original_text=original,
        updated_text=updated,
        messages=[
            "Hermes vLLM sync plan built",
            f"base_url: {base_url}",
            f"model: {model_id}",
            f"source run_id: {record.run_id}",
        ],
    )


def write_hermes_vllm_sync_plan(
    plan: HermesVllmSyncPlan,
    *,
    confirmed: bool,
    timestamp: str | None = None,
) -> HermesVllmSyncResult:
    if not confirmed:
        return HermesVllmSyncResult(False, plan.config_path, None, ["Hermes config write cancelled: explicit confirmation is required"])
    if not plan.ok or not plan.config_path:
        return HermesVllmSyncResult(False, plan.config_path, None, ["Hermes config write refused: sync plan is not ready"] + plan.messages)

    path = Path(plan.config_path).expanduser()
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{stamp}")
    try:
        backup_path.write_text(plan.original_text)
        _atomic_write_text(path, plan.updated_text)
    except Exception as exc:
        return HermesVllmSyncResult(False, str(path), str(backup_path), [f"Hermes config write failed: {exc}"])

    return HermesVllmSyncResult(True, str(path), str(backup_path), [f"Hermes config updated: {path}", f"backup saved: {backup_path}"])


def served_model_id_from_vllm_record(record: VllmRunRecord) -> str:
    command = list(record.command)
    for index, part in enumerate(command):
        if part == "--served-model-name" and index + 1 < len(command):
            return str(command[index + 1])
    snapshot = record.profile_snapshot if isinstance(record.profile_snapshot, dict) else {}
    if snapshot.get("model"):
        return str(snapshot["model"])
    if "serve" in command:
        serve_index = command.index("serve")
        if serve_index + 1 < len(command):
            return str(command[serve_index + 1])
    return str(record.preset_id)


def update_hermes_config_text(original: str, *, base_url: str, model_id: str, config_path: str = "") -> str:
    if Path(config_path).suffix.lower() == ".json":
        try:
            data = json.loads(original or "{}")
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["base_url"] = base_url
        data["model"] = model_id
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    lines = original.splitlines()
    updated_lines, seen_base, seen_model = _update_yamlish_lines(lines, base_url=base_url, model_id=model_id)
    if not seen_base or not seen_model:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append("# llama-suite vLLM endpoint")
        if not seen_base:
            updated_lines.append(f"base_url: {base_url}")
        if not seen_model:
            updated_lines.append(f"model: {model_id}")
    return "\n".join(updated_lines) + "\n"


def format_hermes_vllm_sync_plan(plan: HermesVllmSyncPlan) -> list[str]:
    lines = ["Hermes vLLM sync preview:"]
    for message in plan.messages:
        lines.append(f"- {message}")
    if plan.ok:
        lines.extend(
            [
                f"config_path: {plan.config_path}",
                f"base_url: {plan.base_url}",
                f"model: {plan.model_id}",
                f"run_id: {plan.run_id}",
                "planned config:",
            ]
        )
        lines.extend(plan.updated_text.splitlines() or [""])
    return lines


def _update_yamlish_lines(lines: list[str], *, base_url: str, model_id: str) -> tuple[list[str], bool, bool]:
    updated: list[str] = []
    seen_base = False
    seen_model = False
    in_root_model_block = False
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not indent and stripped == "model:":
            in_root_model_block = True
            updated.append(line)
        elif in_root_model_block and not indent and stripped:
            in_root_model_block = False
            updated.append(line)
        elif in_root_model_block and indent and stripped.startswith(("base_url:", "api_base:", "endpoint:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {base_url}")
            seen_base = True
        elif in_root_model_block and indent and stripped.startswith(("default:", "model:", "name:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {model_id}")
            seen_model = True
        elif not in_root_model_block and stripped.startswith(("base_url:", "api_base:", "endpoint:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {base_url}")
            seen_base = True
        elif not in_root_model_block and not indent and stripped.startswith("model:") and stripped.split(":", 1)[1].strip():
            updated.append(f"{indent}model: {model_id}")
            seen_model = True
        else:
            updated.append(line)
    return updated, seen_base, seen_model


def _empty_plan(messages: list[str], *, config_path: str | None = None) -> HermesVllmSyncPlan:
    return HermesVllmSyncPlan(False, config_path, None, None, None, "", "", messages)


def _expand_config_path(config_path: Any) -> Path | None:
    if not isinstance(config_path, str) or not config_path.strip():
        return None
    return Path(config_path).expanduser()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)
