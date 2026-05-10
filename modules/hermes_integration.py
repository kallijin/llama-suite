from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.vllm_runner import VllmRunRecord, latest_vllm_run_record, latest_vllm_run_summary


HERMES_MIN_CONTEXT_LENGTH = 64000


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
            f"context_length: {HERMES_MIN_CONTEXT_LENGTH}",
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
        data["context_length"] = HERMES_MIN_CONTEXT_LENGTH
        custom_providers = data.setdefault("custom_providers", [])
        if not isinstance(custom_providers, list):
            custom_providers = []
            data["custom_providers"] = custom_providers
        _upsert_custom_provider_data(custom_providers, base_url=base_url, model_id=model_id)
        auxiliary = data.setdefault("auxiliary", {})
        if not isinstance(auxiliary, dict):
            auxiliary = {}
            data["auxiliary"] = auxiliary
        compression = auxiliary.setdefault("compression", {})
        if not isinstance(compression, dict):
            compression = {}
            auxiliary["compression"] = compression
        compression["context_length"] = HERMES_MIN_CONTEXT_LENGTH
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    lines = original.splitlines()
    updated_lines, seen_base, seen_model, seen_context = _update_yamlish_lines(lines, base_url=base_url, model_id=model_id)
    if not seen_base or not seen_model or not seen_context:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append("# llama-suite vLLM endpoint")
        if not seen_base:
            updated_lines.append(f"base_url: {base_url}")
        if not seen_model:
            updated_lines.append(f"model: {model_id}")
        if not seen_context:
            updated_lines.append(f"context_length: {HERMES_MIN_CONTEXT_LENGTH}")
    updated_lines = _ensure_yamlish_auxiliary_compression_context(updated_lines)
    updated_lines = _ensure_yamlish_custom_provider(updated_lines, base_url=base_url, model_id=model_id)
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


def _update_yamlish_lines(lines: list[str], *, base_url: str, model_id: str) -> tuple[list[str], bool, bool, bool]:
    updated: list[str] = []
    seen_base = False
    seen_model = False
    seen_context = False
    in_root_model_block = False
    root_model_indent = ""
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not indent and stripped == "model:":
            in_root_model_block = True
            root_model_indent = "  "
            updated.append(line)
        elif in_root_model_block and not indent and stripped:
            in_root_model_block = False
            if not seen_context:
                updated.append(f"{root_model_indent}context_length: {HERMES_MIN_CONTEXT_LENGTH}")
                seen_context = True
            updated.append(line)
        elif in_root_model_block and indent and stripped.startswith(("base_url:", "api_base:", "endpoint:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {base_url}")
            seen_base = True
        elif in_root_model_block and indent and stripped.startswith(("default:", "model:", "name:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {model_id}")
            seen_model = True
        elif in_root_model_block and indent and stripped.startswith("context_length:"):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {HERMES_MIN_CONTEXT_LENGTH}")
            seen_context = True
        elif not in_root_model_block and not indent and stripped.startswith(("base_url:", "api_base:", "endpoint:")):
            key = stripped.split(":", 1)[0]
            updated.append(f"{indent}{key}: {base_url}")
            seen_base = True
        elif not in_root_model_block and not indent and stripped.startswith("model:") and stripped.split(":", 1)[1].strip():
            updated.append(f"{indent}model: {model_id}")
            seen_model = True
        elif not in_root_model_block and not indent and stripped.startswith("context_length:"):
            updated.append(f"context_length: {HERMES_MIN_CONTEXT_LENGTH}")
            seen_context = True
        else:
            updated.append(line)
    if in_root_model_block and not seen_context:
        updated.append(f"{root_model_indent}context_length: {HERMES_MIN_CONTEXT_LENGTH}")
        seen_context = True
    return updated, seen_base, seen_model, seen_context


def _ensure_yamlish_auxiliary_compression_context(lines: list[str]) -> list[str]:
    updated: list[str] = []
    in_auxiliary = False
    in_compression = False
    saw_auxiliary = False
    saw_compression = False
    saw_context = False

    def close_compression_block() -> None:
        nonlocal in_compression, saw_context
        if in_compression and not saw_context:
            updated.append(f"    context_length: {HERMES_MIN_CONTEXT_LENGTH}")
            saw_context = True
        in_compression = False

    def close_auxiliary_block() -> None:
        nonlocal in_auxiliary, saw_compression, saw_context
        close_compression_block()
        if in_auxiliary and not saw_compression:
            updated.append("  compression:")
            updated.append(f"    context_length: {HERMES_MIN_CONTEXT_LENGTH}")
            saw_compression = True
            saw_context = True
        in_auxiliary = False

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not indent and stripped:
            if in_auxiliary and stripped != "auxiliary:":
                close_auxiliary_block()
            if stripped == "auxiliary:":
                in_auxiliary = True
                saw_auxiliary = True
                saw_compression = False
                saw_context = False
                updated.append(line)
                continue
        if in_auxiliary and indent == "  " and stripped:
            if in_compression and stripped != "compression:":
                close_compression_block()
            if stripped == "compression:":
                in_compression = True
                saw_compression = True
                saw_context = False
                updated.append(line)
                continue
        if in_compression and indent == "    " and stripped.startswith("context_length:"):
            updated.append(f"{indent}context_length: {HERMES_MIN_CONTEXT_LENGTH}")
            saw_context = True
            continue
        updated.append(line)

    if in_auxiliary:
        close_auxiliary_block()
    if not saw_auxiliary:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append("auxiliary:")
        updated.append("  compression:")
        updated.append(f"    context_length: {HERMES_MIN_CONTEXT_LENGTH}")
    return updated


def _upsert_custom_provider_data(custom_providers: list[Any], *, base_url: str, model_id: str) -> None:
    provider = None
    for candidate in custom_providers:
        if isinstance(candidate, dict) and candidate.get("name") == "llama-suite vLLM":
            provider = candidate
            break
    if provider is None:
        provider = {"name": "llama-suite vLLM"}
        custom_providers.append(provider)
    provider["base_url"] = base_url
    provider["api_key"] = "local"
    provider["model"] = model_id
    models = provider.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        provider["models"] = models
    entry = models.setdefault(model_id, {})
    if not isinstance(entry, dict):
        entry = {}
        models[model_id] = entry
    entry["context_length"] = HERMES_MIN_CONTEXT_LENGTH


def _ensure_yamlish_custom_provider(lines: list[str], *, base_url: str, model_id: str) -> list[str]:
    custom_index = _top_level_key_index(lines, "custom_providers:")
    if custom_index is None:
        provider_block = _custom_provider_block(base_url=base_url, model_id=model_id, list_indent="  ")
        updated = list(lines)
        if updated and updated[-1].strip():
            updated.append("")
        updated.append("custom_providers:")
        updated.extend(provider_block)
        return updated

    block_end = _top_level_block_end(lines, custom_index)
    list_indent = _custom_provider_list_indent(lines, custom_index + 1, block_end)
    provider_block = _custom_provider_block(base_url=base_url, model_id=model_id, list_indent=list_indent)
    provider_start = _llama_suite_vllm_provider_start(lines, custom_index + 1, block_end)
    if provider_start is None:
        updated = list(lines)
        updated[block_end:block_end] = provider_block
        return updated

    provider_end = _custom_provider_item_end(lines, provider_start, block_end)
    return list(lines[:provider_start]) + provider_block + list(lines[provider_end:])


def _custom_provider_block(*, base_url: str, model_id: str, list_indent: str) -> list[str]:
    field_indent = f"{list_indent}  "
    nested_indent = f"{list_indent}    "
    value_indent = f"{list_indent}      "
    return [
        f"{list_indent}- name: llama-suite vLLM",
        f"{field_indent}base_url: {base_url}",
        f"{field_indent}api_key: local",
        f"{field_indent}model: {model_id}",
        f"{field_indent}models:",
        f"{nested_indent}{model_id}:",
        f"{value_indent}context_length: {HERMES_MIN_CONTEXT_LENGTH}",
    ]


def _top_level_key_index(lines: list[str], key: str) -> int | None:
    for index, line in enumerate(lines):
        if line == key:
            return index
    return None


def _top_level_block_end(lines: list[str], start: int) -> int:
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not lines[index].startswith((" ", "\t", "- ")):
            return index
    return len(lines)


def _custom_provider_list_indent(lines: list[str], start: int, end: int) -> str:
    for index in range(start, end):
        stripped = lines[index].lstrip()
        if stripped.startswith("- "):
            return lines[index][: len(lines[index]) - len(stripped)]
    return "  "


def _llama_suite_vllm_provider_start(lines: list[str], start: int, end: int) -> int | None:
    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()
        if stripped == "- name: llama-suite vLLM":
            return index
        if stripped == "name: llama-suite vLLM":
            previous = index - 1
            if previous >= start and lines[previous].lstrip().startswith("-"):
                return previous
    return None


def _custom_provider_item_end(lines: list[str], start: int, block_end: int) -> int:
    for index in range(start + 1, block_end):
        if lines[index].lstrip().startswith("- "):
            return index
    return block_end


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
