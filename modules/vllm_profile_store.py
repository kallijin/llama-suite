from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.vllm_profiles import (
    VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL,
    VllmProfile,
    build_vllm_command,
    validate_vllm_profile,
    verified_gemma4_26b_awq_vllm_profile,
)


DEFAULT_VLLM_PROFILE_STORE_ROOT = "~/.local/state/llama-suite/profiles/vllm"
VLLM_PROFILE_SCHEMA = "llama-suite.vllm-profile.v1"
VLLM_SELECTED_PROFILE_SCHEMA = "llama-suite.vllm-selected-profile.v1"
VLLM_MODEL_PROFILE_HINT_FILENAME = "llama-suite-vllm-profile.json"


@dataclass(frozen=True)
class VllmProfileStoreResult:
    ok: bool
    profile: VllmProfile | None
    profile_path: str | None
    messages: list[str]
    profile_id: str | None = None


@dataclass(frozen=True)
class VllmStoredProfileInfo:
    profile_id: str
    profile_path: str
    model: str
    validation_messages: list[str]


@dataclass(frozen=True)
class VllmProfileListResult:
    ok: bool
    profiles: list[VllmStoredProfileInfo]
    store_root: str
    messages: list[str]


@dataclass(frozen=True)
class VllmSelectedProfileResult:
    ok: bool
    profile_id: str | None
    state_path: str
    messages: list[str]


@dataclass(frozen=True)
class VllmProfileSelectionResult:
    ok: bool
    profile: VllmProfile | None
    profile_id: str
    profile_path: str | None
    selected_state_path: str | None
    messages: list[str]


@dataclass(frozen=True)
class VllmProfileBackupResult:
    ok: bool
    profile_path: str
    backup_path: str | None
    messages: list[str]


@dataclass(frozen=True)
class VllmModelProfileHintSaveResult:
    ok: bool
    profile: VllmProfile | None
    profile_id: str | None
    profile_path: str | None
    backup_path: str | None
    messages: list[str]


def default_vllm_profile_path(
    profile_id: str = "custom-draft",
    *,
    store_root: str | Path | None = None,
) -> str:
    root = Path(store_root or DEFAULT_VLLM_PROFILE_STORE_ROOT).expanduser()
    return str(root / f"{_safe_name(profile_id)}.json")


def default_vllm_selected_profile_path(*, store_root: str | Path | None = None) -> str:
    root = Path(store_root or DEFAULT_VLLM_PROFILE_STORE_ROOT).expanduser()
    return str(root / "selected" / "latest.json")


def list_vllm_profile_drafts(*, store_root: str | Path | None = None) -> VllmProfileListResult:
    root = Path(store_root or DEFAULT_VLLM_PROFILE_STORE_ROOT).expanduser()
    if not root.is_dir():
        return VllmProfileListResult(False, [], str(root), [f"vLLM profile store does not exist: {root}"])

    profiles: list[VllmStoredProfileInfo] = []
    messages: list[str] = []
    for path in sorted(root.glob("*.json")):
        result = _read_vllm_profile_payload(path)
        if not result.ok or result.profile is None:
            messages.extend(f"{path.name}: {message}" for message in result.messages)
            continue
        profile_id = _profile_id_from_path(path)
        try:
            payload = json.loads(path.read_text())
            profile_id = str(payload.get("profile_id") or profile_id)
        except Exception:
            pass
        validation_messages = validate_vllm_profile(result.profile)
        profiles.append(
            VllmStoredProfileInfo(
                profile_id=profile_id,
                profile_path=str(path),
                model=str(result.profile.model or ""),
                validation_messages=validation_messages,
            )
        )

    ok = bool(profiles)
    if not profiles and not messages:
        messages.append(f"no vLLM profile drafts found under {root}")
    return VllmProfileListResult(ok, profiles, str(root), messages)


def save_vllm_profile_draft(
    profile: VllmProfile,
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
) -> VllmProfileStoreResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    validation_messages = validate_vllm_profile(profile)
    payload = vllm_profile_draft_payload(profile, profile_id=profile_id)
    try:
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        return VllmProfileStoreResult(False, profile, profile_path, [f"vLLM profile draft save failed: {exc}"], profile_id)

    messages = [f"vLLM profile draft saved: {profile_path}"]
    if validation_messages:
        messages.append("saved draft has validation messages: " + "; ".join(validation_messages))
    return VllmProfileStoreResult(True, profile, profile_path, messages, profile_id)


def backup_vllm_profile_draft(
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
    timestamp: str | None = None,
) -> VllmProfileBackupResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    path = Path(profile_path)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = str(path.with_name(path.name + f".{stamp}.bak"))
    if not path.exists():
        return VllmProfileBackupResult(True, profile_path, None, [f"no existing vLLM profile draft to back up: {profile_path}"])
    try:
        Path(backup_path).write_text(path.read_text())
    except Exception as exc:
        return VllmProfileBackupResult(False, profile_path, backup_path, [f"vLLM profile draft backup failed: {exc}"])
    return VllmProfileBackupResult(True, profile_path, backup_path, [f"vLLM profile draft backup created: {backup_path}"])


def vllm_profile_draft_payload(profile: VllmProfile, *, profile_id: str = "custom-draft") -> dict[str, Any]:
    return {
        "schema": VLLM_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "profile": profile.to_dict(),
        "validation_messages": validate_vllm_profile(profile),
    }


def format_vllm_profile_draft_json(profile: VllmProfile, *, profile_id: str = "custom-draft") -> str:
    return json.dumps(vllm_profile_draft_payload(profile, profile_id=profile_id), indent=2, sort_keys=True) + "\n"


def load_vllm_profile_draft(
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
) -> VllmProfileStoreResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    return _read_vllm_profile_payload(Path(profile_path))


def save_selected_vllm_profile_id(
    profile_id: str,
    *,
    store_root: str | Path | None = None,
) -> VllmSelectedProfileResult:
    state_path = default_vllm_selected_profile_path(store_root=store_root)
    payload = {
        "schema": VLLM_SELECTED_PROFILE_SCHEMA,
        "profile_id": str(profile_id or "custom-draft"),
    }
    try:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        return VllmSelectedProfileResult(False, profile_id, state_path, [f"selected vLLM profile save failed: {exc}"])
    return VllmSelectedProfileResult(True, profile_id, state_path, [f"selected vLLM profile saved: {profile_id}"])


def save_verified_gemma4_26b_awq_beta_profile(
    *,
    store_root: str | Path | None = None,
) -> VllmProfileSelectionResult:
    profile_id = VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL
    profile = verified_gemma4_26b_awq_vllm_profile()
    saved = save_vllm_profile_draft(profile, profile_id=profile_id, store_root=store_root)
    messages = list(saved.messages)
    if not saved.ok:
        return VllmProfileSelectionResult(False, profile, profile_id, saved.profile_path, None, messages)

    selected = save_selected_vllm_profile_id(profile_id, store_root=store_root)
    messages.extend(selected.messages)
    ok = bool(saved.ok and selected.ok)
    return VllmProfileSelectionResult(ok, profile, profile_id, saved.profile_path, selected.state_path, messages)


def load_selected_vllm_profile_id(*, store_root: str | Path | None = None) -> VllmSelectedProfileResult:
    state_path = default_vllm_selected_profile_path(store_root=store_root)
    try:
        payload = json.loads(Path(state_path).read_text())
    except FileNotFoundError:
        return VllmSelectedProfileResult(False, None, state_path, ["no selected vLLM profile saved yet"])
    except Exception as exc:
        return VllmSelectedProfileResult(False, None, state_path, [f"selected vLLM profile load failed: {exc}"])

    if payload.get("schema") != VLLM_SELECTED_PROFILE_SCHEMA:
        return VllmSelectedProfileResult(False, None, state_path, ["invalid selected vLLM profile schema"])

    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        return VllmSelectedProfileResult(False, None, state_path, ["selected vLLM profile id is empty"])
    return VllmSelectedProfileResult(True, profile_id, state_path, [f"selected vLLM profile loaded: {profile_id}"])


def load_selected_vllm_profile_draft(*, store_root: str | Path | None = None) -> VllmProfileStoreResult:
    selected = load_selected_vllm_profile_id(store_root=store_root)
    if not selected.ok or not selected.profile_id:
        return VllmProfileStoreResult(False, None, selected.state_path, selected.messages)
    loaded = load_vllm_profile_draft(profile_id=selected.profile_id, store_root=store_root)
    messages = selected.messages + loaded.messages
    return VllmProfileStoreResult(loaded.ok, loaded.profile, loaded.profile_path, messages, selected.profile_id)


def load_vllm_profile_json_file(profile_path: str | Path) -> VllmProfileStoreResult:
    return _read_vllm_profile_payload(Path(profile_path))


def validate_vllm_profile_json_file(profile_path: str | Path) -> VllmProfileStoreResult:
    result = _read_vllm_profile_payload(Path(profile_path))
    if not result.ok:
        return result
    messages = [f"vLLM profile JSON validated: {result.profile_path or profile_path}"]
    messages.extend(message for message in result.messages if "validation messages" in message)
    return VllmProfileStoreResult(True, result.profile, result.profile_path, messages, result.profile_id)


def vllm_model_profile_hint_path(model_path: str | Path) -> str:
    return str(Path(model_path).expanduser() / VLLM_MODEL_PROFILE_HINT_FILENAME)


def load_vllm_model_profile_hint(model_path: str | Path) -> VllmProfileStoreResult:
    model_dir = Path(model_path).expanduser()
    hint_path = model_dir / VLLM_MODEL_PROFILE_HINT_FILENAME
    if not model_dir.is_dir():
        return VllmProfileStoreResult(False, None, str(hint_path), [f"vLLM model path is not a directory: {model_dir}"])
    if not hint_path.is_file():
        return VllmProfileStoreResult(False, None, str(hint_path), [f"{VLLM_MODEL_PROFILE_HINT_FILENAME} not found in model directory"])
    return _read_vllm_profile_payload(hint_path)


def import_vllm_model_profile_hint(
    model_path: str | Path,
    *,
    selected_profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
    confirmed: bool = False,
) -> VllmProfileSelectionResult:
    loaded = load_vllm_model_profile_hint(model_path)
    messages = list(loaded.messages)
    if not loaded.ok or loaded.profile is None:
        return VllmProfileSelectionResult(False, loaded.profile, selected_profile_id, loaded.profile_path, None, messages)

    validation_messages = validate_vllm_profile(loaded.profile)
    if validation_messages:
        messages.extend(validation_messages)
        return VllmProfileSelectionResult(False, loaded.profile, loaded.profile_id or selected_profile_id, loaded.profile_path, None, messages)

    command, command_messages = build_vllm_command(loaded.profile)
    if command is None:
        messages.extend(command_messages)
        return VllmProfileSelectionResult(False, loaded.profile, loaded.profile_id or selected_profile_id, loaded.profile_path, None, messages)

    if not confirmed:
        messages.append("vLLM model profile hint import cancelled: explicit confirmation is required")
        return VllmProfileSelectionResult(False, loaded.profile, loaded.profile_id or selected_profile_id, loaded.profile_path, None, messages)

    imported_profile_id = _profile_id_for_model_hint(model_path, loaded.profile_id, loaded.profile_path)
    backup = backup_vllm_profile_draft(profile_id=selected_profile_id, store_root=store_root)
    messages.extend(backup.messages)
    if not backup.ok:
        return VllmProfileSelectionResult(False, loaded.profile, imported_profile_id, loaded.profile_path, None, messages)

    saved = save_vllm_profile_draft(loaded.profile, profile_id=imported_profile_id, store_root=store_root)
    messages.extend(saved.messages)
    if not saved.ok:
        return VllmProfileSelectionResult(False, loaded.profile, imported_profile_id, saved.profile_path, None, messages)

    selected = save_selected_vllm_profile_id(imported_profile_id, store_root=store_root)
    messages.extend(selected.messages)
    ok = bool(saved.ok and selected.ok)
    return VllmProfileSelectionResult(ok, loaded.profile, imported_profile_id, saved.profile_path, selected.state_path, messages)


def save_vllm_model_profile_hint(
    profile: VllmProfile,
    *,
    profile_id: str = "custom-draft",
    confirmed: bool = False,
    timestamp: str | None = None,
) -> VllmModelProfileHintSaveResult:
    model_text = str(getattr(profile, "model", "") or "").strip()
    messages: list[str] = []
    if not model_text:
        return VllmModelProfileHintSaveResult(False, profile, profile_id, None, None, ["model should not be empty"])
    if _looks_like_hf_model_id(model_text):
        return VllmModelProfileHintSaveResult(
            False,
            profile,
            profile_id,
            None,
            None,
            ["cannot save local profile hint for a Hugging Face model ID; choose a local model directory first"],
        )

    model_path = Path(model_text).expanduser()
    hint_path = model_path / VLLM_MODEL_PROFILE_HINT_FILENAME
    if model_path.suffix.lower() == ".gguf":
        return VllmModelProfileHintSaveResult(
            False,
            profile,
            profile_id,
            str(hint_path),
            None,
            ["cannot save vLLM profile hint into a GGUF file path; GGUF belongs to the llama.cpp workflow by default"],
        )
    if not model_path.is_dir():
        return VllmModelProfileHintSaveResult(False, profile, profile_id, str(hint_path), None, [f"model path is not a local directory: {model_path}"])

    validation_messages = validate_vllm_profile(profile)
    if validation_messages:
        messages.extend(validation_messages)
        return VllmModelProfileHintSaveResult(False, profile, profile_id, str(hint_path), None, messages)

    command, command_messages = build_vllm_command(profile)
    if command is None:
        messages.extend(command_messages)
        return VllmModelProfileHintSaveResult(False, profile, profile_id, str(hint_path), None, messages)

    if not confirmed:
        return VllmModelProfileHintSaveResult(
            False,
            profile,
            profile_id,
            str(hint_path),
            None,
            ["vLLM model profile hint save cancelled: exact confirmation is required"],
        )

    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path: Path | None = None
    try:
        if hint_path.exists():
            backup_path = hint_path.with_name(hint_path.name + f".{stamp}.bak")
            shutil.copy2(hint_path, backup_path)
            messages.append(f"existing vLLM model profile hint backup created: {backup_path}")
        _atomic_write_text(hint_path, format_vllm_profile_draft_json(profile, profile_id=profile_id))
    except Exception as exc:
        return VllmModelProfileHintSaveResult(False, profile, profile_id, str(hint_path), str(backup_path) if backup_path else None, [f"vLLM model profile hint save failed: {exc}"])

    messages.append(f"vLLM model profile hint saved: {hint_path}")
    messages.append("launch was not started")
    return VllmModelProfileHintSaveResult(True, profile, profile_id, str(hint_path), str(backup_path) if backup_path else None, messages)


def delete_vllm_profile_draft(
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
    confirmed: bool = False,
) -> VllmProfileStoreResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    if not confirmed:
        return VllmProfileStoreResult(False, None, profile_path, ["vLLM profile draft delete cancelled: explicit confirmation is required"], profile_id)

    path = Path(profile_path)
    try:
        path.unlink()
    except FileNotFoundError:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft delete failed: file not found: {profile_path}"], profile_id)
    except Exception as exc:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft delete failed: {exc}"], profile_id)
    return VllmProfileStoreResult(True, None, profile_path, [f"vLLM profile draft deleted: {profile_path}"], profile_id)


def _read_vllm_profile_payload(path: Path) -> VllmProfileStoreResult:
    profile_path = str(path.expanduser())
    try:
        payload = json.loads(Path(profile_path).read_text())
    except Exception as exc:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft load failed: {exc}"])

    if payload.get("schema") != VLLM_PROFILE_SCHEMA:
        return VllmProfileStoreResult(False, None, profile_path, ["invalid vLLM profile draft schema"])

    profile_data = payload.get("profile")
    if not isinstance(profile_data, dict):
        return VllmProfileStoreResult(False, None, profile_path, ["vLLM profile draft payload is missing profile data"])

    profile_id = str(payload.get("profile_id") or _profile_id_from_path(Path(profile_path)))

    try:
        profile = _profile_from_raw_dict(profile_data)
    except Exception as exc:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft parse failed: {exc}"])

    validation_messages = validate_vllm_profile(profile)
    messages = [f"vLLM profile draft loaded: {profile_path}"]
    if validation_messages:
        messages.append("loaded draft has validation messages: " + "; ".join(validation_messages))
    return VllmProfileStoreResult(True, profile, profile_path, messages, profile_id)


def _profile_from_raw_dict(data: dict[str, Any]) -> VllmProfile:
    defaults = VllmProfile().to_dict()
    raw = {key: data.get(key, value) for key, value in defaults.items()}
    return VllmProfile(**raw)


def _atomic_write_text(path: Path, payload: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(path)


def _safe_name(value: str) -> str:
    text = str(value or "custom-draft").strip()
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in text) or "custom-draft"


def _profile_id_from_path(path: Path) -> str:
    return path.stem or "custom-draft"


def _profile_id_for_model_hint(model_path: str | Path, loaded_profile_id: str | None, loaded_profile_path: str | None) -> str:
    hint_default_id = _profile_id_from_path(Path(loaded_profile_path or VLLM_MODEL_PROFILE_HINT_FILENAME))
    loaded_id = str(loaded_profile_id or "").strip()
    if loaded_id and loaded_id != hint_default_id:
        return _safe_name(loaded_id)
    return _safe_name(Path(model_path).expanduser().name)


def _looks_like_hf_model_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text.startswith(("/", "~", ".")):
        return False
    if "\\" in text:
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(parts) and not any(part in (".", "..") for part in parts)
