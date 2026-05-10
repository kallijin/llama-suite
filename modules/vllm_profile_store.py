from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.vllm_profiles import VllmProfile, validate_vllm_profile


DEFAULT_VLLM_PROFILE_STORE_ROOT = "~/.local/state/llama-suite/profiles/vllm"
VLLM_PROFILE_SCHEMA = "llama-suite.vllm-profile.v1"


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


def default_vllm_profile_path(
    profile_id: str = "custom-draft",
    *,
    store_root: str | Path | None = None,
) -> str:
    root = Path(store_root or DEFAULT_VLLM_PROFILE_STORE_ROOT).expanduser()
    return str(root / f"{_safe_name(profile_id)}.json")


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


def load_vllm_profile_json_file(profile_path: str | Path) -> VllmProfileStoreResult:
    return _read_vllm_profile_payload(Path(profile_path))


def validate_vllm_profile_json_file(profile_path: str | Path) -> VllmProfileStoreResult:
    result = _read_vllm_profile_payload(Path(profile_path))
    if not result.ok:
        return result
    messages = [f"vLLM profile JSON validated: {result.profile_path or profile_path}"]
    messages.extend(message for message in result.messages if "validation messages" in message)
    return VllmProfileStoreResult(True, result.profile, result.profile_path, messages, result.profile_id)


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
