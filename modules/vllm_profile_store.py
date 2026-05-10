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


def default_vllm_profile_path(
    profile_id: str = "custom-draft",
    *,
    store_root: str | Path | None = None,
) -> str:
    root = Path(store_root or DEFAULT_VLLM_PROFILE_STORE_ROOT).expanduser()
    return str(root / f"{_safe_name(profile_id)}.json")


def save_vllm_profile_draft(
    profile: VllmProfile,
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
) -> VllmProfileStoreResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    validation_messages = validate_vllm_profile(profile)
    payload = {
        "schema": VLLM_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "profile": profile.to_dict(),
        "validation_messages": validation_messages,
    }
    try:
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        return VllmProfileStoreResult(False, profile, profile_path, [f"vLLM profile draft save failed: {exc}"])

    messages = [f"vLLM profile draft saved: {profile_path}"]
    if validation_messages:
        messages.append("saved draft has validation messages: " + "; ".join(validation_messages))
    return VllmProfileStoreResult(True, profile, profile_path, messages)


def load_vllm_profile_draft(
    *,
    profile_id: str = "custom-draft",
    store_root: str | Path | None = None,
) -> VllmProfileStoreResult:
    profile_path = default_vllm_profile_path(profile_id, store_root=store_root)
    try:
        payload = json.loads(Path(profile_path).read_text())
    except Exception as exc:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft load failed: {exc}"])

    if payload.get("schema") != VLLM_PROFILE_SCHEMA:
        return VllmProfileStoreResult(False, None, profile_path, ["invalid vLLM profile draft schema"])

    profile_data = payload.get("profile")
    if not isinstance(profile_data, dict):
        return VllmProfileStoreResult(False, None, profile_path, ["vLLM profile draft payload is missing profile data"])

    try:
        profile = _profile_from_raw_dict(profile_data)
    except Exception as exc:
        return VllmProfileStoreResult(False, None, profile_path, [f"vLLM profile draft parse failed: {exc}"])

    validation_messages = validate_vllm_profile(profile)
    messages = [f"vLLM profile draft loaded: {profile_path}"]
    if validation_messages:
        messages.append("loaded draft has validation messages: " + "; ".join(validation_messages))
    return VllmProfileStoreResult(True, profile, profile_path, messages)


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
