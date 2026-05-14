from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILES_PATH = Path(os.path.expanduser("~/.hermes/llama-profiles.json"))
VALID_STATUSES = {"unknown", "pass", "fail", "partial"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _profile_key(model_name: str, _model_path: str) -> str:
    return model_name


def _status(value: str) -> str:
    if value not in VALID_STATUSES:
        return "unknown"
    return value


def default_profiles() -> dict[str, Any]:
    return {
        "version": 1,
        "models": {},
    }


def load_profiles() -> dict[str, Any]:
    data = default_profiles()
    if PROFILES_PATH.exists():
        try:
            with PROFILES_PATH.open() as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                data.update(saved)
        except Exception as e:
            print(f"  ⚠️  프로파일 파일 읽기 실패: {e}")

    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    return data


def save_profiles(data: dict[str, Any]) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILES_PATH.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def default_model_profile(model_name: str, model_path: str) -> dict[str, Any]:
    now = _now_iso()
    profile_key = _profile_key(model_name, model_path)
    return {
        "profile_key": profile_key,
        "model_name": model_name,
        "model_path": model_path,
        "model_file": Path(model_path).name,
        "recommended_backend": None,
        "stable_ctx_size": None,
        "sampler": {
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "repeat_penalty": None,
        },
        "reasoning": {
            "mode": None,
            "budget": None,
            "enable_thinking": None,
        },
        "chat_template": {
            "status": _status("unknown"),
            "template": None,
            "notes": [],
        },
        "direct_tool_call": {
            "status": _status("unknown"),
            "notes": [],
        },
        "hermes_tool_call": {
            "status": _status("unknown"),
            "notes": [],
        },
        "known_issues": [],
        "known_fixes": [],
        "notes": [],
        "updated_at": now,
    }


def get_model_profile(data: dict[str, Any], model_name: str, model_path: str) -> dict[str, Any]:
    if not isinstance(data.get("models"), dict):
        data["models"] = {}

    profile_key = _profile_key(model_name, model_path)
    profile = data["models"].get(profile_key)
    if not isinstance(profile, dict):
        profile = default_model_profile(model_name, model_path)
        data["models"][profile_key] = profile
    return profile


def upsert_model_profile(data: dict[str, Any], profile: dict[str, Any]) -> None:
    if not isinstance(data.get("models"), dict):
        data["models"] = {}

    profile_key = str(profile.get("profile_key") or profile.get("model_name") or "")
    if not profile_key:
        raise ValueError("profile must include profile_key or model_name")

    profile["profile_key"] = profile_key
    profile["updated_at"] = _now_iso()
    data["models"][profile_key] = profile
