from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from modules.vllm_model_scan import (
    ModelReadiness,
    VllmModelCandidate,
    render_readiness_text,
)


MODEL_REGISTRY_SCHEMA = "llama-suite.model-registry.v1"
DEFAULT_MODEL_REGISTRY_PATH = "~/.local/state/llama-suite/model-registry.json"


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    backend: str
    source: dict[str, Any]
    classification: dict[str, Any]
    readiness: dict[str, Any]
    runtime: dict[str, Any]
    human: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRegistry:
    models: list[RegisteredModel]
    schema: str = MODEL_REGISTRY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True)
class ModelRegistryResult:
    ok: bool
    registry: ModelRegistry | None
    registry_path: str | None
    messages: list[str]


def registered_model_from_candidate(
    candidate: VllmModelCandidate,
    *,
    alias: str | None = None,
) -> RegisteredModel:
    model_alias = _safe_alias(alias or candidate.source.original_name)
    readiness = candidate.readiness.to_dict()
    if readiness["state"] == "needs_registration" and not readiness["missing"]:
        readiness = {"state": "ready", "missing": [], "blocking": False}

    return RegisteredModel(
        id=model_id_from_source(candidate.source.path, alias=model_alias),
        backend=candidate.candidate_backend,
        source=candidate.source.to_dict(),
        classification={
            "family": candidate.classification_guess.family,
            "format": candidate.classification_guess.format,
            "quant": candidate.classification_guess.quant,
            "size_b": candidate.classification_guess.size_b,
            "confidence": candidate.classification_guess.confidence,
            "evidence": list(candidate.classification_guess.evidence),
        },
        readiness=readiness,
        runtime={
            "served_model_name": model_alias,
            "preferred_profile_id": model_alias,
        },
        human={
            "alias": model_alias,
            "notes": "",
        },
        events=[],
    )


def model_id_from_source(path: str, *, alias: str | None = None) -> str:
    name = _safe_alias(alias or Path(path).name or "model")
    digest = hashlib.sha1(str(Path(path).expanduser()).encode("utf-8")).hexdigest()[:8]
    return f"{name}__{digest}"


def upsert_registered_model(registry: ModelRegistry, model: RegisteredModel) -> ModelRegistry:
    models: list[RegisteredModel] = []
    replaced = False
    new_path = str(model.source.get("path") or "")
    for existing in registry.models:
        existing_path = str(existing.source.get("path") or "")
        if existing.id == model.id or (new_path and existing_path == new_path):
            models.append(model)
            replaced = True
        else:
            models.append(existing)
    if not replaced:
        models.append(model)
    return ModelRegistry(models=models)


def save_model_registry(
    registry: ModelRegistry,
    *,
    registry_path: str | Path | None = None,
) -> ModelRegistryResult:
    path = Path(registry_path or DEFAULT_MODEL_REGISTRY_PATH).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        return ModelRegistryResult(False, registry, str(path), [f"model registry save failed: {exc}"])
    return ModelRegistryResult(True, registry, str(path), [f"model registry saved: {path}"])


def load_model_registry(*, registry_path: str | Path | None = None) -> ModelRegistryResult:
    path = Path(registry_path or DEFAULT_MODEL_REGISTRY_PATH).expanduser()
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return ModelRegistryResult(False, None, str(path), [f"model registry not found: {path}"])
    except Exception as exc:
        return ModelRegistryResult(False, None, str(path), [f"model registry load failed: {exc}"])

    if payload.get("schema") != MODEL_REGISTRY_SCHEMA:
        return ModelRegistryResult(False, None, str(path), ["invalid model registry schema"])

    models: list[RegisteredModel] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        models.append(
            RegisteredModel(
                id=str(item.get("id") or ""),
                backend=str(item.get("backend") or ""),
                source=dict(item.get("source") or {}),
                classification=dict(item.get("classification") or {}),
                readiness=dict(item.get("readiness") or {}),
                runtime=dict(item.get("runtime") or {}),
                human=dict(item.get("human") or {}),
                events=list(item.get("events") or []),
            )
        )
    return ModelRegistryResult(True, ModelRegistry(models=models), str(path), [f"model registry loaded: {path}"])


def registered_model_summary_line(model: RegisteredModel) -> str:
    alias = str(model.human.get("alias") or model.id)
    quant = str(model.classification.get("quant") or "unknown").upper()
    backend = str(model.backend or "unknown")
    readiness = render_readiness_text(model.readiness)
    return f"{alias} ({backend}, {quant}, {readiness})"


def readiness_human_text(readiness: ModelReadiness | dict[str, Any]) -> str:
    return render_readiness_text(readiness)


def _safe_alias(value: str) -> str:
    text = str(value or "model").strip().lower()
    text = re.sub(r"[^a-z0-9._+-]+", "-", text)
    text = text.strip("-._")
    return text or "model"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)
