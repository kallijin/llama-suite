from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISCOVERY_CACHE_SCHEMA = "llama-suite.discovery-cache.v1"

TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "spiece.model",
    "merges.txt",
}
WEIGHT_SUFFIXES = (".safetensors", ".safetensors.index.json", ".bin")
VLLM_MODEL_PROFILE_HINT_FILENAME = "llama-suite-vllm-profile.json"


@dataclass(frozen=True)
class ModelSource:
    kind: str
    path: str
    original_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationGuess:
    quant: str
    confidence: float
    evidence: list[str]
    family: str = "unknown"
    format: str = "unknown"
    size_b: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelReadiness:
    state: str
    missing: list[str]
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VllmModelCandidate:
    source: ModelSource
    candidate_backend: str
    classification_guess: ClassificationGuess
    readiness: ModelReadiness
    has_suite_profile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "candidate_backend": self.candidate_backend,
            "classification_guess": self.classification_guess.to_dict(),
            "readiness": self.readiness.to_dict(),
            "has_suite_profile": self.has_suite_profile,
        }


@dataclass(frozen=True)
class VllmDiscoveryCache:
    scanned_at: str
    roots: list[str]
    candidates: list[VllmModelCandidate]
    messages: list[str]
    schema: str = DISCOVERY_CACHE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scanned_at": self.scanned_at,
            "roots": list(self.roots),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "messages": list(self.messages),
        }


def scan_vllm_model_candidates(
    roots: list[str | Path],
    *,
    scanned_at: str | None = None,
) -> VllmDiscoveryCache:
    messages: list[str] = []
    candidates: list[VllmModelCandidate] = []
    seen_paths: set[str] = set()
    root_strings = [str(Path(root).expanduser()) for root in roots]

    for root_value in root_strings:
        root = Path(root_value).expanduser()
        if not root.exists():
            messages.append(f"model root does not exist: {root}")
            continue

        if root.is_dir():
            paths = [root] if _has_local_model_marker(root) else []
            paths.extend(path for path in sorted(root.iterdir()) if path.is_dir())
        else:
            paths = []

        for path in paths:
            candidate = inspect_vllm_model_directory(path)
            if candidate is None:
                continue
            if candidate.source.path in seen_paths:
                continue
            seen_paths.add(candidate.source.path)
            candidates.append(candidate)

        if root.is_file() and root.suffix.lower() == ".gguf":
            candidate = inspect_gguf_model_file(root)
            if candidate.source.path not in seen_paths:
                seen_paths.add(candidate.source.path)
                candidates.append(candidate)
        elif root.is_dir():
            for gguf in sorted(root.rglob("*.gguf")):
                candidate = inspect_gguf_model_file(gguf)
                if candidate.source.path in seen_paths:
                    continue
                seen_paths.add(candidate.source.path)
                candidates.append(candidate)

    return VllmDiscoveryCache(
        scanned_at=scanned_at or datetime.now(timezone.utc).isoformat(),
        roots=root_strings,
        candidates=candidates,
        messages=messages,
    )


def inspect_vllm_model_directory(path: str | Path) -> VllmModelCandidate | None:
    model_dir = Path(path).expanduser()
    if not model_dir.is_dir():
        return None

    evidence: list[str] = []
    missing: list[str] = []

    config = model_dir / "config.json"
    config_data = _read_config_json(config)
    if config.is_file():
        evidence.append("config_json_exists")
    else:
        missing.append("config")

    tokenizer_found = _has_tokenizer_file(model_dir)
    if tokenizer_found:
        evidence.append("tokenizer_file_found")
    else:
        missing.append("tokenizer")

    weights_found = _has_weight_file(model_dir)
    if weights_found:
        evidence.append("safetensors_weights_found" if _has_safetensors_file(model_dir) else "pytorch_weights_found")
    else:
        missing.append("weights")
    has_suite_profile = (model_dir / VLLM_MODEL_PROFILE_HINT_FILENAME).is_file()
    if has_suite_profile:
        evidence.append("llama_suite_vllm_profile_hint_found")

    name_text = model_dir.name.lower()
    config_text = json.dumps(config_data, sort_keys=True).lower() if config_data else ""
    quant, quant_evidence = _guess_quant(name_text, config_text)
    evidence.extend(quant_evidence)
    size_b, size_evidence = _guess_size_b(name_text)
    evidence.extend(size_evidence)
    family, family_evidence = _guess_family(name_text, config_data)
    evidence.extend(family_evidence)

    if not evidence and quant == "unknown":
        return None

    readiness = ModelReadiness(
        state="needs_files" if missing else "needs_registration",
        missing=missing,
        blocking=bool(missing),
    )
    classification = ClassificationGuess(
        quant=quant,
        confidence=_confidence_for(quant, evidence, missing),
        evidence=_dedupe(evidence),
        family=family,
        format="hf_safetensors" if weights_found else "local_hf_directory",
        size_b=size_b,
    )
    return VllmModelCandidate(
        source=ModelSource(
            kind="local_hf_directory",
            path=str(model_dir),
            original_name=model_dir.name,
        ),
        candidate_backend="vllm",
        classification_guess=classification,
        readiness=readiness,
        has_suite_profile=has_suite_profile,
    )


def inspect_gguf_model_file(path: str | Path) -> VllmModelCandidate:
    gguf = Path(path).expanduser()
    name_text = gguf.name.lower()
    size_b, size_evidence = _guess_size_b(name_text)
    evidence = ["gguf_file_found", *size_evidence]
    return VllmModelCandidate(
        source=ModelSource(
            kind="gguf_file",
            path=str(gguf),
            original_name=gguf.name,
        ),
        candidate_backend="llama.cpp",
        classification_guess=ClassificationGuess(
            quant="llama",
            confidence=0.95,
            evidence=_dedupe(evidence),
            family="llama.cpp",
            format="gguf",
            size_b=size_b,
        ),
        readiness=ModelReadiness(
            state="needs_registration",
            missing=[],
            blocking=False,
        ),
        has_suite_profile=False,
    )


def render_unregistered_candidate_summary_line(candidate: VllmModelCandidate) -> str:
    alias = candidate.source.original_name
    quant = candidate.classification_guess.quant.upper()
    readiness = render_readiness_text(candidate.readiness)
    return f"{alias} ({candidate.candidate_backend}, {quant}, {readiness})"


def render_readiness_text(readiness: ModelReadiness | dict[str, Any]) -> str:
    if isinstance(readiness, ModelReadiness):
        state = readiness.state
        missing = readiness.missing
    else:
        state = str(readiness.get("state") or "unknown")
        missing = list(readiness.get("missing") or [])

    if state == "needs_files" and missing:
        return "missing " + ", ".join(missing)
    if state == "needs_registration":
        return "needs registration"
    if state == "ready":
        return "READY"
    return state


def render_readiness_human_lines(readiness: ModelReadiness | dict[str, Any]) -> list[str]:
    if isinstance(readiness, ModelReadiness):
        state = readiness.state
        missing = list(readiness.missing)
        blocking = readiness.blocking
    else:
        state = str(readiness.get("state") or "unknown")
        missing = list(readiness.get("missing") or [])
        blocking = bool(readiness.get("blocking"))

    if state == "needs_files":
        lines = ["상태: 파일 보완 필요"]
        if missing:
            lines.append("다음 파일이 필요합니다: " + ", ".join(missing))
        lines.append("실행 가능 여부: 불가" if blocking else "실행 가능 여부: 가능")
        return lines
    if state == "needs_registration":
        return ["상태: 등록 필요", "실행 가능 여부: 가능"]
    if state == "ready":
        return ["상태: 실행 준비 완료", "실행 가능 여부: 가능"]
    return [f"상태: {state}", "실행 가능 여부: 불가" if blocking else "실행 가능 여부: 가능"]


def render_vllm_model_candidate_lines(cache: VllmDiscoveryCache, *, limit: int = 20) -> list[str]:
    lines = [f"vLLM model folders: {len(cache.candidates)} found"]
    if cache.messages:
        lines.append("scan messages:")
        lines.extend(f"- {message}" for message in cache.messages)
    if not cache.candidates:
        lines.append("상태: 후보 없음")
        return lines

    for index, candidate in enumerate(cache.candidates[:limit], 1):
        classification = candidate.classification_guess
        family = classification.family if classification.family != "unknown" else "unknown family"
        quant = classification.quant.upper()
        size = f"{classification.size_b}B" if classification.size_b is not None else "unknown size"
        lines.append("")
        lines.append(f"[{index}] {candidate.source.original_name}")
        lines.append(f"    추정: {family} / {size} / {quant}")
        lines.append(f"    backend: {candidate.candidate_backend}")
        lines.append(f"    files: {_file_status_text(candidate)}")
        for readiness_line in render_readiness_human_lines(candidate.readiness):
            lines.append(f"    {readiness_line}")
        evidence = ", ".join(candidate.classification_guess.evidence[:5]) or "-"
        lines.append(f"    근거: {evidence}")
    if len(cache.candidates) > limit:
        lines.append("")
        lines.append(f"... {len(cache.candidates) - limit} more candidates not shown")
    return lines


def _file_status_text(candidate: VllmModelCandidate) -> str:
    missing = set(candidate.readiness.missing)
    config = "MISSING" if "config" in missing else "OK"
    tokenizer = "MISSING" if "tokenizer" in missing else "OK"
    weights = "MISSING" if "weights" in missing else "OK"
    suite_profile = "OK" if candidate.has_suite_profile else "MISSING"
    if candidate.source.kind == "gguf_file":
        return "GGUF file OK / suite profile n/a"
    return f"config {config} / tokenizer {tokenizer} / weights {weights} / suite profile {suite_profile}"


def _read_config_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _has_tokenizer_file(path: Path) -> bool:
    return any((path / name).is_file() for name in TOKENIZER_FILES)


def _has_local_model_marker(path: Path) -> bool:
    return (path / "config.json").is_file() or _has_tokenizer_file(path) or _has_weight_file(path)


def _has_safetensors_file(path: Path) -> bool:
    return any(child.is_file() and child.suffix == ".safetensors" for child in path.iterdir())


def _has_weight_file(path: Path) -> bool:
    return any(child.is_file() and child.name.lower().endswith(WEIGHT_SUFFIXES) for child in path.iterdir())


def _guess_quant(name_text: str, config_text: str) -> tuple[str, list[str]]:
    evidence: list[str] = []
    combined = f"{name_text} {config_text}"
    if "awq" in combined:
        if "awq" in name_text:
            evidence.append("path_contains_awq")
        if "awq" in config_text:
            evidence.append("config_contains_awq")
        return "awq", evidence
    if "gptq" in combined:
        if "gptq" in name_text:
            evidence.append("path_contains_gptq")
        if "gptq" in config_text:
            evidence.append("config_contains_gptq")
        return "gptq", evidence
    if "bf16" in combined or "bfloat16" in combined:
        evidence.append("dtype_hint_bf16")
        return "bf16", evidence
    return "unknown", evidence


def _guess_family(name_text: str, config_data: dict[str, Any]) -> tuple[str, list[str]]:
    evidence: list[str] = []
    config_family = str(config_data.get("model_type") or "").lower()
    if config_family:
        evidence.append(f"config_model_type_{config_family}")
        return config_family, evidence
    if "gemma" in name_text:
        evidence.append("path_contains_gemma")
        return "gemma", evidence
    if "qwen" in name_text:
        evidence.append("path_contains_qwen")
        return "qwen", evidence
    if "llama" in name_text:
        evidence.append("path_contains_llama")
        return "llama", evidence
    return "unknown", evidence


def _guess_size_b(name_text: str) -> tuple[int | None, list[str]]:
    import re

    match = re.search(r"(\d{1,3})(?:\.\d+)?\s*b", name_text)
    if not match:
        return None, []
    size_b = int(match.group(1))
    return size_b, [f"path_contains_{size_b}b"]


def _confidence_for(quant: str, evidence: list[str], missing: list[str]) -> float:
    if quant in {"awq", "gptq"}:
        base = 0.75
    elif quant == "bf16":
        base = 0.65
    else:
        base = 0.35
    if "config_json_exists" in evidence:
        base += 0.05
    if "safetensors_weights_found" in evidence:
        base += 0.05
    if missing:
        base -= 0.05
    return round(max(0.1, min(base, 0.95)), 2)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
