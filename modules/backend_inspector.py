from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from modules.backends import BACKENDS


DISCOVERY_ROOTS = [
    "~/src",
    "~/.local/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt",
]
DENY_ROOTS = {
    "/proc",
    "/sys",
    "/dev",
    "/run",
}
CANDIDATE_NAMES = {
    "llama-server",
    "llama_server",
}
EXPECTED_FEATURES = {
    "jinja",
    "reasoning",
    "chat_template_kwargs",
    "alias",
}


@dataclass
class BackendInspection:
    path: str
    exists: bool
    executable: bool
    status: str
    human_label: str
    backend_guess: str | None
    score: int
    supported_features: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discovery_roots() -> list[Path]:
    roots: list[Path] = []
    for root in DISCOVERY_ROOTS:
        path = Path(root).expanduser()
        if _is_denied(path):
            continue
        roots.append(path)
    return roots


def discover_llama_server_candidates(
    roots: list[str | Path] | None = None,
    max_depth: int = 5,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    search_roots = [Path(p).expanduser() for p in roots] if roots is not None else discovery_roots()

    for root in search_roots:
        if _is_denied(root) or not root.exists():
            continue

        if root.is_file() and root.name in CANDIDATE_NAMES:
            _append_candidate(candidates, seen, root)
            continue

        if not root.is_dir():
            continue

        root_depth = len(root.resolve().parts)
        stack = [root]
        while stack:
            current = stack.pop()
            if _is_denied(current):
                continue
            try:
                entries = list(current.iterdir())
            except OSError:
                continue

            for entry in entries:
                if _is_denied(entry):
                    continue
                if entry.is_file() and entry.name in CANDIDATE_NAMES:
                    _append_candidate(candidates, seen, entry)
                    continue
                if entry.is_dir():
                    try:
                        depth = len(entry.resolve().parts) - root_depth
                    except OSError:
                        continue
                    if depth <= max_depth:
                        stack.append(entry)

    return sorted(candidates, key=lambda p: str(p))


def inspect_backend_binary(path: str | Path) -> BackendInspection:
    target = Path(path).expanduser()
    warnings: list[str] = []
    evidence: list[str] = []
    raw: dict[str, str] = {}

    if _is_denied(target):
        warnings.append("path is under a denied discovery root")
        result = BackendInspection(
            path=str(target),
            exists=False,
            executable=False,
            status="denied",
            human_label="시동도 안 걸리는 폐차",
            backend_guess=None,
            score=0,
            warnings=warnings,
            raw=raw,
        )
        return result

    exists = target.exists()
    executable = target.is_file() and os.access(target, os.X_OK)

    if not exists:
        warnings.append("path does not exist")
        result = BackendInspection(
            path=str(target),
            exists=False,
            executable=False,
            status="missing",
            human_label="시동도 안 걸리는 폐차",
            backend_guess=None,
            score=0,
            warnings=warnings,
            raw=raw,
        )
        return result

    if not target.is_file():
        warnings.append("path exists but is not a file")

    raw["file"] = summarize_file(target)
    raw["ldd"] = summarize_ldd(target)
    raw["help"] = summarize_help(target) if executable else ""
    raw["version"] = summarize_version(target) if executable else ""

    _add_static_evidence(target, raw, evidence, warnings)
    backend_guess = guess_backend(raw, str(target))
    supported_features = detect_supported_features(raw)
    missing_features = detect_missing_features(raw)

    if "cuda" in _combined_text(raw) and backend_guess != "cuda":
        warnings.append(
            "cuda-like strings can appear in GGML/HIP builds; backend was not judged as CUDA from that alone"
        )

    result = BackendInspection(
        path=str(target),
        exists=exists,
        executable=executable,
        status="ok" if executable else "not_executable",
        human_label="",
        backend_guess=backend_guess,
        score=0,
        supported_features=supported_features,
        missing_features=missing_features,
        evidence=evidence,
        warnings=warnings,
        raw=raw,
    )
    result.score = score_inspection(result)
    result.human_label = label_from_score(result)
    return result


def inspect_known_backend_candidates() -> list[BackendInspection]:
    paths: list[str] = []
    seen: set[str] = set()
    for backend in BACKENDS.values():
        for candidate in backend.server_candidates:
            if candidate == "llama-server":
                for directory in os.environ.get("PATH", "").split(os.pathsep):
                    if not directory:
                        continue
                    resolved = str(Path(directory) / "llama-server")
                    if resolved not in seen:
                        seen.add(resolved)
                        paths.append(resolved)
                continue

            resolved = str(Path(candidate).expanduser())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    return [inspect_backend_binary(path) for path in paths]


def summarize_help(path: Path, timeout: int = 3) -> str:
    return _run_summary([str(path), "--help"], timeout=timeout)


def summarize_version(path: Path, timeout: int = 3) -> str:
    return _run_summary([str(path), "--version"], timeout=timeout)


def summarize_file(path: Path, timeout: int = 3) -> str:
    return _run_summary(["file", str(path)], timeout=timeout)


def summarize_ldd(path: Path, timeout: int = 3) -> str:
    return _run_summary(["ldd", str(path)], timeout=timeout)


def guess_backend(raw: dict[str, str], path: str) -> str | None:
    text = _combined_text(raw)
    path_l = path.lower()

    rocm_hits = _count_hits(
        text + "\n" + path_l,
        ["libamdhip64", "librocblas", "libhipblas", "libhsa-runtime64", "rocm", "hip"],
    )
    cuda_hits = _count_hits(
        text + "\n" + path_l,
        ["libcuda", "libcudart", "libcublas", "nvidia"],
    )
    vulkan_hits = _count_hits(text + "\n" + path_l, ["libvulkan", "vulkan"])

    if "exaone" in text or "exaone" in path_l:
        return "exaone_fork"
    if rocm_hits:
        return "rocm"
    if vulkan_hits:
        return "vulkan"
    if cuda_hits >= 1:
        return "cuda"
    if "llama.cpp" in text or "llama-server" in text or "llama_server" in text:
        return "unknown_llama_cpp"
    if "cpu" in text:
        return "cpu"
    return None


def detect_supported_features(raw: dict[str, str]) -> list[str]:
    text = _combined_text(raw)
    features: set[str] = set()

    if "--jinja" in text:
        features.add("jinja")
    if "--reasoning" in text:
        features.add("reasoning")
    if "--chat-template-kwargs" in text:
        features.add("chat_template_kwargs")
    if "--alias" in text:
        features.add("alias")
    if "/v1" in text or "openai" in text:
        features.add("openai_api")
    if "embedding" in text or "embeddings" in text:
        features.add("embeddings")
    if "rerank" in text or "reranking" in text:
        features.add("reranking")

    backend = guess_backend(raw, "")
    if backend in {"rocm", "vulkan", "cuda", "cpu"}:
        features.add(backend)
    if backend == "exaone_fork":
        features.add("exaone_fork")

    return sorted(features)


def detect_missing_features(raw: dict[str, str]) -> list[str]:
    supported = set(detect_supported_features(raw))
    return sorted(EXPECTED_FEATURES - supported)


def score_inspection(result: BackendInspection) -> int:
    if not result.exists:
        return 0

    score = 10
    if result.executable:
        score += 15

    raw = result.raw
    file_text = raw.get("file", "").lower()
    ldd_text = raw.get("ldd", "").lower()
    help_text = raw.get("help", "").lower()
    version_text = raw.get("version", "").lower()

    if "elf" in file_text or "executable" in file_text:
        score += 10
    if ldd_text and "error:" not in ldd_text and "not a dynamic executable" not in ldd_text:
        score += 10
    if help_text and "error:" not in help_text and "timeout" not in help_text:
        score += 15
    if version_text and "error:" not in version_text and "timeout" not in version_text:
        score += 5
    if "llama" in _combined_text(raw):
        score += 10
    if result.backend_guess in {"rocm", "vulkan", "cuda", "cpu", "exaone_fork"}:
        score += 10

    for feature in ("jinja", "reasoning", "chat_template_kwargs", "alias"):
        if feature in result.supported_features:
            score += 5

    if "timeout" in help_text or "timeout" in version_text:
        score -= 10
    if "not a dynamic executable" in ldd_text:
        score -= 5
    if "jinja" in result.missing_features:
        score -= 5
    if "alias" in result.missing_features:
        score -= 5
    if "chat_template_kwargs" in result.missing_features:
        score -= 5
    if "reasoning" in result.missing_features:
        score -= 3
    if result.backend_guess == "cuda" and ("rocm" in result.path.lower() or "hip" in result.path.lower()):
        score -= 10

    return max(0, min(100, score))


def label_from_score(result: BackendInspection) -> str:
    if not result.exists or not result.executable:
        return "시동도 안 걸리는 폐차"
    if result.backend_guess == "unknown_llama_cpp" and result.score < 50:
        return "족보 불명 llama.cpp"
    if result.score >= 85:
        return "왕족 엔진"
    if result.score >= 70:
        return "쓸만한 엔진"
    if result.score >= 50:
        return "굴러는 감"
    if result.score >= 30:
        return "수상한 엔진"
    if result.backend_guess == "unknown_llama_cpp":
        return "족보 불명 llama.cpp"
    return "시동도 안 걸리는 폐차"


def _append_candidate(candidates: list[Path], seen: set[str], path: Path) -> None:
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    if key in seen:
        return
    seen.add(key)
    candidates.append(path)


def _is_denied(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    return any(str(resolved) == root or str(resolved).startswith(root + os.sep) for root in DENY_ROOTS)


def _run_summary(args: list[str], timeout: int) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return f"error: command not found: {e.filename}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout}s"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

    output = "\n".join(x for x in [result.stdout, result.stderr] if x)
    summary = _trim_lines(output)
    if result.returncode != 0:
        prefix = f"exit {result.returncode}"
        return f"{prefix}\n{summary}" if summary else prefix
    return summary


def _trim_lines(text: str, max_lines: int = 80, max_chars: int = 6000) -> str:
    lines = text.splitlines()
    trimmed = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        trimmed += f"\n... truncated {len(lines) - max_lines} lines"
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "\n... truncated chars"
    return trimmed


def _combined_text(raw: dict[str, str]) -> str:
    return "\n".join(raw.values()).lower()


def _count_hits(text: str, needles: list[str]) -> int:
    text_l = text.lower()
    return sum(1 for needle in needles if needle in text_l)


def _add_static_evidence(
    path: Path,
    raw: dict[str, str],
    evidence: list[str],
    warnings: list[str],
) -> None:
    text = _combined_text(raw)
    path_l = str(path).lower()

    if "llama" in path_l:
        evidence.append("path contains llama")
    if "rocm" in path_l or "hip" in path_l:
        evidence.append("path hints ROCm/HIP")
    if "vulkan" in path_l:
        evidence.append("path hints Vulkan")
    if "exaone" in path_l:
        evidence.append("path hints EXAONE fork")
    if "elf" in raw.get("file", "").lower():
        evidence.append("file reports ELF")
    if "llama" in text:
        evidence.append("raw output contains llama")
    if "libamdhip64" in text or "librocblas" in text or "libhipblas" in text:
        evidence.append("ldd hints ROCm/HIP libraries")
    if "libvulkan" in text:
        evidence.append("ldd hints Vulkan library")
    if "libcuda" in text or "libcudart" in text or "libcublas" in text:
        evidence.append("ldd hints CUDA libraries")
    if "not a dynamic executable" in raw.get("ldd", "").lower():
        warnings.append("ldd reports not a dynamic executable")
