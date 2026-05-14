from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from modules.llama_cpp.backends import BACKENDS


DISCOVERY_ROOTS = [
    "~",
]
DENY_ROOTS = {
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/var",
    "/tmp",
    "/lost+found",
}
INSPECTION_DENY_ROOTS = {
    "/proc",
    "/sys",
    "/dev",
    "/run",
}
OPTIONAL_DISCOVERY_PARENTS = {
    "/mnt",
    "/mount",
    "/media",
    "/opt",
}
CANDIDATE_NAMES = {
    "llama-server",
    "llama_server",
}
EXPECTED_FEATURES = {
    "jinja",
    "reasoning",
    "reasoning_budget",
    "chat_template_kwargs",
    "alias",
}


@dataclass(frozen=True)
class DiscoveryFuel:
    max_depth: int = 5
    max_seconds: float = 5.0
    max_dirs: int = 500
    max_files: int = 2000
    max_candidates: int = 20


@dataclass
class DiscoveryResult:
    candidates: list[str] = field(default_factory=list)
    status: str = "ok"
    reason: str | None = None
    scanned_dirs: int = 0
    scanned_files: int = 0
    skipped_dirs: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    probe_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("probe_text", None)
        return data


def discovery_roots() -> list[Path]:
    return default_safe_roots()


def is_root_user() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def display_path(path: str | Path) -> str:
    target = Path(path).expanduser()
    home = Path.home()
    try:
        rel = target.resolve().relative_to(home.resolve())
    except (OSError, ValueError):
        return str(target)
    rel_s = str(rel)
    return "$HOME" if rel_s == "." else f"$HOME/{rel_s}"


def default_safe_roots() -> list[Path]:
    if is_root_user():
        return []
    home = Path.home()
    return [] if is_denied_path(home) else [home]


def optional_discovery_parents() -> list[Path]:
    parents: list[Path] = []
    for parent in sorted(OPTIONAL_DISCOVERY_PARENTS):
        path = Path(parent)
        if path.exists() and not is_denied_path(path):
            parents.append(path)
    return parents


def is_denied_path(path: str | Path) -> bool:
    return _is_denied(Path(path))


def discovery_choices() -> dict[str, Any]:
    return {
        "safe_roots": [display_path(path) for path in default_safe_roots()],
        "optional_parents": [display_path(path) for path in optional_discovery_parents()],
        "denied_roots": sorted(DENY_ROOTS),
        "warnings": [
            "root 권한은 편의 기능이 아니라 폭약입니다.",
            "backend discovery는 일반 사용자 계정에서만 실행됩니다.",
        ] if is_root_user() else [],
    }


def discover_llama_server_candidates(
    root: str | Path,
    fuel: DiscoveryFuel | None = None,
) -> DiscoveryResult:
    started = time.monotonic()
    fuel = fuel or DiscoveryFuel()
    result = DiscoveryResult()

    if is_root_user():
        result.status = "denied"
        result.reason = "root discovery denied"
        result.warnings.extend([
            "root 권한은 편의 기능이 아니라 폭약입니다.",
            "backend discovery는 일반 사용자 계정에서만 실행됩니다.",
        ])
        result.elapsed_seconds = time.monotonic() - started
        return result

    root_path = Path(root).expanduser()
    if _is_denied(root_path):
        result.status = "skipped"
        result.reason = "path is denied"
        result.skipped_dirs.append(display_path(root_path))
        result.elapsed_seconds = time.monotonic() - started
        return result

    if not root_path.exists():
        result.status = "skipped"
        result.reason = "path does not exist"
        result.skipped_dirs.append(display_path(root_path))
        result.elapsed_seconds = time.monotonic() - started
        return result

    seen: set[str] = set()

    if root_path.is_file():
        result.scanned_files += 1
        if root_path.name in CANDIDATE_NAMES:
            _append_candidate(result.candidates, seen, root_path)
        result.elapsed_seconds = time.monotonic() - started
        return result

    if not root_path.is_dir():
        result.status = "skipped"
        result.reason = "path is neither file nor directory"
        result.elapsed_seconds = time.monotonic() - started
        return result

    stack: list[tuple[Path, int]] = [(root_path, 0)]
    while stack:
        limit_reason = _fuel_limit_reason(result, fuel, started)
        if limit_reason:
            result.status = "stopped"
            result.reason = limit_reason
            break

        current, depth = stack.pop()
        if _is_denied(current):
            result.skipped_dirs.append(display_path(current))
            continue
        if current.is_symlink():
            result.skipped_dirs.append(f"{display_path(current)} (symlink)")
            continue

        result.scanned_dirs += 1
        try:
            entries = list(current.iterdir())
        except OSError as e:
            result.skipped_dirs.append(display_path(current))
            result.warnings.append(f"skipped {display_path(current)}: {e}")
            continue

        for entry in entries:
            limit_reason = _fuel_limit_reason(result, fuel, started)
            if limit_reason:
                result.status = "stopped"
                result.reason = limit_reason
                break

            if _is_denied(entry):
                if entry.is_dir():
                    result.skipped_dirs.append(display_path(entry))
                continue
            if entry.is_symlink():
                if not entry.exists():
                    result.warnings.append(f"skipped broken symlink: {display_path(entry)}")
                continue

            try:
                if entry.is_file():
                    result.scanned_files += 1
                    if entry.name in CANDIDATE_NAMES:
                        _append_candidate(result.candidates, seen, entry)
                    continue
                if entry.is_dir() and depth < fuel.max_depth:
                    stack.append((entry, depth + 1))
            except OSError as e:
                result.warnings.append(f"skipped {display_path(entry)}: {e}")
                continue

        if result.status == "stopped":
            break

    result.candidates.sort()
    result.elapsed_seconds = time.monotonic() - started
    return result


def inspect_backend_binary(path: str | Path) -> BackendInspection:
    target = Path(path).expanduser()
    warnings: list[str] = []
    evidence: list[str] = []
    raw: dict[str, str] = {}

    if _is_under_roots(target, INSPECTION_DENY_ROOTS):
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
    full_help = _run_full([str(target), "--help"], timeout=3) if executable else ""
    raw["help"] = _summarize_full_output(full_help)
    raw["version"] = summarize_version(target) if executable else ""
    probe_text = _combined_text(raw) + "\n" + full_help.lower()

    _add_static_evidence(target, raw, evidence, warnings, probe_text=probe_text)
    backend_guess = guess_backend(raw, str(target), probe_text=probe_text)
    supported_features = detect_supported_features(raw, probe_text=probe_text)
    missing_features = detect_missing_features(raw, probe_text=probe_text)

    if "cuda" in probe_text and backend_guess != "cuda":
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
        probe_text=probe_text,
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


def guess_backend(raw: dict[str, str], path: str, probe_text: str | None = None) -> str | None:
    text = probe_text if probe_text is not None else _combined_text(raw)
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

    if "exaone" in path_l:
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


def detect_supported_features(raw: dict[str, str], probe_text: str | None = None) -> list[str]:
    text = probe_text if probe_text is not None else _combined_text(raw)
    features: set[str] = set()

    if "--jinja" in text:
        features.add("jinja")
    if "--reasoning" in text:
        features.add("reasoning")
    if "--reasoning-budget" in text:
        features.add("reasoning_budget")
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

    backend = guess_backend(raw, "", probe_text=probe_text)
    if backend in {"rocm", "vulkan", "cuda", "cpu"}:
        features.add(backend)
    if backend == "exaone_fork":
        features.add("exaone_fork")

    return sorted(features)


def detect_missing_features(raw: dict[str, str], probe_text: str | None = None) -> list[str]:
    supported = set(detect_supported_features(raw, probe_text=probe_text))
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
    probe_text = result.probe_text or _combined_text(raw)

    if "elf" in file_text or "executable" in file_text:
        score += 10
    if ldd_text and "error:" not in ldd_text and "not a dynamic executable" not in ldd_text:
        score += 10
    if help_text and "error:" not in help_text and "timeout" not in help_text:
        score += 15
    if version_text and "error:" not in version_text and "timeout" not in version_text:
        score += 5
    if "llama" in probe_text:
        score += 10
    if result.backend_guess in {"rocm", "vulkan", "cuda", "cpu", "exaone_fork"}:
        score += 10

    for feature in ("jinja", "reasoning", "reasoning_budget", "chat_template_kwargs", "alias"):
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
    if "reasoning_budget" in result.missing_features:
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


def _append_candidate(candidates: list[str], seen: set[str], path: Path) -> None:
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    if key in seen:
        return
    seen.add(key)
    candidates.append(key)


def _fuel_limit_reason(result: DiscoveryResult, fuel: DiscoveryFuel, started: float) -> str | None:
    if time.monotonic() - started > fuel.max_seconds:
        return "max_seconds exceeded"
    if result.scanned_dirs >= fuel.max_dirs:
        return "max_dirs exceeded"
    if result.scanned_files >= fuel.max_files:
        return "max_files exceeded"
    if len(result.candidates) >= fuel.max_candidates:
        return "max_candidates exceeded"
    return None


def _is_denied(path: Path) -> bool:
    return _is_under_roots(path, DENY_ROOTS)


def _is_under_roots(path: Path, roots: set[str]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    return any(str(resolved) == root or str(resolved).startswith(root + os.sep) for root in roots)


def _run_summary(args: list[str], timeout: int) -> str:
    return _summarize_full_output(_run_full(args, timeout=timeout))


def _run_full(args: list[str], timeout: int) -> str:
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
    if result.returncode != 0:
        prefix = f"exit {result.returncode}"
        return f"{prefix}\n{output}" if output else prefix
    return output


def _summarize_full_output(text: str) -> str:
    return _trim_lines(text)


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
    probe_text: str | None = None,
) -> None:
    text = probe_text if probe_text is not None else _combined_text(raw)
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
