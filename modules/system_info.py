from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_INFO_CACHE_PATH = Path(os.path.expanduser("~/.hermes/llama-system-info-cache.json"))
SYSTEM_INFO_CACHE_SECONDS = 300


@dataclass
class SystemInfo:
    kernel: str | None
    arch: str | None
    gpu_vendor_guess: str | None
    gpu_devices: list[str] = field(default_factory=list)
    rocm_available: bool = False
    rocm_summary: str | None = None
    vulkan_available: bool = False
    vulkan_summary: str | None = None
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_system_info(timeout: int = 5, use_cache: bool = True) -> SystemInfo:
    if use_cache:
        cached = load_cached_system_info(max_age_seconds=SYSTEM_INFO_CACHE_SECONDS)
        if cached is not None:
            return cached

    info = collect_system_info_uncached(timeout=timeout)
    if use_cache:
        save_cached_system_info(info)
    return info


def collect_system_info_uncached(timeout: int = 5) -> SystemInfo:
    raw = {
        "uname_r": run_capture(["uname", "-r"], timeout=timeout),
        "uname_m": run_capture(["uname", "-m"], timeout=timeout),
        "rocm_smi": run_capture(
            ["rocm-smi", "--showproductname", "--showmemuse", "--showuse"],
            timeout=timeout,
        ),
        "rocminfo": run_capture(["rocminfo"], timeout=timeout),
        "hipcc_version": run_capture(["hipcc", "--version"], timeout=timeout),
        "vulkaninfo_summary": run_capture(["vulkaninfo", "--summary"], timeout=timeout),
        "lspci_gpu": _filter_lspci_gpu(run_capture(["lspci"], timeout=timeout)),
    }

    rocm_available, rocm_summary = summarize_rocm(raw)
    vulkan_available, vulkan_summary = summarize_vulkan(raw)
    gpu_vendor_guess = guess_gpu_vendor(raw)
    warnings = system_warnings(raw, gpu_vendor_guess)
    if rocm_available and not any("cuda prefix" in w for w in warnings):
        warnings.append(
            "AMD ROCm/HIP 환경에서도 llama.cpp/GGML 로그나 내부 함수명에 cuda prefix가 보일 수 있습니다."
        )
        warnings.append(
            "cuda 문자열 하나만으로 CUDA backend라고 판정하지 말고 rocm-smi/rocminfo, ldd, 라이브러리, 빌드 경로, detected device 근거를 함께 봐야 합니다."
        )

    return SystemInfo(
        kernel=_clean_value(raw.get("uname_r")),
        arch=_clean_value(raw.get("uname_m")),
        gpu_vendor_guess=gpu_vendor_guess,
        gpu_devices=extract_gpu_devices(raw),
        rocm_available=rocm_available,
        rocm_summary=rocm_summary,
        vulkan_available=vulkan_available,
        vulkan_summary=vulkan_summary,
        warnings=warnings,
        raw=raw,
    )


def load_cached_system_info(
    max_age_seconds: int = SYSTEM_INFO_CACHE_SECONDS,
    cache_path: str | os.PathLike[str] = SYSTEM_INFO_CACHE_PATH,
) -> SystemInfo | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    try:
        with path.open() as f:
            payload = json.load(f)
    except Exception:
        return None

    try:
        created_at = datetime.fromisoformat(str(payload.get("created_at")))
    except Exception:
        return None

    age = (datetime.now(timezone.utc) - created_at).total_seconds()
    if age > max_age_seconds:
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    try:
        return SystemInfo(
            kernel=data.get("kernel"),
            arch=data.get("arch"),
            gpu_vendor_guess=data.get("gpu_vendor_guess"),
            gpu_devices=list(data.get("gpu_devices") or []),
            rocm_available=bool(data.get("rocm_available")),
            rocm_summary=data.get("rocm_summary"),
            vulkan_available=bool(data.get("vulkan_available")),
            vulkan_summary=data.get("vulkan_summary"),
            warnings=list(data.get("warnings") or []),
            raw=dict(data.get("raw") or {}),
        )
    except Exception:
        return None


def save_cached_system_info(
    info: SystemInfo,
    cache_path: str | os.PathLike[str] = SYSTEM_INFO_CACHE_PATH,
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": info.to_dict(),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_rocm(raw: dict[str, str]) -> tuple[bool, str | None]:
    parts: list[str] = []
    available = False

    rocm_smi = raw.get("rocm_smi", "")
    rocminfo = raw.get("rocminfo", "")
    hipcc = raw.get("hipcc_version", "")

    if _command_ok(rocm_smi):
        available = True
        parts.append("rocm-smi ok")
    else:
        parts.append(_short_failure("rocm-smi", rocm_smi))

    if _command_ok(rocminfo) and _has_any(rocminfo, ["agent", "gfx", "amd", "hsa"]):
        available = True
        parts.append("rocminfo reports ROCm/HSA agents")
    else:
        parts.append(_short_failure("rocminfo", rocminfo))

    if _command_ok(hipcc):
        parts.append("hipcc found")
    else:
        parts.append(_short_failure("hipcc", hipcc))

    return available, ", ".join(x for x in parts if x)


def summarize_vulkan(raw: dict[str, str]) -> tuple[bool, str | None]:
    vulkan = raw.get("vulkaninfo_summary", "")
    if _command_ok(vulkan):
        return True, "vulkaninfo summary ok"
    return False, _short_failure("vulkaninfo", vulkan)


def guess_gpu_vendor(raw: dict[str, str]) -> str | None:
    text = "\n".join(raw.values()).lower()
    vendors = set()

    if _has_any(text, ["amd", "advanced micro devices", "radeon", "gfx"]):
        vendors.add("amd")
    if _has_any(text, ["nvidia", "geforce", "libcuda", "cublas"]):
        vendors.add("nvidia")
    if _has_any(text, ["intel", "arc graphics"]):
        vendors.add("intel")

    if len(vendors) > 1:
        return "mixed"
    if vendors:
        return next(iter(vendors))
    return None


def extract_gpu_devices(raw: dict[str, str]) -> list[str]:
    devices: list[str] = []
    for key in ("lspci_gpu", "rocm_smi", "vulkaninfo_summary"):
        text = raw.get(key, "")
        if not text or text.startswith("error:") or text.startswith("timeout"):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if _has_any(line.lower(), ["vga", "3d controller", "display", "amd", "radeon", "nvidia", "geforce", "intel", "arc"]):
                if line not in devices:
                    devices.append(line)
    return devices[:20]


def system_warnings(raw: dict[str, str], gpu_vendor_guess: str | None) -> list[str]:
    warnings: list[str] = []

    for name, text in raw.items():
        if text.startswith("error: command not found"):
            warnings.append(f"{name}: command not found")
        elif text.startswith("timeout"):
            warnings.append(f"{name}: {text}")
        elif text.startswith("exit "):
            first = text.splitlines()[0]
            warnings.append(f"{name}: {first}")

    if gpu_vendor_guess == "amd":
        warnings.append(
            "AMD ROCm/HIP 환경에서도 llama.cpp/GGML 로그나 내부 함수명에 cuda prefix가 보일 수 있습니다."
        )
        warnings.append(
            "cuda 문자열 하나만으로 CUDA backend라고 판정하지 말고 rocm-smi/rocminfo, ldd, 라이브러리, 빌드 경로, detected device 근거를 함께 봐야 합니다."
        )

    return _dedupe(warnings)


def run_capture(args: list[str], timeout: int) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return f"error: command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout}s"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

    output = "\n".join(x for x in [result.stdout, result.stderr] if x)
    summary = _trim_text(output)
    if result.returncode != 0:
        prefix = f"exit {result.returncode}"
        return f"{prefix}\n{summary}" if summary else prefix
    return summary


def _filter_lspci_gpu(text: str) -> str:
    if text.startswith("error:") or text.startswith("timeout") or text.startswith("exit "):
        return text
    keep: list[str] = []
    needles = ["vga", "3d controller", "display controller", "amd", "radeon", "nvidia", "geforce", "intel", "arc"]
    for line in text.splitlines():
        if _has_any(line.lower(), needles):
            keep.append(line)
    return _trim_text("\n".join(keep))


def _trim_text(text: str, max_lines: int = 80, max_chars: int = 6000) -> str:
    lines = text.splitlines()
    trimmed = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        trimmed += f"\n... truncated {len(lines) - max_lines} lines"
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "\n... truncated chars"
    return trimmed


def _command_ok(text: str) -> bool:
    return bool(text) and not (
        text.startswith("error:")
        or text.startswith("timeout")
        or text.startswith("exit ")
    )


def _short_failure(name: str, text: str) -> str:
    if not text:
        return f"{name} empty"
    return text.splitlines()[0]


def _has_any(text: str, needles: list[str]) -> bool:
    text_l = text.lower()
    return any(needle in text_l for needle in needles)


def _clean_value(text: str | None) -> str | None:
    if not text:
        return None
    if text.startswith("error:") or text.startswith("timeout") or text.startswith("exit "):
        return None
    first = text.splitlines()[0].strip()
    return first or None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
