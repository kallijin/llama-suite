from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_WRAPPER_PATH = "~/bin/vllm-rocm"
DEFAULT_VLLM_CACHE_ROOT = "/mnt/data_main/ai-cache/vllm"
DEFAULT_HF_HOME = "/mnt/data_main/ai-cache/huggingface"
DEFAULT_TRANSFORMERS_CACHE = "/mnt/data_main/ai-cache/huggingface"


@dataclass
class VllmProfile:
    wrapper_path: str = DEFAULT_WRAPPER_PATH
    model: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.70
    tensor_parallel_size: int = 1
    vllm_cache_root: str = DEFAULT_VLLM_CACHE_ROOT
    hf_home: str = DEFAULT_HF_HOME
    transformers_cache: str = DEFAULT_TRANSFORMERS_CACHE
    extra_args: str = ""
    errors: list[str] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("errors", None)
        return data


def default_vllm_profile(model: str = "") -> VllmProfile:
    return VllmProfile(model=model)


def vllm_profile_from_dict(data: dict[str, Any]) -> VllmProfile:
    profile = VllmProfile()
    for key in profile.to_dict():
        if key in data:
            setattr(profile, key, data[key])
    profile.port = _coerce_int(profile.port, default=8000)
    profile.max_model_len = _coerce_int(profile.max_model_len, default=4096)
    profile.tensor_parallel_size = _coerce_int(profile.tensor_parallel_size, default=1)
    profile.gpu_memory_utilization = _coerce_float(profile.gpu_memory_utilization, default=0.70)
    profile.extra_args = str(profile.extra_args or "")
    return profile


def validate_vllm_profile(profile: VllmProfile) -> list[str]:
    errors: list[str] = []

    if not str(profile.wrapper_path).strip():
        errors.append("wrapper path should not be empty")
    if not str(profile.model).strip():
        errors.append("model should not be empty")

    port = _try_int(profile.port)
    if port is None or not 1 <= port <= 65535:
        errors.append("port should be 1-65535")

    gpu_memory_utilization = _try_float(profile.gpu_memory_utilization)
    if gpu_memory_utilization is None or not 0 < gpu_memory_utilization <= 1:
        errors.append("gpu_memory_utilization should be between 0 and 1")

    tensor_parallel_size = _try_int(profile.tensor_parallel_size)
    if tensor_parallel_size is None or tensor_parallel_size < 1:
        errors.append("tensor_parallel_size should be >= 1")

    max_model_len = _try_int(profile.max_model_len)
    if max_model_len is None or max_model_len <= 0:
        errors.append("max_model_len should be > 0")

    return errors


def host_guidance_lines() -> list[str]:
    return [
        "127.0.0.1 = local only",
        "Tailscale IP = private remote access",
        "0.0.0.0 = advanced/exposed",
    ]


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _try_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
