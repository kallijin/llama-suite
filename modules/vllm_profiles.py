from __future__ import annotations

import shlex
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_WRAPPER_PATH = "~/bin/vllm-rocm"
DEFAULT_VLLM_CACHE_ROOT = "/mnt/data_main/ai-cache/vllm"
DEFAULT_HF_HOME = "/mnt/data_main/ai-cache/huggingface"
DEFAULT_TRANSFORMERS_CACHE = "/mnt/data_main/ai-cache/huggingface"
FUTURE_LAUNCH_PRESET_ID = "smoke-qwen-0.5b"


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


@dataclass(frozen=True)
class VllmProfilePreset:
    id: str
    label: str
    description: str
    profile: VllmProfile


@dataclass
class VllmPreflightCheck:
    name: str
    ok: bool
    message: str


@dataclass
class VllmPreflightReport:
    checks: list[VllmPreflightCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def default_vllm_profile(model: str = "") -> VllmProfile:
    return VllmProfile(model=model)


def smoke_vllm_profile() -> VllmProfile:
    return VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")


def builtin_vllm_profile_presets() -> list[VllmProfilePreset]:
    return [
        VllmProfilePreset(
            id="default",
            label="Default vLLM profile",
            description="Blank read-only starting point. It is expected to need a model before launch.",
            profile=default_vllm_profile(),
        ),
        VllmProfilePreset(
            id="smoke-qwen-0.5b",
            label="Smoke Qwen 0.5B",
            description="Known working read-only smoke-test profile for this system.",
            profile=smoke_vllm_profile(),
        ),
    ]


def future_launch_preset_id() -> str:
    return FUTURE_LAUNCH_PRESET_ID


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


def build_vllm_command(profile: VllmProfile) -> tuple[list[str] | None, list[str]]:
    messages = validate_vllm_profile(profile)
    if messages:
        return None, messages

    try:
        extra_args = shlex.split(str(profile.extra_args or ""))
    except ValueError as exc:
        return None, [f"extra_args could not be parsed: {exc}"]

    command = [
        str(profile.wrapper_path),
        "serve",
        str(profile.model),
        "--host",
        str(profile.host),
        "--port",
        str(profile.port),
        "--dtype",
        str(profile.dtype),
        "--max-model-len",
        str(profile.max_model_len),
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
    ]
    command.extend(extra_args)
    return command, []


def cache_env_preview_lines(profile: VllmProfile) -> list[str]:
    return [
        f"VLLM_CACHE_ROOT={profile.vllm_cache_root}",
        f"HF_HOME={profile.hf_home}",
        f"TRANSFORMERS_CACHE={profile.transformers_cache}",
    ]


def run_vllm_preflight(profile: VllmProfile, port_check: Any = None) -> VllmPreflightReport:
    checks: list[VllmPreflightCheck] = []

    validation_messages = validate_vllm_profile(profile)
    checks.append(
        VllmPreflightCheck(
            "profile validation",
            not validation_messages,
            "; ".join(validation_messages) if validation_messages else "profile values look valid",
        )
    )

    checks.append(_wrapper_preflight_check(profile.wrapper_path))

    command, command_messages = build_vllm_command(profile)
    checks.append(
        VllmPreflightCheck(
            "command preview",
            command is not None,
            "; ".join(command_messages) if command_messages else "command preview can be built",
        )
    )

    check_port = port_check or _port_preflight_check
    checks.append(check_port(profile.host, profile.port))
    checks.append(VllmPreflightCheck("host guidance", True, host_access_note(str(profile.host))))
    return VllmPreflightReport(checks)


def host_access_note(host: str) -> str:
    host = str(host).strip()
    if host == "127.0.0.1":
        return "127.0.0.1 = local only"
    if host == "0.0.0.0":
        return "0.0.0.0 = advanced/exposed"
    if _looks_like_tailscale_ip(host):
        return "Tailscale IP = private remote access"
    return "custom host; review network exposure before launch"


def host_guidance_lines() -> list[str]:
    return [
        "127.0.0.1 = local only",
        "Tailscale IP = private remote access",
        "0.0.0.0 = advanced/exposed",
    ]


def launch_confirmation_guidance_lines() -> list[str]:
    return [
        "This is guidance only; no launch button is implemented yet.",
        "vLLM launch may download model files if they are not cached.",
        "vLLM may use GPU memory immediately.",
        "vLLM may run torch compile / graph capture on first launch.",
        "The selected host/port will be bound by the server.",
        "127.0.0.1 is local-only.",
        "Tailscale IP is for private remote access.",
        "0.0.0.0 is advanced/exposed.",
    ]


def _wrapper_preflight_check(wrapper_path: Any) -> VllmPreflightCheck:
    path_text = str(wrapper_path or "").strip()
    if not path_text:
        return VllmPreflightCheck("wrapper executable", False, "wrapper path should not be empty")

    path = Path(path_text).expanduser()
    if not path.exists():
        return VllmPreflightCheck("wrapper executable", False, f"wrapper path does not exist: {path}")
    if not path.is_file():
        return VllmPreflightCheck("wrapper executable", False, f"wrapper path is not a file: {path}")
    if not path.stat().st_mode & 0o111:
        return VllmPreflightCheck("wrapper executable", False, f"wrapper path is not executable: {path}")
    return VllmPreflightCheck("wrapper executable", True, f"wrapper executable found: {path}")


def _port_preflight_check(host: Any, port: Any) -> VllmPreflightCheck:
    port_number = _try_int(port)
    if port_number is None or not 1 <= port_number <= 65535:
        return VllmPreflightCheck("port availability", False, "port should be 1-65535")

    bind_host = "127.0.0.1" if str(host).strip() == "0.0.0.0" else str(host).strip()
    if not bind_host:
        bind_host = "127.0.0.1"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port_number))
    except OSError as exc:
        return VllmPreflightCheck(
            "port availability",
            False,
            f"port {port_number} is not available on {bind_host}: {exc}",
        )
    return VllmPreflightCheck("port availability", True, f"port {port_number} is available on {bind_host}")


def _looks_like_tailscale_ip(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return False
    return nums[0] == 100 and all(0 <= num <= 255 for num in nums)


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
