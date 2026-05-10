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
DEFAULT_MODEL_DOWNLOAD_ROOT = "/mnt/data_main/downloads/models"
FUTURE_LAUNCH_PRESET_ID = "smoke-qwen-0.5b"
VLLM_EDITABLE_PROFILE_FIELDS = [
    "wrapper_path",
    "model",
    "host",
    "port",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "kv_cache_dtype",
    "max_num_seqs",
    "max_num_batched_tokens",
    "vllm_cache_root",
    "hf_home",
    "transformers_cache",
    "extra_args",
]


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
    kv_cache_dtype: str = "auto"
    max_num_seqs: int | str = ""
    max_num_batched_tokens: int | str = ""
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


@dataclass(frozen=True)
class VllmProfileFieldSpec:
    name: str
    group: str
    label: str
    help: str
    input_hint: str
    example: str


@dataclass(frozen=True)
class VllmModelSourceCheck:
    level: str
    name: str
    message: str


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


def local_large_q4_vllm_profile() -> VllmProfile:
    return VllmProfile(
        model=f"{DEFAULT_MODEL_DOWNLOAD_ROOT}/local-large-q4-hf",
        max_model_len=8192,
        gpu_memory_utilization=0.82,
        kv_cache_dtype="auto",
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        extra_args="--served-model-name local-large-q4",
    )


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
        VllmProfilePreset(
            id="template-local-large-q4",
            label="Local large Q4 template",
            description="Read-only template for a local large HF/safetensors quantized model directory. It is not a launch target.",
            profile=local_large_q4_vllm_profile(),
        ),
    ]


def future_launch_preset_id() -> str:
    return FUTURE_LAUNCH_PRESET_ID


def editable_vllm_profile_fields() -> list[str]:
    return list(VLLM_EDITABLE_PROFILE_FIELDS)


def editable_vllm_profile_field_specs() -> list[VllmProfileFieldSpec]:
    return [
        VllmProfileFieldSpec("wrapper_path", "Runtime", "Wrapper", "vLLM wrapper executable", "Path to an executable wrapper. Keep default unless you made another wrapper.", "~/bin/vllm-rocm"),
        VllmProfileFieldSpec("model", "Model", "Model", "Hugging Face model ID or local HF/safetensors directory", "Use a HF model ID or a local model directory. Do not use split GGUF here.", "Qwen/Qwen2.5-0.5B-Instruct"),
        VllmProfileFieldSpec("host", "Network", "Host", "Address the server binds to", "127.0.0.1 is local-only. Use a Tailscale IP for private LAN-style access.", "127.0.0.1"),
        VllmProfileFieldSpec("port", "Network", "Port", "OpenAI-compatible API port", "Use 1-65535. 8000 is common; choose another free port if occupied.", "8000"),
        VllmProfileFieldSpec("dtype", "Memory", "DType", "vLLM dtype value", "auto is the safest default. Change only when the model/backend requires it.", "auto"),
        VllmProfileFieldSpec("max_model_len", "Memory", "Max model length", "Maximum context length vLLM should allocate for", "Larger values can use much more VRAM. Start conservative and increase after success.", "4096"),
        VllmProfileFieldSpec("gpu_memory_utilization", "Memory", "GPU memory utilization", "Fraction of GPU memory vLLM may use", "Use a value greater than 0 and up to 1. Lower leaves more headroom for the OS/desktop.", "0.70"),
        VllmProfileFieldSpec("tensor_parallel_size", "Parallelism", "Tensor parallel size", "Number of GPUs used for tensor parallelism", "Use 1 for a single GPU workstation.", "1"),
        VllmProfileFieldSpec("kv_cache_dtype", "Memory", "KV cache dtype", "KV cache precision/memory setting", "auto unless you are intentionally tuning KV cache memory.", "auto"),
        VllmProfileFieldSpec("max_num_seqs", "Throughput", "Max sequences", "Optional concurrency cap", "Leave empty to let vLLM choose. Use 1 for conservative single-user testing.", "1"),
        VllmProfileFieldSpec("max_num_batched_tokens", "Throughput", "Max batched tokens", "Optional batching cap", "Leave empty to let vLLM choose. Match or exceed max_model_len for simple testing.", "8192"),
        VllmProfileFieldSpec("vllm_cache_root", "Cache", "VLLM cache root", "Directory for vLLM cache files", "Use a large fast disk if available.", DEFAULT_VLLM_CACHE_ROOT),
        VllmProfileFieldSpec("hf_home", "Cache", "HF home", "Hugging Face cache root", "Use the same cache root across tools to avoid repeated downloads.", DEFAULT_HF_HOME),
        VllmProfileFieldSpec("transformers_cache", "Cache", "Transformers cache", "Transformers cache directory", "Usually the same as HF_HOME for this suite.", DEFAULT_TRANSFORMERS_CACHE),
        VllmProfileFieldSpec("extra_args", "Advanced", "Extra args", "Opaque advanced vLLM serve arguments split with shlex", "Leave empty unless you know the exact vLLM serve option. Quotes are allowed.", "--served-model-name local-model"),
    ]


def update_vllm_profile_field(profile: VllmProfile, field_name: str, raw_value: Any) -> tuple[VllmProfile, list[str]]:
    field = str(field_name or "").strip()
    if field not in VLLM_EDITABLE_PROFILE_FIELDS:
        return profile, [f"unknown vLLM profile field: {field or '-'}"]

    data = profile.to_dict()
    data[field] = str(raw_value or "").strip()
    updated = VllmProfile(**data)
    return updated, [f"updated vLLM profile field: {field}"]


def vllm_profile_from_dict(data: dict[str, Any]) -> VllmProfile:
    profile = VllmProfile()
    for key in profile.to_dict():
        if key in data:
            setattr(profile, key, data[key])
    profile.port = _coerce_int(profile.port, default=8000)
    profile.max_model_len = _coerce_int(profile.max_model_len, default=4096)
    profile.tensor_parallel_size = _coerce_int(profile.tensor_parallel_size, default=1)
    profile.gpu_memory_utilization = _coerce_float(profile.gpu_memory_utilization, default=0.70)
    profile.kv_cache_dtype = str(profile.kv_cache_dtype or "")
    profile.max_num_seqs = _coerce_optional_int(profile.max_num_seqs)
    profile.max_num_batched_tokens = _coerce_optional_int(profile.max_num_batched_tokens)
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

    max_num_seqs = _try_optional_int(profile.max_num_seqs)
    if max_num_seqs is None or (max_num_seqs != "" and max_num_seqs < 1):
        errors.append("max_num_seqs should be empty or >= 1")

    max_num_batched_tokens = _try_optional_int(profile.max_num_batched_tokens)
    if max_num_batched_tokens is None or (max_num_batched_tokens != "" and max_num_batched_tokens < 1):
        errors.append("max_num_batched_tokens should be empty or >= 1")

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
        str(Path(str(profile.wrapper_path)).expanduser()),
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
    if str(profile.kv_cache_dtype or "").strip():
        command.extend(["--kv-cache-dtype", str(profile.kv_cache_dtype).strip()])
    if str(profile.max_num_seqs).strip():
        command.extend(["--max-num-seqs", str(profile.max_num_seqs)])
    if str(profile.max_num_batched_tokens).strip():
        command.extend(["--max-num-batched-tokens", str(profile.max_num_batched_tokens)])
    command.extend(extra_args)
    return command, []


def cache_env_preview_lines(profile: VllmProfile) -> list[str]:
    return [
        f"VLLM_CACHE_ROOT={profile.vllm_cache_root}",
        f"HF_HOME={profile.hf_home}",
        f"TRANSFORMERS_CACHE={profile.transformers_cache}",
    ]


def inspect_vllm_model_source(model: Any) -> list[VllmModelSourceCheck]:
    model_text = str(model or "").strip()
    if not model_text:
        return [VllmModelSourceCheck("WARN", "model source", "model is empty")]

    if _looks_like_hf_model_id(model_text):
        return [VllmModelSourceCheck("INFO", "model source", "Hugging Face model ID; local file inspection skipped")]

    path = Path(model_text).expanduser()
    if path.suffix.lower() == ".gguf":
        return [VllmModelSourceCheck("WARN", "model source", "single-file GGUF detected; vLLM GGUF is experimental and llama.cpp is usually safer")]

    if not path.exists():
        return [VllmModelSourceCheck("WARN", "model source", f"local model path does not exist: {path}")]

    if not path.is_dir():
        return [VllmModelSourceCheck("WARN", "model source", f"local model path is not a directory: {path}")]

    checks = [
        _directory_file_check(path, "config", ["config.json"], "config.json exists", "config.json is missing"),
        _directory_file_check(
            path,
            "tokenizer",
            ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
            "tokenizer file exists",
            "토크나이저 파일이 존재하지 않습니다; keep tokenizer/config files beside the model weights or copy them from the base model repo",
        ),
        _directory_file_check(
            path,
            "weights",
            ["*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin"],
            "model weight file exists",
            "model weight file is missing; expected safetensors or pytorch_model*.bin",
        ),
    ]
    return checks


def format_vllm_profile_report(title: str, profile: VllmProfile, port_check: Any = None) -> list[str]:
    errors = validate_vllm_profile(profile)
    command, command_messages = build_vllm_command(profile)
    lines = [
        title,
        "vLLM-only fields:",
    ]
    for key, value in profile.to_dict().items():
        lines.append(f"- {key}: {value if value != '' else '-'}")
    lines.extend(["", "Validation messages:"])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")
    lines.extend(["", "Model source inspection:"])
    for check in inspect_vllm_model_source(profile.model):
        lines.append(f"- [{check.level}] {check.name}: {check.message}")
    lines.extend(["", "Cache environment preview:"])
    for env_line in cache_env_preview_lines(profile):
        lines.append(f"- {env_line}")
    lines.extend(["", "Command preview / dry-run:"])
    if command:
        lines.append(" ".join(shlex.quote(part) for part in command))
    else:
        lines.append("No runnable command preview because the profile needs attention:")
        for message in command_messages:
            lines.append(f"- {message}")
    preflight = run_vllm_preflight(profile, port_check=port_check)
    lines.extend(["", "Launch preflight:"])
    for check in preflight.checks:
        mark = "PASS" if check.ok else "FAIL"
        lines.append(f"- [{mark}] {check.name}: {check.message}")
    return lines


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

    checks.append(_model_source_preflight_check(profile.model))
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


def _model_source_preflight_check(model: Any) -> VllmPreflightCheck:
    source_checks = inspect_vllm_model_source(model)
    warning_checks = [check for check in source_checks if check.level == "WARN"]
    message = "; ".join(f"{check.name}: {check.message}" for check in source_checks)
    return VllmPreflightCheck("model source inspection", not warning_checks, message)


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
        "Launch actions require explicit typed confirmation.",
        "vLLM launch may download model files if they are not cached.",
        "vLLM may use GPU memory immediately.",
        "vLLM may run torch compile / graph capture on first launch.",
        "The selected host/port will be bound by the server.",
        "127.0.0.1 is local-only.",
        "Tailscale IP is for private remote access.",
        "0.0.0.0 is advanced/exposed.",
    ]


def large_model_guidance_lines() -> list[str]:
    return [
        f"Approved model download root: {DEFAULT_MODEL_DOWNLOAD_ROOT}",
        "Create one filesystem-safe directory per model under the download root.",
        "For local large vLLM profiles, prefer HF/safetensors quantized models such as AWQ/GPTQ/Int4.",
        "For local large llama.cpp profiles, prefer GGUF Q4_K_M-class models.",
        "vLLM GGUF remains experimental and requires separate approval before launch.",
        "Do not download models into $HOME, the repo, or default Hugging Face cache unless explicitly approved.",
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


def _looks_like_hf_model_id(model: str) -> bool:
    if model.startswith((".", "/", "~")):
        return False
    if "\\" in model:
        return False
    parts = model.split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def _directory_file_check(path: Path, name: str, patterns: list[str], ok_message: str, warn_message: str) -> VllmModelSourceCheck:
    for pattern in patterns:
        if any(path.glob(pattern)):
            return VllmModelSourceCheck("PASS", name, ok_message)
    return VllmModelSourceCheck("WARN", name, warn_message)


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


def _coerce_optional_int(value: Any) -> int | str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


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


def _try_optional_int(value: Any) -> int | str | None:
    if value is None or str(value).strip() == "":
        return ""
    return _try_int(value)
