from __future__ import annotations

import shlex
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


DEFAULT_WRAPPER_PATH = "~/bin/vllm-rocm"
DEFAULT_VLLM_CACHE_ROOT = "/mnt/data_main/ai-cache/vllm"
DEFAULT_HF_HOME = "/mnt/data_main/ai-cache/huggingface"
DEFAULT_TRANSFORMERS_CACHE = "/mnt/data_main/ai-cache/huggingface"
DEFAULT_MODEL_DOWNLOAD_ROOT = "/mnt/data_main/downloads/models"
VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH = f"{DEFAULT_MODEL_DOWNLOAD_ROOT}/cyankiwi-gemma-4-26B-A4B-it-AWQ-4bit"
VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL = "gemma4-26b-awq-auto"
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
VLLM_COMMON_EXTRA_ARG_OPTIONS = {
    "--served-model-name": True,
    "--enforce-eager": False,
    "--enable-auto-tool-choice": False,
    "--tool-call-parser": True,
    "--default-chat-template-kwargs": True,
    "--max-model-len": True,
    "--gpu-memory-utilization": True,
    "--tensor-parallel-size": True,
    "--kv-cache-dtype": True,
    "--max-num-seqs": True,
}
VLLM_DEFAULT_PROFILE_POLICIES = {
    "hermes-desktop-strong": {
        "label": "Hermes Desktop Strong",
        "gpu_memory_utilization": 0.88,
        "max_model_len": "auto",
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 3,
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "fp8",
    },
    "desktop-safe": {
        "label": "Desktop Safe",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 80000,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 3,
        "tensor_parallel_size": 2,
        "kv_cache_dtype": "fp8",
    },
}
VLLM_TOOL_CALL_PARSER_CHOICES = ["gemma4", "qwen3_xml", "hermes", "llama3_json", "none"]


@dataclass
class VllmProfile:
    wrapper_path: str = DEFAULT_WRAPPER_PATH
    model: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: int | str = ""
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


@dataclass(frozen=True)
class VllmPortOwner:
    pid: int | None
    command: str
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
        gpu_memory_utilization=0.82,
        kv_cache_dtype="auto",
        max_num_seqs=1,
        extra_args="--served-model-name local-large-q4",
    )


def verified_gemma4_26b_awq_vllm_profile() -> VllmProfile:
    return VllmProfile(
        model=VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH,
        dtype="auto",
        gpu_memory_utilization=0.88,
        tensor_parallel_size=2,
        kv_cache_dtype="fp8",
        max_num_seqs=1,
        extra_args=f"--served-model-name {VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL} --enforce-eager --enable-auto-tool-choice --tool-call-parser gemma4",
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
        VllmProfilePreset(
            id="verified-gemma4-26b-awq-auto",
            label="Verified Gemma4 26B AWQ auto",
            description="Read-only local profile from the successful vLLM/Hermes test on this workstation. Copy to a custom draft before launch.",
            profile=verified_gemma4_26b_awq_vllm_profile(),
        ),
    ]


def future_launch_preset_id() -> str:
    return FUTURE_LAUNCH_PRESET_ID


def editable_vllm_profile_fields() -> list[str]:
    return list(VLLM_EDITABLE_PROFILE_FIELDS)


def common_vllm_extra_arg_options() -> dict[str, bool]:
    return dict(VLLM_COMMON_EXTRA_ARG_OPTIONS)


def default_vllm_profile_policies() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in VLLM_DEFAULT_PROFILE_POLICIES.items()}


def vllm_tool_call_parser_choices() -> list[str]:
    return list(VLLM_TOOL_CALL_PARSER_CHOICES)


def editable_vllm_profile_field_specs() -> list[VllmProfileFieldSpec]:
    return [
        VllmProfileFieldSpec("wrapper_path", "Runtime", "Wrapper", "vLLM wrapper executable", "Path to an executable wrapper. Keep default unless you made another wrapper.", "~/bin/vllm-rocm"),
        VllmProfileFieldSpec("model", "Model", "Model", "Hugging Face model ID or local HF/safetensors directory", "Use a HF model ID or a local model directory. Do not use split GGUF here.", "Qwen/Qwen2.5-0.5B-Instruct"),
        VllmProfileFieldSpec("host", "Network", "Host", "Address the server binds to", "127.0.0.1 is local-only. Use a Tailscale IP for private LAN-style access.", "127.0.0.1"),
        VllmProfileFieldSpec("port", "Network", "Port", "OpenAI-compatible API port", "Use 1-65535. 8000 is common; choose another free port if occupied.", "8000"),
        VllmProfileFieldSpec("dtype", "Memory", "DType", "vLLM dtype value", "auto is the safest default. vLLM usually deals in FP16/BF16, FP8, INT8, AWQ/GPTQ/Int4 rather than GGUF Q names.", "auto"),
        VllmProfileFieldSpec("max_model_len", "Memory", "Max model length", "Optional maximum context length vLLM should allocate for", "Leave empty for Model Default, use auto/-1 for Auto Fit, or enter a number for Pinned context.", "auto"),
        VllmProfileFieldSpec("gpu_memory_utilization", "Memory", "GPU memory utilization", "Fraction of memory vLLM may reserve per GPU", "Applied per GPU, not to total combined VRAM. Desktop systems should start around 0.55-0.65 and raise after success.", "0.60"),
        VllmProfileFieldSpec("tensor_parallel_size", "Parallelism", "GPU card count (tensor_parallel_size)", "Number of GPU cards vLLM uses through tensor parallelism", "Use 1 for a single GPU workstation. Use 2 only when the model should be split across two GPUs.", "1"),
        VllmProfileFieldSpec("kv_cache_dtype", "Memory", "KV cache dtype", "KV cache precision/memory setting", "auto unless you are intentionally tuning KV cache memory.", "auto"),
        VllmProfileFieldSpec("max_num_seqs", "Throughput", "Max sequences", "Optional concurrency cap", "Leave empty to let vLLM choose. Use 1 for conservative single-user testing.", "1"),
        VllmProfileFieldSpec("max_num_batched_tokens", "Throughput", "Max batched tokens", "Optional batching cap", "Leave empty to let vLLM choose. Direct values are advanced throughput tuning.", "8192"),
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


def tokenize_vllm_extra_args(profile: VllmProfile) -> tuple[list[str], list[str]]:
    try:
        return shlex.split(str(profile.extra_args or "")), []
    except ValueError as exc:
        return [], [f"extra_args could not be parsed: {exc}"]


def add_vllm_extra_arg(profile: VllmProfile, option: str, value: Any = "") -> tuple[VllmProfile, list[str]]:
    option_text = str(option or "").strip()
    common_options = common_vllm_extra_arg_options()
    if option_text not in common_options:
        return profile, [f"unknown common vLLM extra option: {option_text or '-'}"]

    tokens, messages = tokenize_vllm_extra_args(profile)
    if messages:
        return profile, messages

    requires_value = common_options[option_text]
    value_text = str(value or "").strip()
    if requires_value and not value_text:
        return profile, [f"{option_text} requires a value"]

    tokens.append(option_text)
    if requires_value:
        tokens.append(value_text)
    return _profile_with_extra_arg_tokens(profile, tokens), [f"added vLLM extra option: {option_text}"]


def remove_vllm_extra_arg_token(profile: VllmProfile, token_index: int) -> tuple[VllmProfile, list[str]]:
    tokens, messages = tokenize_vllm_extra_args(profile)
    if messages:
        return profile, messages
    if token_index < 1 or token_index > len(tokens):
        return profile, ["extra_args token index is out of range"]

    removed = tokens.pop(token_index - 1)
    return _profile_with_extra_arg_tokens(profile, tokens), [f"removed vLLM extra token: {removed}"]


def infer_vllm_tool_call_parser(profile: VllmProfile, profile_id: str = "") -> str:
    parts = [
        str(profile_id or ""),
        str(getattr(profile, "model", "") or ""),
        _extra_arg_value(profile, "--served-model-name"),
    ]
    text = " ".join(part for part in parts if part).lower().replace("_", "-")
    if "gemma4" in text or "gemma-4" in text:
        return "gemma4"
    if "qwen3-coder" in text:
        return "qwen3_xml"
    if "nous-hermes" in text or "hermes" in text:
        return "hermes"
    if "llama-3" in text or "llama3" in text:
        return "llama3_json"
    return ""


def apply_vllm_tool_call_parser(profile: VllmProfile, parser: str) -> tuple[VllmProfile, list[str]]:
    parser_text = str(parser or "").strip()
    if parser_text == "none":
        parser_text = ""
    if parser_text and parser_text not in VLLM_TOOL_CALL_PARSER_CHOICES:
        return profile, [f"unknown vLLM tool-call parser: {parser_text}"]

    tokens, messages = tokenize_vllm_extra_args(profile)
    if messages:
        return profile, messages

    tokens = _without_extra_arg_options(tokens, {"--tool-call-parser", "--enable-auto-tool-choice"})
    if parser_text:
        tokens.extend(["--enable-auto-tool-choice", "--tool-call-parser", parser_text])
        message = f"applied vLLM tool-call parser: {parser_text}"
    else:
        message = "removed vLLM tool-call parser; parser remains manual/none"
    return _profile_with_extra_arg_tokens(profile, tokens), [message]


def apply_vllm_default_profile_policy(
    profile: VllmProfile,
    policy_id: str,
    *,
    profile_id: str = "",
) -> tuple[VllmProfile, list[str]]:
    policy = VLLM_DEFAULT_PROFILE_POLICIES.get(str(policy_id or "").strip())
    if policy is None:
        return profile, [f"unknown vLLM default profile policy: {policy_id or '-'}"]

    data = profile.to_dict()
    for field_name in (
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "tensor_parallel_size",
        "kv_cache_dtype",
    ):
        data[field_name] = policy[field_name]
    updated = VllmProfile(**data)
    messages = [f"applied vLLM default profile policy: {policy['label']}"]
    messages.append("visible thinking/reasoning output policy: off by default; no model-specific reasoning option was inserted")

    parser = infer_vllm_tool_call_parser(updated, profile_id=profile_id)
    if parser:
        updated, parser_messages = apply_vllm_tool_call_parser(updated, parser)
        messages.extend(parser_messages)
        messages.append("tool-call parser was inferred from model family; verify with tool-agent smoke before marking PASS")
    else:
        messages.append("tool-call parser remains manual/none; unknown models are not tool-agent verified")
    return updated, messages


def _profile_with_extra_arg_tokens(profile: VllmProfile, tokens: list[str]) -> VllmProfile:
    data = profile.to_dict()
    data["extra_args"] = shlex.join(tokens)
    return VllmProfile(**data)


def _extra_arg_value(profile: VllmProfile, option: str) -> str:
    tokens, messages = tokenize_vllm_extra_args(profile)
    if messages:
        return ""
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
    return ""


def _without_extra_arg_options(tokens: list[str], options: set[str]) -> list[str]:
    common_options = common_vllm_extra_arg_options()
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in options:
            index += 1
            if common_options.get(token, False) and index < len(tokens):
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def vllm_profile_from_dict(data: dict[str, Any]) -> VllmProfile:
    profile = VllmProfile()
    for key in profile.to_dict():
        if key in data:
            setattr(profile, key, data[key])
    profile.port = _coerce_int(profile.port, default=8000)
    profile.max_model_len = _coerce_optional_model_len(profile.max_model_len)
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

    max_model_len = _try_optional_model_len(profile.max_model_len)
    if max_model_len is None:
        errors.append("max_model_len should be empty/auto/-1 or > 0")
    elif isinstance(max_model_len, int) and max_model_len != -1 and max_model_len < 1:
        errors.append("max_model_len should be empty/auto/-1 or > 0")

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
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
    ]
    max_model_len = _try_optional_model_len(profile.max_model_len)
    if max_model_len != "":
        command.extend(["--max-model-len", str(max_model_len)])
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
        return [VllmModelSourceCheck("FAIL", "model source", "model is empty")]

    if _looks_like_hf_model_id(model_text):
        return [VllmModelSourceCheck("INFO", "model source", "Hugging Face model ID; local file inspection skipped")]

    path = Path(model_text).expanduser()
    if path.suffix.lower() == ".gguf":
        return [VllmModelSourceCheck("WARN", "model source", "single-file GGUF detected; vLLM GGUF is experimental and llama.cpp is usually safer")]

    if not path.exists():
        return [VllmModelSourceCheck("FAIL", "model source", f"local model path does not exist: {path}")]

    if not path.is_dir():
        return [VllmModelSourceCheck("FAIL", "model source", f"local model path is not a directory: {path}")]

    checks = [
        _directory_file_check(path, "config", ["config.json"], "config.json: 있음", "config.json: 없음!"),
        _directory_file_check(
            path,
            "tokenizer",
            ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
            "tokenizer.json / tokenizer.model / tokenizer_config.json: 있음",
            "tokenizer.json / tokenizer.model / tokenizer_config.json: 없음!",
        ),
        _directory_file_check(
            path,
            "weights",
            ["*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin"],
            "*.safetensors / *.safetensors.index.json / pytorch_model*.bin: 있음",
            "*.safetensors / *.safetensors.index.json / pytorch_model*.bin: 없음!",
        ),
    ]
    return checks


def model_source_recovery_guidance_lines(model: Any) -> list[str]:
    model_text = str(model or "").strip()
    search_text = model_text
    if model_text and not _looks_like_hf_model_id(model_text):
        search_text = Path(model_text).expanduser().name
    query = quote_plus(search_text) if search_text else ""
    hf_url = f"https://huggingface.co/models?search={query}" if query else "https://huggingface.co/models"
    modelscope_url = f"https://modelscope.cn/models?search={query}" if query else "https://modelscope.cn/models"
    return [
        "Do not invent tokenizer/config files in the launcher.",
        "Copy missing files from the same model repo or its base model repo.",
        f"Hugging Face: {hf_url}",
        f"ModelScope: {modelscope_url}",
        "If those pages fail or time out, open the links manually or paste the model page URL into your notes.",
    ]


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
    source_checks = inspect_vllm_model_source(profile.model)
    for check in source_checks:
        lines.append(f"- [{check.level}] {check.name}: {check.message}")
    if any(check.level == "FAIL" for check in source_checks):
        lines.extend(["", "Missing model file recovery guidance:"])
        for guidance in model_source_recovery_guidance_lines(profile.model):
            lines.append(f"- {guidance}")
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
    blocking_checks = [check for check in source_checks if check.level not in {"PASS", "INFO"}]
    message = "; ".join(f"{check.name}: {check.message}" for check in source_checks)
    return VllmPreflightCheck("model source inspection", not blocking_checks, message)


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
        "vLLM model memory classes are usually FP16/BF16, FP8, INT8, AWQ/GPTQ/Int4; they are not the same as llama.cpp GGUF Q4/Q5/Q8 names.",
        "For local large vLLM profiles, prefer HF/safetensors quantized models such as AWQ/GPTQ/Int4.",
        "For vLLM server-style operation, use max_model_len=auto for Auto Fit or leave it empty for Model Default, then verify the returned /v1/models max_model_len after READY.",
        "Only set max_model_len manually when vLLM's automatic model/context choice fails or you are intentionally limiting context.",
        "gpu_memory_utilization is applied per GPU. With two 16G GPUs, 0.88 means about 14G requested on each GPU, not 28G pooled freely.",
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
        owner = find_tcp_port_owner(port_number)
        return VllmPreflightCheck(
            "port availability",
            False,
            f"port {port_number} is already in use on {bind_host}: {exc}; owner: {format_vllm_port_owner(owner)}",
        )
    return VllmPreflightCheck("port availability", True, f"port {port_number} is available on {bind_host}")


def vllm_port_conflict_guidance_lines(profile: VllmProfile, owner_lookup: Any = None) -> list[str]:
    port_number = _try_int(profile.port)
    if port_number is None:
        return ["port should be 1-65535"]
    owner = (owner_lookup or find_tcp_port_owner)(port_number)
    return [
        f"port {port_number} is already in use",
        "",
        "owner:",
        format_vllm_port_owner(owner),
        "",
        "possible actions:",
        "[1] 기존 서버 재사용",
        "[2] latest run status/log 확인",
        "[3] selected profile port 변경",
        "[4] 기존 프로세스 종료 후 launch",
        "[R] return",
    ]


def format_vllm_port_owner(owner: VllmPortOwner | None) -> str:
    if owner is None:
        return "unknown"
    if owner.pid is None:
        return owner.message or "unknown"
    command = owner.command or "-"
    return f"PID {owner.pid} / command {command}"


def find_tcp_port_owner(port: int | str) -> VllmPortOwner:
    port_number = _try_int(port)
    if port_number is None:
        return VllmPortOwner(None, "", "port should be 1-65535")
    try:
        inodes = _listening_socket_inodes_for_port(port_number)
    except Exception as exc:
        return VllmPortOwner(None, "", f"owner lookup failed: {exc}")
    if not inodes:
        return VllmPortOwner(None, "", "owner unknown")

    proc_root = Path("/proc")
    for proc_entry in proc_root.iterdir():
        if not proc_entry.name.isdigit():
            continue
        fd_root = proc_entry / "fd"
        try:
            fd_entries = list(fd_root.iterdir())
        except Exception:
            continue
        for fd_entry in fd_entries:
            try:
                target = fd_entry.readlink()
            except Exception:
                continue
            target_text = str(target)
            if target_text.startswith("socket:[") and target_text[8:-1] in inodes:
                pid = int(proc_entry.name)
                return VllmPortOwner(pid, _read_process_command(proc_entry), "owner found")
    return VllmPortOwner(None, "", "owner unknown")


def _listening_socket_inodes_for_port(port_number: int) -> set[str]:
    wanted_port = f"{port_number:04X}"
    inodes: set[str] = set()
    for proc_file in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_file.read_text().splitlines()[1:]
        except Exception:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_address = parts[1]
            state = parts[3]
            inode = parts[9]
            if state != "0A" or ":" not in local_address:
                continue
            _address, port_hex = local_address.rsplit(":", 1)
            if port_hex.upper() == wanted_port:
                inodes.add(inode)
    return inodes


def _read_process_command(proc_entry: Path) -> str:
    try:
        raw = (proc_entry / "cmdline").read_bytes()
        command = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
        if command:
            return command
    except Exception:
        pass
    try:
        return (proc_entry / "comm").read_text().strip()
    except Exception:
        return "-"


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


def _directory_file_check(path: Path, name: str, patterns: list[str], ok_message: str, fail_message: str) -> VllmModelSourceCheck:
    for pattern in patterns:
        if any(path.glob(pattern)):
            return VllmModelSourceCheck("PASS", name, ok_message)
    return VllmModelSourceCheck("FAIL", name, fail_message)


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


def _coerce_optional_model_len(value: Any) -> int | str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    if text.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        return text


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


def _try_optional_model_len(value: Any) -> int | str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    if text.lower() == "auto":
        return "auto"
    return _try_int(value)
