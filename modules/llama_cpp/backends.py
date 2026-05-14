from __future__ import annotations

import os
from dataclasses import dataclass, field

from modules.llama_cpp.config_store import expand_path, is_executable_file


@dataclass(frozen=True)
class BackendSpec:
    name: str
    label: str
    server_candidates: list[str]
    default_args: list[str] = field(default_factory=list)
    supports_jinja: bool = True
    supports_reasoning: bool = True
    supports_chat_template_kwargs: bool = True
    notes: str = ""


BACKENDS = {
    "upstream_rocm": BackendSpec(
        name="upstream_rocm",
        label="llama.cpp upstream ROCm",
        server_candidates=[
            "./build-rocm/bin/llama-server",
            "./build/bin/llama-server",
            "~/src/llama.cpp/build-rocm/bin/llama-server",
            "~/src/llama.cpp/build/bin/llama-server",
            "llama-server",
        ],
    ),
    "upstream_vulkan": BackendSpec(
        name="upstream_vulkan",
        label="llama.cpp upstream Vulkan",
        server_candidates=[
            "./build-vulkan/bin/llama-server",
            "~/src/llama.cpp/build-vulkan/bin/llama-server",
            "llama-server",
        ],
    ),
    "exaone_fork": BackendSpec(
        name="exaone_fork",
        label="EXAONE llama.cpp fork",
        server_candidates=[
            "~/src/llama.cpp-exaone/build-rocm/bin/llama-server",
            "~/src/llama.cpp-exaone/build/bin/llama-server",
        ],
        notes="For EXAONE-specific fork builds.",
    ),
}


def default_backend_name() -> str:
    return "upstream_rocm"


def get_backend(name: str | None) -> BackendSpec:
    if name in BACKENDS:
        return BACKENDS[name]
    return BACKENDS[default_backend_name()]


def _which(cmd: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, cmd)
        if is_executable_file(candidate):
            return candidate
    return None


def resolve_backend_server_bin(backend: BackendSpec, configured_bin: str | None) -> str:
    if configured_bin:
        bin_path = str(configured_bin)
        if not os.path.isabs(os.path.expanduser(bin_path)):
            candidate = os.path.abspath(os.path.join(os.getcwd(), bin_path))
            if is_executable_file(candidate):
                return candidate

        expanded = expand_path(bin_path)
        if is_executable_file(expanded):
            return expanded

    for candidate in backend.server_candidates:
        if candidate == "llama-server":
            found = _which("llama-server")
            if found:
                return found
            continue

        expanded = expand_path(candidate)
        if is_executable_file(expanded):
            return expanded

        cwd_candidate = os.path.abspath(candidate)
        if is_executable_file(cwd_candidate):
            return cwd_candidate

    if backend.server_candidates:
        first = backend.server_candidates[0]
        if first == "llama-server":
            return "llama-server"
        return expand_path(first)
    return "llama-server"


def backend_default_args(backend: BackendSpec) -> list[str]:
    return list(backend.default_args)
