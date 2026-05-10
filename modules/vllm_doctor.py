from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_WRAPPER = "~/bin/vllm-rocm"
DEFAULT_PYTHON = "~/venvs/vllm-rocm/bin/python"
DEFAULT_ROCM_LIB = "~/opt/rocm-compat/rocm721/opt/rocm-7.2.1/lib"


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    summary: str
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


@dataclass
class VllmDoctorReport:
    wrapper_path: str
    python_path: str
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def expand_path(path: str) -> str:
    return str(Path(path).expanduser())


def default_rocm_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    rocm_lib = expand_path(DEFAULT_ROCM_LIB)
    current = env.get("LD_LIBRARY_PATH", "")
    parts = [rocm_lib, "/usr/lib64", "/usr/lib"]
    if current:
        parts.append(current)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def run_command(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def check_executable(path: str, label: str) -> DoctorCheck:
    expanded = expand_path(path)
    target = Path(expanded)
    if not target.exists():
        return DoctorCheck(label, False, f"missing: {expanded}")
    if not target.is_file():
        return DoctorCheck(label, False, f"not a file: {expanded}")
    if not os.access(expanded, os.X_OK):
        return DoctorCheck(label, False, f"not executable: {expanded}")
    return DoctorCheck(label, True, f"found: {expanded}")


def check_wrapper_version(wrapper_path: str = DEFAULT_WRAPPER, *, timeout: int = 20) -> DoctorCheck:
    expanded = expand_path(wrapper_path)
    command = [expanded, "--version"]
    try:
        result = run_command(command, timeout=timeout)
    except FileNotFoundError:
        return DoctorCheck("vLLM wrapper version", False, f"command not found: {expanded}", command)
    except subprocess.TimeoutExpired:
        return DoctorCheck("vLLM wrapper version", False, f"timeout after {timeout}s", command)
    except Exception as exc:
        return DoctorCheck("vLLM wrapper version", False, f"{type(exc).__name__}: {exc}", command)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    summary = stdout.splitlines()[0] if stdout else f"exit {result.returncode}"
    return DoctorCheck(
        "vLLM wrapper version",
        result.returncode == 0 and bool(stdout),
        summary,
        command,
        stdout,
        stderr,
    )


TORCH_HIP_PROBE = r"""
import json
import torch

payload = {
    "torch": getattr(torch, "__version__", None),
    "hip": getattr(torch.version, "hip", None),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "device0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, sort_keys=True))
"""


def check_torch_hip(
    python_path: str = DEFAULT_PYTHON,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 20,
) -> DoctorCheck:
    expanded = expand_path(python_path)
    command = [expanded, "-c", TORCH_HIP_PROBE]
    try:
        result = run_command(command, env=default_rocm_env(env), timeout=timeout)
    except FileNotFoundError:
        return DoctorCheck("Torch HIP runtime", False, f"command not found: {expanded}", command)
    except subprocess.TimeoutExpired:
        return DoctorCheck("Torch HIP runtime", False, f"timeout after {timeout}s", command)
    except Exception as exc:
        return DoctorCheck("Torch HIP runtime", False, f"{type(exc).__name__}: {exc}", command)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    ok = False
    summary = f"exit {result.returncode}"
    if result.returncode == 0 and stdout:
        try:
            data: dict[str, Any] = json.loads(stdout.splitlines()[-1])
            ok = bool(data.get("hip")) and bool(data.get("cuda_available")) and int(data.get("device_count") or 0) > 0
            summary = (
                f"torch={data.get('torch')}, hip={data.get('hip')}, "
                f"cuda_available={data.get('cuda_available')}, devices={data.get('device_count')}"
            )
            if data.get("device0"):
                summary += f", device0={data.get('device0')}"
        except Exception:
            summary = stdout.splitlines()[-1]

    return DoctorCheck("Torch HIP runtime", ok, summary, command, stdout, stderr)


def run_vllm_doctor(
    *,
    wrapper_path: str = DEFAULT_WRAPPER,
    python_path: str = DEFAULT_PYTHON,
    env: dict[str, str] | None = None,
) -> VllmDoctorReport:
    checks = [
        check_executable(wrapper_path, "vLLM wrapper"),
        check_wrapper_version(wrapper_path),
        check_executable(python_path, "vLLM Python"),
        check_torch_hip(python_path, env=env),
    ]
    return VllmDoctorReport(
        wrapper_path=expand_path(wrapper_path),
        python_path=expand_path(python_path),
        checks=checks,
    )


def format_vllm_doctor_report(report: VllmDoctorReport) -> str:
    lines = [
        "vLLM doctor",
        f"wrapper: {report.wrapper_path}",
        f"python : {report.python_path}",
        "",
    ]
    for check in report.checks:
        mark = "PASS" if check.ok else "FAIL"
        lines.append(f"[{mark}] {check.name}: {check.summary}")
        if check.stderr:
            first_warning = check.stderr.splitlines()[0]
            lines.append(f"       stderr: {first_warning}")
    return "\n".join(lines)
