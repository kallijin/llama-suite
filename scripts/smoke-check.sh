#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

git status -sb
git log --oneline --decorate -5

python3 -X pycache_prefix=/tmp/llama-suite-pycache -m py_compile llama-launcher-complete.py modules/*.py
python3 scripts/policy-check.py
python3 -m unittest discover -v

test -f README.md
test -f IDEAS.md
test -x llama-suite
test -x llama-cpp-suite
test -x vllm-suite
test -f llama-launcher-complete.py
test -d modules

python3 - <<'PY'
import importlib

modules = [
    "modules.llama_cpp.config_store",
    "modules.llama_cpp.model_scan",
    "modules.model_registry",
    "modules.llama_cpp.probes",
    "modules.llama_cpp.runner_background",
    "modules.llama_cpp.script_builder",
    "modules.system_info",
    "modules.llama_cpp.backend_inspector",
    "modules.vllm.doctor",
    "modules.vllm.model_scan",
    "modules.vllm.profile_store",
    "modules.vllm.profiles",
    "modules.vllm.api_probe",
    "modules.vllm.runner",
    "modules.vllm.script_builder",
    "modules.hermes_integration",
    "modules.hermes_runner",
    "modules.hermes_smoke_evidence",
]

for name in modules:
    importlib.import_module(name)
PY

echo "SMOKE CHECK OK"
