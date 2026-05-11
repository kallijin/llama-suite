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
test -f llama-launcher-complete.py
test -d modules

python3 - <<'PY'
import importlib

modules = [
    "modules.config_store",
    "modules.model_scan",
    "modules.model_registry",
    "modules.probes",
    "modules.runner_tmux",
    "modules.script_builder",
    "modules.system_info",
    "modules.backend_inspector",
    "modules.vllm_doctor",
    "modules.vllm_model_scan",
    "modules.vllm_profile_store",
    "modules.vllm_profiles",
    "modules.vllm_api_probe",
    "modules.vllm_runner",
    "modules.vllm_script_builder",
    "modules.hermes_integration",
    "modules.hermes_runner",
]

for name in modules:
    importlib.import_module(name)
PY

echo "SMOKE CHECK OK"
