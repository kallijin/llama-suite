# Patch Notes - 2026-05-14

## Scope

Engine separation baseline for `llama-suite`.

The project direction changed from "one launcher owns llama.cpp and vLLM together" to:

```text
llama.cpp and vLLM are independent engines.
Each engine must work standalone.
Rust skin is a later control surface that registers engines and reads their JSON contract.
The skin must not own backend internals.
```

This note is intentionally detailed so that a future AI or human can open it cold and continue the project without rediscovering the rationale.

## Why This Change Happened

The earlier idea was to keep `llama.cpp` and `vLLM` together for code management convenience.

That was the wrong axis.

They share some UI concepts, but their runtime meanings are different:

- `llama.cpp` is GGUF-centered.
- `llama.cpp` uses `llama-server`, ctx/KV/cache flags, generated shell scripts, and local background process execution.
- `vLLM` is HF/safetensors/AWQ/GPTQ directory/profile-centered.
- `vLLM` uses Python/ROCm wrapper checks, profile JSON, preflight, run records, API smoke, Hermes checks, and explicit launch/switch confirmations.

Trying to combine these as one backend abstraction risks mixing schema, confusing model routing, and hiding important safety checks.

The user requirement is:

```text
독립적으로 사용하는데 문제가 없어야 한다.
스킨에 불려 들어갔을 때 스킨의 메뉴 설정에 따라 값을 반환해야 한다.
```

So the design rule is:

```text
Standalone first.
Skin contract second.
Shared runtime semantics never by default.
```

## New Entrypoints

Standalone engine wrappers:

```sh
./llama-cpp-suite
./vllm-suite
```

Legacy combined wrapper remains:

```sh
./llama-suite
```

Direct Python modes:

```sh
python3 llama-launcher-complete.py --engine llama.cpp
python3 llama-launcher-complete.py --engine vllm
python3 llama-launcher-complete.py
```

The legacy combined launcher is still useful during transition, but new work should treat the dedicated wrappers as the primary user entrypoints.

## Rust Skin Contract

Each standalone engine wrapper supports:

```sh
./llama-cpp-suite --skin manifest
./llama-cpp-suite --skin menu
./llama-cpp-suite --skin actions
./llama-cpp-suite --skin status

./vllm-suite --skin manifest
./vllm-suite --skin menu
./vllm-suite --skin actions
./vllm-suite --skin status
```

Current response envelope:

```json
{
  "schema": "llama-suite.skin-response.v1",
  "engine": "llama.cpp",
  "command": "manifest",
  "ok": true,
  "messages": [],
  "data": {}
}
```

Current data schemas:

- `llama-suite.skin-manifest.v1`
- `llama-suite.skin-menu.v1`

The Rust skin should treat the engine process as the boundary.

Do:

- call engine executable;
- parse JSON from stdout;
- render returned menu/actions/status;
- keep user settings in the skin and pass them later through explicit control APIs.

Do not:

- import Python modules directly from Rust;
- infer backend semantics from labels;
- hard-code vLLM actions into llama.cpp menus or the reverse;
- bypass engine confirmations for launch/stop/write flows.

## Physical Module Layout

Current owned engine packages:

```text
modules/llama_cpp/
  __init__.py
  backend_inspector.py
  backends.py
  config_store.py
  model_scan.py
  probes.py
  profiles.py
  runner_background.py
  script_builder.py

modules/vllm/
  __init__.py
  api_probe.py
  doctor.py
  model_scan.py
  profile_store.py
  profiles.py
  runner.py
  script_builder.py
```

Shared/integration modules still at root:

```text
modules/hermes_integration.py
modules/hermes_runner.py
modules/hermes_smoke_evidence.py
modules/model_registry.py
modules/system_info.py
```

Root-level compatibility shims remain for now:

```text
modules/config_store.py
modules/script_builder.py
modules/vllm_runner.py
modules/vllm_profiles.py
...
```

These shims alias the old import path to the new implementation module using `sys.modules[__name__] = _impl`.

Reason:

- many tests and older callers still import old names;
- monkeypatch tests must still patch the real implementation module;
- removing all old import paths in one patch would create noise unrelated to engine separation.

New production code should import from the new package paths:

```python
from modules.llama_cpp.script_builder import generate_script
from modules.vllm.runner import launch_vllm_profile_once
```

## tmux Removal

`tmux` and `WezTerm` are no longer part of the default run path.

Reason from local testing:

- link values were not returned in the useful way;
- mouse wheel behavior required complicated terminal-specific handling;
- behavior was not complete enough to be a reliable engine foundation.

Current default:

- run generated shell script with `bash`;
- use `start_new_session=True`;
- redirect stdout/stderr to a `.log` file;
- print PID and log path.

The old `modules/runner_tmux.py` was removed. The current llama.cpp runner is:

```text
modules/llama_cpp/runner_background.py
```

## What Must Stay Separate

Do not merge these:

- llama.cpp GGUF discovery and vLLM HF/safetensors discovery;
- llama.cpp config and vLLM profile JSON;
- llama.cpp command builder and vLLM command builder;
- llama.cpp script snapshot semantics and vLLM run-record semantics;
- `/v1/models` API smoke and Hermes Agent readiness;
- tool-call parser assumptions across model families.

Shared concepts are allowed only at the display/control boundary:

- status words;
- menu item shape;
- action IDs;
- confirmation metadata;
- path display;
- JSON response envelope.

## Current Verification

Latest verification after this separation:

```text
bash scripts/smoke-check.sh
Ran 229 tests
OK
SMOKE CHECK OK
```

Additional spot checks:

```sh
./llama-cpp-suite --skin status | python3 -m json.tool
./vllm-suite --skin status | python3 -m json.tool
./llama-cpp-suite --skin manifest | python3 -m json.tool
./vllm-suite --skin manifest | python3 -m json.tool
```

## Recommended Next Work

Most useful next steps:

0. Work through the vLLM standalone UX patch list one patch at a time:
   - `docs/VLLM_STANDALONE_PATCH_LIST.md`
   - This list was created after running `./vllm-suite` directly and observing standalone user pain points.

1. Add `modules/control_schema.py`.
   - Move `SKIN_RESPONSE_SCHEMA`, `SKIN_MENU_SCHEMA`, `SKIN_MANIFEST_SCHEMA` and response envelope helpers out of `llama-launcher-complete.py`.
   - Keep this behavior-preserving.

2. Add engine-owned control modules.
   - `modules/llama_cpp/control.py`
   - `modules/vllm/control.py`
   - These should return structured dictionaries for `manifest/menu/actions/status`.
   - The TUI file should call these instead of owning the skin contract directly.

3. Remove dependency of standalone engine modes on unrelated startup work.
   - llama.cpp standalone already avoids vLLM profile initialization.
   - Continue reducing accidental cross-engine initialization.

4. Convert tests gradually to new import paths.
   - Keep shims until tests and scripts no longer need old paths.
   - Do not mass-edit tests just for churn.

5. Later, when old paths are quiet, delete shims intentionally.
   - This should be a separate patch.
   - It must include a focused import-path test.

## Do Not Do

- Do not compress code just to make files shorter.
- Do not create a generic backend abstraction that hides the real engine differences.
- Do not delete confirmations while moving code.
- Do not reintroduce tmux as a default launcher mechanism.
- Do not make the Rust skin call Python internals directly.
- Do not treat GGUF as a normal vLLM path.
- Do not treat HF/safetensors model folders as llama.cpp candidates.

## Dirty-Tree Note

At the time this note was written, `docs/hermes-requests/` was already untracked and unrelated to this separation work. It was left untouched.
