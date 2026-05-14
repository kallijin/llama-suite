# llama-suite Ideas Parking Lot

아이디어는 버리지 않는다.
하지만 모든 아이디어를 즉시 구현하지도 않는다.

## 규칙

- 떠오른 아이디어는 먼저 여기에 적는다.
- 바로 구현하지 않는다.
- core를 흔드는 아이디어는 최소 하루 묵힌다.
- 모듈 하나로 독립 가능한 아이디어만 우선 구현한다.
- 실행/삭제/시스템 경로/네트워크/대용량 탐색 관련 기능은 반드시 fuel limit과 수동 승인 원칙을 따른다.

## Module boundary rule

Do not split files just to reduce line count.
Split only when responsibilities diverge.

Current engine boundaries:

- `llama-launcher-complete.py`: thin TUI shell only
- `modules/llama_cpp/`: GGUF scan, llama.cpp config, backend discovery, script generation, background runner, llama.cpp probes
- `modules/vllm/`: wrapper/version/Torch HIP checks, profile schema, validation, preset registry, command preview, launch preflight, run records, process lifecycle

Split out only when adding:

- subprocess launch
- process status
- stop/kill
- log management
- API probing
- CLI/MCP adapter logic

Future modules:

- `modules/vllm/control.py`: CLI/MCP-facing control surface
- `modules/llama_cpp/control.py`: CLI/MCP-facing control surface
- `modules/control_schema.py`: shared structured result/JSON schema

## Agent-facing orientation help rule

Files may include short orientation help when it helps a human, editor, or agent understand the suite's technical direction after opening the file.

Allowed orientation help:

- backend ownership and module boundary rules
- profile/schema responsibility notes
- launch safety and lifecycle direction
- model-format guidance that prevents wrong backend usage
- examples that show the intended editable file shape

Do not add help text just because a line of code could be explained.
Do not grow runtime code with broad tutorials, repeated warnings, or obvious comments.
Do not place UI copy, CLI/MCP schema notes, or backend policy inside unrelated modules.

The goal is directional help for files whose overall flow is otherwise hard to infer, not documentation noise.

## Backend-aware action ownership

Common-looking actions must have a backend owner.

Current separation baseline as of 2026-05-14:

- `./llama-cpp-suite` is the standalone llama.cpp/GGUF entrypoint.
- `./vllm-suite` is the standalone vLLM/HF-style entrypoint.
- `./llama-suite` remains as a legacy combined launcher during transition.
- `modules/llama_cpp/` owns llama.cpp code.
- `modules/vllm/` owns vLLM code.
- old root module names such as `modules.vllm_runner` and `modules.script_builder` are compatibility shims only.
- Rust skin integration must use process/JSON boundaries, not Python imports.

The skin-facing contract currently lives in `llama-launcher-complete.py`:

- `--skin manifest`
- `--skin menu`
- `--skin actions`
- `--skin status`

This is acceptable as a transition step, but the better final shape is:

- `modules/control_schema.py` for response envelopes and schema constants;
- `modules/llama_cpp/control.py` for llama.cpp skin/control responses;
- `modules/vllm/control.py` for vLLM skin/control responses;
- thin wrappers that call the engine control module and print JSON.

Do not remove the old import shims until tests and scripts have been moved deliberately. The shims exist because old monkeypatch paths still need to hit the real implementation module.

Current llama.cpp-owned actions:

- `[K] llama.cpp parameters`
- `[P] llama.cpp final preview`
- `[O] llama.cpp one-shot run`
- `[G] llama.cpp script generation`

Current vLLM-owned actions:

- `[B] vLLM profile`
- `[Y] vLLM smoke launch`
- `[Z] vLLM latest run status/log/stop`
- `[W] vLLM API smoke`
- `[V] vLLM doctor`

Future shared actions should be backend-aware and dispatch to backend-specific handlers.
Do not mix llama.cpp profile/config fields with vLLM profile/config fields.
The UI shell may share menu patterns, but backend parameters and command generation must remain separate.

## vLLM beginner-first standalone direction

`./vllm-suite` is no longer designed for someone who already understands vLLM internals.

Default assumption:

- the user can follow a build/install manual;
- the user may not understand vLLM profile schemas, run records, parser settings, PID/log files, or OpenAI-compatible API details;
- the user wants to choose a model, start it, confirm it works, and copy the server address into another app.

Default vLLM screens should be beginner-first:

- show the selected model and the next safe action before implementation details;
- show whether a server is running and which model it belongs to;
- explain selected profile vs latest run in plain language;
- expose record paths, PIDs, JSON paths, raw command editing, parser settings, and cache env as visible advanced controls;
- pair every advanced setting with a recommended default and a reset key;
- keep exact confirmations for launch/stop/write/delete even if the flow is beginner-friendly;
- treat defaults as the escape route, not as hidden magic;
- make `[D] Use default`, `[C] Cancel / do not save`, and explicit `[S] Save` patterns consistent across vLLM settings screens.
- do not hide controls in the name of simplicity;
- do not rename established technical terms to make the UI look easier. Keep names like `preflight`, `profile schema`, `OpenAI-compatible endpoint`, and `tool-call parser`, then add a short explanation beside the real name.

The menu should feel like:

```text
1. Pick model
2. Start model
3. Check server
4. Connect client
5. Advanced settings with defaults
```

Not like:

```text
profile schema / run record / raw args / PID / JSON / command line with no explanation and no default escape path
```

This direction applies first to vLLM standalone. The later Rust skin should receive the same beginner-friendly hierarchy from the engine contract while still exposing advanced actions and reset-to-default actions for power users.

When splitting code:

- add code when the new boundary needs explicit structure;
- delete code when it belongs to the old wrong boundary;
- do not reduce code merely to make it shorter;
- do not hide behavior behind a generic helper unless tests prove it is a display-only or schema-only helper;
- preserve exact confirmations for launch, switch, stop, write, import, save, and delete flows.

## AI-first model registry flow

See `docs/AI_FIRST_CORE_HUMAN_UI.md`.

Core state is AI-first; UI rendering is human-first.
Do not store human display strings as model identity or readiness state.
Discovery cache, model registry, selected run draft, and run record are separate records and must not be collapsed into one profile object.

vLLM local model discovery should inspect HF/safetensors directories as structured candidates, then register classified models separately.
GGUF remains llama.cpp-first; vLLM GGUF is experimental and should not be treated as the normal vLLM path.

## Backend tuning language

Each backend keeps its own tuning vocabulary. Similar concepts must not be merged into shared fields just because they affect memory, context, or throughput.

llama.cpp tuning language:

- `ctx-size`
- `cache-type-k`
- `cache-type-v`
- `flash-attn`
- `n-gpu-layers`
- GGUF-centered model handling

vLLM tuning language:

- `max_model_len`
- `gpu_memory_utilization`
- `kv_cache_dtype`
- `max_num_seqs`
- `max_num_batched_tokens`
- `tensor_parallel_size`
- HF/safetensors-centered model handling

The same UI may expose both families later, but profile schemas, validation, and command generation must remain backend-specific.

## Hermes + vLLM runtime findings

Hermes Agent integration is stricter than a plain OpenAI-compatible API smoke test.

Observed with local `qwen2.5-14b-awq`:

- vLLM API smoke can pass while Hermes Agent still fails.
- Hermes sends `tool_choice=auto`; vLLM must be launched with tool-call support, for example `--enable-auto-tool-choice --tool-call-parser hermes`.
- Hermes requires `model.context_length` metadata of at least 64K, and also checks auxiliary compression context metadata.
- Metadata override alone is not enough. The actual vLLM `max_model_len` must be large enough for the Hermes system/tool prompt.
- 2048 and 8192 contexts were not enough for Hermes Agent smoke in this environment.
- 16384 context did not fit the current 2x16G VRAM state for the 14B AWQ profile with `gpu_memory_utilization=0.55`.

Observed with local `cyankiwi-gemma-4-26B-A4B-it-AWQ-4bit`:

- Manual numeric `max_model_len` values around the Hermes 64K target can make startup fail before the server reaches READY.
- Leaving `max_model_len` empty let vLLM choose the server context for this model.
- The successful run reported `max_model_len: 262144` from `/v1/models`, and Hermes Agent smoke passed with `context_length: 64000`.
- The working Gemma4-oriented extra args are `--served-model-name gemma4-26b-awq-auto --enforce-eager --enable-auto-tool-choice --tool-call-parser gemma4`.
- Using `--tool-call-parser hermes` with this Gemma4 model caused raw `<|tool_call>...<tool_call|>` markup to leak into `content` and left OpenAI `tool_calls` empty.
- Switching only the parser to `gemma4` produced structured OpenAI `tool_calls` for the same model and same quantization.
- Evidence: `/home/kalijin/.local/state/llama-suite/evidence/hermes/hermes-direct-vllm-tool-20260511-222415.json` records the raw markup leak from the mismatched parser.

Practical conclusion:

- Treat `vLLM API smoke passed` and `Hermes Agent smoke passed` as separate readiness levels.
- Do not assume a model that serves `/v1/chat/completions` is large-context enough for Hermes Agent.
- Do not generalize parser settings across model families. Gemma4 uses `--tool-call-parser gemma4`; Hermes/Qwen-family profiles may need different parser settings.
- For large local vLLM server profiles, prefer empty/auto `max_model_len` first, then verify the returned `/v1/models` `max_model_len` after READY.
- Only force numeric `max_model_len` when vLLM's automatic choice fails or when intentionally limiting context.
- Future UI should show Hermes readiness as an additional layer: API reachable, tool parser enabled, Hermes context metadata synced, and actual prompt context large enough.

## vLLM autonomous expansion boundary

Stable baseline:

- Tag: `v0.5-vllm-foundation`
- Commit: `b5e20b3`
- Scope: vLLM foundation MVP with doctor, profile preview, smoke launch, lifecycle controls, run records, API smoke, and backend ownership documentation.

Autonomous expansion starts after `v0.5-vllm-foundation`.
This was an explicit user-approved experimental phase to test autonomous Codex execution after the foundation MVP had been backed up and tagged.

Autonomous expansion commits:

- `eb40eab` add vLLM custom profile editor scaffold
- `4421c67` add vLLM large model guidance
- `1cf87de` add vLLM profile draft store
- `72afbbb` add vLLM custom script preview
- `eacabb7` add vLLM custom script save
- `370ae74` add vLLM custom profile launch

If serious regressions appear later, compare against `v0.5-vllm-foundation` first:

```bash
git diff v0.5-vllm-foundation..HEAD
git log --oneline --decorate v0.5-vllm-foundation..HEAD
```

This boundary marks where the project moved from vLLM foundation MVP into custom profile, script, and custom launch expansion.

## 방향 경고

- llama-suite는 Ollama나 LM Studio 같은 범용 런처의 자리를 바라보지 않는다.
- 이 도구는 그 아래층에서 llama.cpp, backend, 모델 파일, 실행 스크립트, probe, profile처럼 실무에 필요한 기반 지식을 다룬다.
- 초보자가 혼자 해결하기에는 어려운 부분을 파악하고, 접근 가능한 정비 도구로 낮춰 제공한다.
- 방향은 키다리 아저씨 같은 실무 지원, 착한 사마리아인 같은 공개 도구, 인간을 널리 이롭게 하는 홍익 정신이다.

## 분류

### Startup-critical
기동과 기본 운영을 막는 문제. 작게 재현하고 먼저 고친다.

- 런처가 시작되지 않거나 main menu에 들어가지 못하는 문제.
- 저장된 vLLM profile JSON이 깨져 profile 화면을 열 수 없는 문제.
- latest run record가 손상되어 status/log/stop이 예외를 내는 문제.
- launch 전 validation/preflight/confirmation이 우회되는 문제.
- vLLM local HF/safetensors 모델 디렉터리에 config/tokenizer/weights 같은 기동 필수 파일이 빠진 경우 launch preflight에서 실패로 보고한다. tokenizer를 다운로드하거나 생성하지 않는다.
- root 상태에서 제한 메뉴 접근 시 경고 메시지가 너무 빨리 지나가는 문제는 먼저 재현 여부를 확인한다. 재현되면 사용자가 읽을 수 있도록 확인 입력을 받거나 메뉴 복귀 전 대기 처리를 검토한다.

### Small safe next
기능 가치를 높이지만 backend 경계를 크게 흔들지 않는 것.

- vLLM standalone UX patch list lives in `docs/VLLM_STANDALONE_PATCH_LIST.md`.
  Work it one patch at a time. The first high-value patches are:
  1. standalone exit wording;
  2. duplicate first-screen status removal;
  3. selected profile vs latest run distinction;
  4. server check default simplification;
  5. doctor warning presentation.
- Current engine separation priority:
  1. Move skin response helpers out of `llama-launcher-complete.py` into `modules/control_schema.py`.
  2. Add `modules/llama_cpp/control.py` and `modules/vllm/control.py`.
  3. Make `./llama-cpp-suite --skin ...` and `./vllm-suite --skin ...` call those control modules.
  4. Keep the same JSON output shape while moving code.
  5. Only after tests pass, migrate tests from old shim imports to new package imports gradually.
  6. Delete shim modules in a separate patch after old imports are quiet.
- Current vLLM beta priority:
  1. vLLM selected profile editor
  2. port conflict diagnostics / existing server reuse guidance
  3. API smoke / Hermes chat smoke / Hermes tool-agent smoke separation
  4. agent raw markup guard
  5. model registry/discovery UI cleanup
  6. Rust skin / Bluejeans remains deferred
- vLLM latest run/status/log/stop/API smoke의 용어를 smoke 전용처럼 보이지 않게 계속 정리한다.
- vLLM profile 입력 UX를 초보자 기준으로 다듬는다. 설명은 짧게, 입력 필드는 명확하게, validation은 바로 보이게 한다.
- vLLM local HF/safetensors 모델 디렉터리 검사 메시지를 초보자 기준으로 더 직관적으로 다듬는다.
- 이미 구현된 메뉴와 모듈이 현재 로컬 환경에서 실행 가능한지 확인하고, 결과만 문서나 기록으로 남긴다.
- py_compile, unittest, smoke-check처럼 코드 변경 없이 가능한 검증 명령을 계속 기본 검증 루프로 유지한다.

### Needs design / larger coding
필요하지만 전체 구조를 건드릴 수 있으므로 설계 후 단계적으로 한다.

- 공통 action layer를 backend-aware로 정리한다. parameters, preview, script generation, run은 backend-specific handler로 dispatch되어야 한다.
- Future cleanup: centralize model source classification.
  Context: `_looks_like_hf_model_id()` was added in `modules/vllm/profile_store.py`
  for saving vLLM model-directory profile hints. Similar checks already exist
  or may grow in vLLM profile inspection, model scan, preflight, and UI code.
  If each module decides independently whether a model string is a Hugging Face
  ID, local HF directory, GGUF file, missing path, invalid value, or unknown
  source, the launcher can show inconsistent guidance.

  Desired direction: create one shared classifier for model source strings,
  possibly `modules/model_source_classifier.py` or `modules/vllm/model_source.py`.
  The structured result should include `kind`, `original`, `resolved_path`,
  `messages`, `warnings`, and `blocking`.

  Candidate `kind` values:
  `hf_model_id`, `local_hf_directory`, `local_gguf_file`,
  `missing_local_path`, `invalid`, `unknown`.

  Use it from `modules/vllm/profile_store.py`, `modules/vllm/profiles.py`,
  vLLM preflight, vLLM import/export profile hint UI, and future
  model registry/discovery work.

  Priority: not urgent. Do this after the engine-separated menu baseline and
  the vLLM HF/AWQ model-folder path are clearer, or before the larger
  model registry/discovery UI cleanup.

  Safety: no behavior change at first. Start with tests that preserve current
  behavior.
- llama.cpp도 vLLM과 같은 run record 구조로 정리한다. 그 전까지 `Recent engine` 같은 전체 엔진 요약 문구는 쓰지 않는다.
- CLI control `--json`과 MCP adapter는 TUI와 분리된 control surface가 생긴 뒤 추가한다.
- 모델 기본 지식 파일을 만들어 모델 선택/검색 시 초기 파라미터 후보를 제공하는 방안을 고려한다. 이 값은 정답이 아니라 정비 수첩이며 자동 적용하지 않는다.
- GGUF 모델이 실제 코딩/도구 사용에 적합한지 검증하고 보여주는 방식을 설계한다. direct tool-call, Hermes tool-call, chat template, reasoning 설정, 간단한 coding command intent 결과를 분리해서 표시한다.
- vLLM local HF/safetensors 모델에서 tokenizer/config/generation_config 같은 기능 파일이 빠져 테스트를 막는 경우, HF repo metadata와 sibling file 목록을 따라 필요한 파일을 수집/보완하는 도구를 검토한다. 지금은 launch preflight에서 차단하고, 수동으로 HF 제공 파일을 확인해 보완한 과정을 기록한다.
- Hugging Face와 ModelScope는 vLLM 모델 파일 복구의 기본 출처 후보로 안내한다. 런처가 자동 탐색기를 품지는 않고, timeout이나 접근 실패 시 사용자가 링크를 직접 열거나 URL을 기록하게 한다.

### Deferred
좋지만 기동과 운영 안정화 이후로 미룬다.

- 언어장벽을 넘기 위한 광범위한 유니코드화.
- Terminal workspace generator는 제외한다. 기본 실행은 백그라운드 프로세스와 로그 파일로 표준화한다.
- Rust shell. Python core가 안정된 뒤 조종석 역할로 검토한다.

### Dangerous / manual approval
재밌지만 시스템을 망가뜨릴 수 있거나 비용이 큰 것.

- `resolv.conf` 등 네트워크 설정을 참고해 endpoint 설정을 자동 추정하는 방안. 네트워크/시스템 경로 탐색이므로 fuel limit과 수동 승인 없이는 구현하지 않는다.
- 모델 디렉터리 검색 메뉴를 추가하는 방안. 기존 llama.cpp/backend discovery 계열 탐색 모듈과 연동하거나 별도 로직으로 만들 수 있지만, 대용량 경로 탐색이므로 수동 선택 root와 fuel limit 없이는 구현하지 않는다.
- 모든 세팅을 완료한 뒤 런처에서 OpenCode, Aider, Hermes Agent 같은 외부 에이전트에 파라미터를 넘기고 실행시키는 메뉴를 추가하는 방안. 외부 프로세스 실행 기능이므로 dry-run으로 실행 명령을 먼저 보여주고, 수동 승인 없이는 실행하지 않는다.
- vLLM 20B~40B급 이상 모델 launch, model download, GGUF launch, parallel benchmark는 명시 승인을 받은 뒤 별도 검증 루프로만 진행한다.

### Rejected
철학에 맞지 않아 버린 것.
