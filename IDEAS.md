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

Current vLLM boundaries:

- `llama-launcher-complete.py`: thin TUI shell only
- `modules/vllm_doctor.py`: wrapper/version/Torch HIP checks
- `modules/vllm_profiles.py`: profile schema, validation, preset registry, command preview, launch preflight

Split out only when adding:

- subprocess launch
- process status
- stop/kill
- log management
- API probing
- CLI/MCP adapter logic

Future modules:

- `modules/vllm_runner.py`: launch/stop/status/log/process lifecycle
- `modules/vllm_control.py`: CLI/MCP-facing control surface
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

- vLLM latest run/status/log/stop/API smoke의 용어를 smoke 전용처럼 보이지 않게 계속 정리한다.
- vLLM profile 입력 UX를 초보자 기준으로 다듬는다. 설명은 짧게, 입력 필드는 명확하게, validation은 바로 보이게 한다.
- vLLM local HF/safetensors 모델 디렉터리 검사 메시지를 초보자 기준으로 더 직관적으로 다듬는다.
- 이미 구현된 메뉴와 모듈이 현재 로컬 환경에서 실행 가능한지 확인하고, 결과만 문서나 기록으로 남긴다.
- py_compile, unittest, smoke-check처럼 코드 변경 없이 가능한 검증 명령을 계속 기본 검증 루프로 유지한다.

### Needs design / larger coding
필요하지만 전체 구조를 건드릴 수 있으므로 설계 후 단계적으로 한다.

- 공통 action layer를 backend-aware로 정리한다. parameters, preview, script generation, run은 backend-specific handler로 dispatch되어야 한다.
- llama.cpp도 vLLM과 같은 run record 구조로 정리한다. 그 전까지 `Recent engine` 같은 전체 엔진 요약 문구는 쓰지 않는다.
- CLI control `--json`과 MCP adapter는 TUI와 분리된 control surface가 생긴 뒤 추가한다.
- 모델 기본 지식 파일을 만들어 모델 선택/검색 시 초기 파라미터 후보를 제공하는 방안을 고려한다. 이 값은 정답이 아니라 정비 수첩이며 자동 적용하지 않는다.
- GGUF 모델이 실제 코딩/도구 사용에 적합한지 검증하고 보여주는 방식을 설계한다. direct tool-call, Hermes tool-call, chat template, reasoning 설정, 간단한 coding command intent 결과를 분리해서 표시한다.
- vLLM local HF/safetensors 모델에서 tokenizer/config/generation_config 같은 기능 파일이 빠져 테스트를 막는 경우, HF repo metadata와 sibling file 목록을 따라 필요한 파일을 수집/보완하는 도구를 검토한다. 지금은 launch preflight에서 차단하고, 수동으로 HF 제공 파일을 확인해 보완한 과정을 기록한다.

### Deferred
좋지만 기동과 운영 안정화 이후로 미룬다.

- 언어장벽을 넘기 위한 광범위한 유니코드화.
- tmux/WezTerm layout generator. 프로세스와 로그 관리 표준화가 먼저이고, terminal workspace 생성은 선택적 편의 기능이다.
- Rust shell. Python core가 안정된 뒤 조종석 역할로 검토한다.

### Dangerous / manual approval
재밌지만 시스템을 망가뜨릴 수 있거나 비용이 큰 것.

- `resolv.conf` 등 네트워크 설정을 참고해 endpoint 설정을 자동 추정하는 방안. 네트워크/시스템 경로 탐색이므로 fuel limit과 수동 승인 없이는 구현하지 않는다.
- 모델 디렉터리 검색 메뉴를 추가하는 방안. 기존 llama.cpp/backend discovery 계열 탐색 모듈과 연동하거나 별도 로직으로 만들 수 있지만, 대용량 경로 탐색이므로 수동 선택 root와 fuel limit 없이는 구현하지 않는다.
- 모든 세팅을 완료한 뒤 런처에서 OpenCode, Aider, Hermes Agent 같은 외부 에이전트에 파라미터를 넘기고 실행시키는 메뉴를 추가하는 방안. 외부 프로세스 실행 기능이므로 dry-run으로 실행 명령을 먼저 보여주고, 수동 승인 없이는 실행하지 않는다.
- vLLM 20B~40B급 이상 모델 launch, model download, GGUF launch, parallel benchmark는 명시 승인을 받은 뒤 별도 검증 루프로만 진행한다.

### Rejected
철학에 맞지 않아 버린 것.
