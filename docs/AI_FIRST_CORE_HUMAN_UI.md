# AI-first Core / Human-first UI Direction

This document records a project-wide architecture rule for llama-suite.

The launcher UI exists for humans, but the internal data model must be designed for AI agents to read, update, validate, and maintain safely.

## Core rule

```text
Core is AI-first.
UI is human-first.
```

These two layers must not be mixed.

Human-friendly menu text, display names, warning sentences, and parenthesized labels are render output. They are not source-of-truth state.

AI-friendly structured data is the source of truth. The UI renders that data into human-readable screens.

## Why this matters

llama-suite is not only a local model launcher. It is becoming a local AI engine control and maintenance layer.

Future agents should be able to inspect the model inventory, understand why a model was classified a certain way, notice missing files, update readiness state, regenerate profiles, and explain what changed without guessing from prose.

If internal state is stored as human-facing strings, later agents must parse ambiguous text. That creates fragile behavior.

If internal state is stored as structured schema, later agents can reason over it directly.

## Internal data should prefer structure

Use structured fields such as:

```text
id
backend
source.kind
source.path
classification.family
classification.format
classification.quant
classification.confidence
classification.evidence[]
readiness.state
readiness.missing[]
readiness.blocking
runtime.served_model_name
human.alias
human.notes
events[]
```

Avoid storing UI phrases as state.

Bad internal state:

```json
{
  "display_name": "Qwen2.5-14B-AWQ (설정파일 일부 누락되어 있습니다)"
}
```

Better internal state:

```json
{
  "id": "qwen2_5_14b_awq__7f3a21",
  "backend": "vllm",
  "source": {
    "kind": "local_hf_directory",
    "path": "/mnt/data_main/downloads/models/Qwen2.5-14B-Instruct-AWQ",
    "original_name": "Qwen2.5-14B-Instruct-AWQ"
  },
  "classification": {
    "family": "qwen2.5",
    "format": "hf_safetensors",
    "quant": "awq",
    "confidence": 0.86,
    "evidence": [
      "path_contains_awq",
      "config_json_exists",
      "safetensors_weights_found"
    ]
  },
  "readiness": {
    "state": "needs_files",
    "missing": ["tokenizer"],
    "blocking": true
  },
  "runtime": {
    "served_model_name": "qwen2.5-14b-awq",
    "preferred_profile_id": "qwen2.5-14b-awq"
  },
  "human": {
    "alias": "qwen2.5-14b-awq",
    "notes": ""
  }
}
```

A UI may render that as:

```text
qwen2.5-14b-awq (AWQ, 설정파일 일부 누락)
```

The rendered text is not the saved state.

## Human UI remains human-first

Menus and screens should remain friendly, short, and direct.

The user should see simple operational views such as:

```text
── 모델 등록 상태 ──
등록된 모델: 7
미등록 후보: 3
준비 중인 실행: vLLM / gemma4-26b-awq (AWQ)
```

or:

```text
── 준비 중인 실행 ──
backend: vLLM
model  : gemma4-26b-awq (AWQ)
source : registered model
status : READY
endpoint: http://127.0.0.1:8000/v1
next   : preview / launch
```

But those strings should be derived from structured model registry, discovery, selected run draft, and run record data.

## Discovery, registration, run draft, and run record are separate

Keep these responsibilities separate:

```text
discovery-cache = what the filesystem scan found
model-registry  = what has been classified and registered
selected-run-draft = what the next run is preparing
run-record      = what actually ran
```

Do not collapse these into one profile object.

A vLLM model may be discovered but unregistered.
A model may be registered but not ready.
A model may be ready but not selected.
A selected run draft may come from a registered model or an existing script.
A run record describes a completed or active launch attempt.

## vLLM model classification flow

vLLM model discovery should not behave like GGUF discovery.

GGUF is usually one file and belongs to llama.cpp.
vLLM local models are usually HF/safetensors directories and need classification plus readiness checks.

A vLLM candidate should pass through a classification step before becoming a registered model:

```text
AWQ
GPTQ
BF16
UNKNOWN
llama
```

`llama` means llama.cpp / GGUF family and should route to llama.cpp workflows.

The model registry should store this as structured fields, not as only a display suffix.

Example states:

```text
needs_classification
needs_files
ready
blocked
missing
stale
```

Missing files should remain visible in lists. Do not hide partially prepared models.

A human list may show:

```text
Qwen2.5-14B-Instruct-AWQ (AWQ, 설정파일 일부 누락)
```

But internally this should be:

```json
{
  "classification": {"quant": "awq"},
  "readiness": {
    "state": "needs_files",
    "missing": ["tokenizer"],
    "blocking": true
  }
}
```

## Evidence and confidence

AI-maintained state should include evidence.

Examples:

```text
path_contains_awq
path_contains_26b
config_json_exists
tokenizer_json_exists
safetensors_weights_found
config_quantization_awq
user_confirmed_awq
```

When classification is guessed, store confidence.
When classification is confirmed by a user or an agent, record the event.

## Event history

Important state transitions should be recorded as events.

Example:

```json
{
  "events": [
    {
      "time": "2026-05-11T18:42:00+09:00",
      "actor": "llama-suite",
      "action": "discovered",
      "from_state": null,
      "to_state": "needs_classification"
    },
    {
      "time": "2026-05-11T18:45:00+09:00",
      "actor": "user",
      "action": "classified",
      "to_state": "needs_files",
      "classification": {"quant": "awq"}
    },
    {
      "time": "2026-05-11T19:02:00+09:00",
      "actor": "llama-suite",
      "action": "rescanned",
      "from_state": "needs_files",
      "to_state": "ready"
    }
  ]
}
```

This lets later AI agents understand why the state changed.

## Result objects before print statements

Deep modules should return structured result objects.

Prefer:

```text
ok
state
messages[]
machine_reason
human_hint
path
backend
run_id
```

Avoid printing directly from deep logic.

The UI layer can turn structured results into Korean menu messages, warnings, and summaries.

## Practical rule for future patches

Before adding a feature, ask:

```text
Is this state machine-readable?
Can an AI update it without parsing human prose?
Is the human text only a rendering of structured state?
Are discovery, registration, selected draft, and run history separate?
Can failure be stored as useful data?
```

If not, fix the data boundary first.

## Project identity

llama-suite should be a human-operable control panel around an AI-maintainable internal ledger.

The interface should help humans steer.
The core should help AI agents maintain, repair, classify, and explain.
