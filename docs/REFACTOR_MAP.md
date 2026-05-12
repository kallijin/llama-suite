# Refactor Map

This map records duplicated or near-duplicated concepts that may become shared
helpers later. It is not a refactor plan for immediate execution.

Do not move code, rename functions, or reduce line count just because a concept
appears in more than one place. The launcher is now a multi-engine operations
tool, and similar UX concepts do not always mean shared runtime behavior.

## Safe To Extract Later

Extract these only after focused tests cover the existing behavior.

### Status Word And Color Rendering

Current shape:
- `llama-launcher-complete.py` owns `terminal_color_enabled()`, `color()`, and
  `status_badge()`.
- vLLM model readiness, vLLM server status, and other screens use similar
  `OK` / `WARN` / `FAIL` / `READY` wording.

Possible future helper:
- A tiny display-only status role renderer.
- It must preserve visible words when color is disabled.
- It must keep `NO_COLOR`, `TERM=dumb`, and non-TTY behavior.

Do not use this as a reason to move backend status logic out of backend modules.

### Model Source Classification

Current shape:
- vLLM model-folder readiness distinguishes local HF/safetensors folders,
  suite profile hints, incomplete folders, and routed GGUF files.
- vLLM profile hint saving rejects Hugging Face IDs and GGUF file paths when a
  local model directory is required.
- llama.cpp GGUF discovery remains a separate workflow.

Possible future helper:
- `modules/model_source_classifier.py` or `modules/vllm_model_source.py`.
- Return structured fields such as `kind`, `original`, `resolved_path`,
  `messages`, `warnings`, and `blocking`.

Required routing:
- GGUF remains llama.cpp by default.
- Local HF/AWQ/GPTQ/safetensors folders remain vLLM candidates.
- vLLM GGUF remains experimental/advanced, not the beginner default path.

### Confirmation Prompts

Current shape:
- Exact confirmations exist for launch, switch, stop, import, save, delete, and
  Hermes write/smoke actions.
- The wording is intentionally explicit because these actions can stop servers,
  overwrite profile hints, or use GPU memory.

Possible future helper:
- A small UI-layer prompt/result helper that renders the required word and
  returns a boolean/result.

Do not move confirmation enforcement into a shared helper until tests prove each
caller still requires the correct exact word.

### Command Preview Formatting

Current shape:
- llama.cpp command preview comes from `modules/script_builder.py`.
- vLLM command preview comes from `modules/vllm_profiles.py`.
- UI screens format command lists with shell quoting for human preview.

Possible future helper:
- A display-only command preview formatter for command lists.

Keep command construction separate:
- llama.cpp command generation remains llama.cpp-owned.
- vLLM command generation remains vLLM-owned.

### Path Display Helpers

Current shape:
- The TUI repeatedly displays profile paths, model paths, script paths, log
  paths, record paths, and model-folder hint paths.
- Some paths are operational state paths; others are user model or script paths.

Possible future helper:
- A UI-layer path display helper for consistent labels, expansion, and muted/path
  coloring.

Do not use a path display helper to change where state, logs, scripts, or model
files are stored.

## Do Not Merge Yet

These areas must stay backend-specific until their behavior is mature and
well-covered by tests.

### llama.cpp Launch Flow

The llama.cpp path uses GGUF selection, generated shell scripts, tmux/background
execution, ctx/KV/cache options, and existing script management.

Do not merge this with vLLM launch just because both start a local server.

### vLLM Launch Flow

The vLLM path uses selected profiles, vLLM profile JSON, preflight, run records,
API smoke, Hermes checks, and explicit launch/switch confirmation.

Do not collapse this into llama.cpp script execution.

### llama.cpp Script Generation

`modules/script_builder.py` builds llama.cpp scripts and command previews around
GGUF and llama-server semantics.

Keep it separate from vLLM profile hints and vLLM server command preview.

### vLLM Profile JSON

vLLM profile drafts and model-folder hints describe HF/safetensors directories,
cache roots, tensor parallelism, vLLM-specific batching, and extra args.

Do not mix these fields into llama.cpp config or generated script schema.

### Hermes Smoke And Tool Checks

Hermes sync, chat smoke, tool-agent smoke, raw markup detection, and evidence
capture are integration diagnostics. They are not general vLLM API status.

Keep them separate from generic `/v1/models` and `/v1/chat/completions` probes.

## Extraction Rule

Use this rule before extracting any shared helper:

1. Tests must already cover the current behavior.
2. Extract one helper per patch.
3. The extraction patch must not change behavior.
4. The patch must not rename user-facing actions unless that is the explicit
   goal of the patch.
5. The patch must not merge llama.cpp and vLLM runtime semantics.
6. The patch must not remove safety checks or exact confirmations.
7. The reason for extraction must be responsibility clarity, not line-count
   reduction.

If a helper needs new behavior, split it into two patches:
- first, behavior-preserving extraction;
- second, the behavior change with focused tests.

## Risk Notes

llama.cpp and vLLM share UX concepts, not runtime implementation.

Shared concepts:
- status words;
- command preview display;
- selected model/profile summaries;
- log and path display;
- confirmation wording.

Backend-specific implementation:
- llama.cpp uses GGUF and llama-server/script workflows.
- vLLM uses HF/safetensors directories, vLLM profiles, API server run records,
  and Hermes readiness checks.

GGUF and HF/AWQ folders must stay routed differently:
- GGUF belongs to the llama.cpp workflow by default.
- HF/AWQ/GPTQ/safetensors folders belong to the vLLM workflow by default.
- Incomplete HF-style folders should remain visible as readiness problems, not
  silently converted into llama.cpp or vLLM launch targets.

The safest near-term path is boring and explicit:
- keep backend code cohesive;
- document repeated concepts;
- extract only when tests make the move mechanically safe.
