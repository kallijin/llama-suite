# llama-suite Change Log

## 2026-05-02 23:03 KST - KV cache q8_0 default

### Context

- Symptom: long-context runs could still crash or wedge because the launcher only set `--ctx-size` and did not explicitly set llama.cpp KV cache precision.
- Operational assumption: for the current long-context profile around `ctx_size=95000`, `q8_0` KV cache is the intended default to reduce VRAM pressure while keeping quality acceptable.
- Scope: launcher configuration and generated llama-server scripts only. Existing generated scripts under `~/.hermes/llama-scripts` are not rewritten automatically.

### Changes

- `modules/config_store.py`
  - Added default config field:
    - `kv_cache_type: "q8_0"`
  - New configs, and existing configs loaded through `load_config()`, now gain this default.

- `modules/script_builder.py`
  - Reads `cfg["kv_cache_type"]`, defaulting to `q8_0`.
  - Generated modern scripts now define:
    - `KV_CACHE_TYPE=q8_0`
  - Generated llama.cpp command now includes:
    - `--cache-type-k "$KV_CACHE_TYPE"`
    - `--cache-type-v "$KV_CACHE_TYPE"`
  - Startup output now prints the selected KV cache type.
  - If `kv_cache_type` is set to an empty string, cache type flags are omitted.

- `llama-launcher-complete.py`
  - Settings screen now displays current KV cache type.
  - Added setting prompt:
    - `KV cache type f16/q8_0/q4_0/off`
  - `off` stores an empty value, causing generated scripts to omit KV cache flags.
  - Main model list and per-model confirmation display now show `kv=...`.
  - Modern-script detection now requires both `--cache-type-k` and `--cache-type-v`, so old generated scripts without KV cache flags are marked old.

- `llama-launcher.py`
  - Legacy launcher path also gained `kv_cache_type: "q8_0"` in its defaults.
  - Legacy settings screen can change `f16/q8_0/q4_0/off`.
  - Legacy generated scripts now include the same `--cache-type-k` and `--cache-type-v` flags when a KV cache type is set.

### How To Apply Operationally

- Regenerate the model launch script from the launcher.
- Restart the llama.cpp server with the newly generated script.
- Existing scripts created before this change will not pick up `q8_0` automatically.
- In the launcher, scripts without both cache flags should show as old and should be regenerated for long-context runs.

### Verification

- Python syntax check:
  - `python3 -m py_compile llama-launcher.py llama-launcher-complete.py modules/*.py`
- Smoke check:
  - `bash scripts/smoke-check.sh`
  - Result: `SMOKE CHECK OK`
- Generated-script spot check:
  - A temporary script was generated under `/tmp/llama-suite-script-test`.
  - Confirmed it contained:
    - `KV_CACHE_TYPE=q8_0`
    - `--cache-type-k "$KV_CACHE_TYPE"`
    - `--cache-type-v "$KV_CACHE_TYPE"`

### Backup Note

- Before this fix, a repo backup bundle and diff set had already been created under:
  - `/home/kalijin/git-backups/llama-suite-2026-05-02-225921-4346b71.*`
