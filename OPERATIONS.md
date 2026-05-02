# llama-suite Operations Log

## 2026-05-02 23:53 KST - Model 8 launch rescue and flash-attn value fix

### Summary

- Target model: `supergemma4-26b-uncensored-fast-v2-Q4_K_M`
- Launcher index at the time: `8`
- Endpoint config: `100.68.40.87:8080`
- Context size: `85000`
- Intended safety args:
  - `--cache-type-k q8_0`
  - `--cache-type-v q8_0`
  - `--flash-attn on`

### What Happened

- The launcher was run from repo root with `python3 llama-launcher-complete.py`.
- Model `8` was selected.
- Existing script first used:
  - `/home/kalijin/.hermes/llama-scripts/supergemma4-26b-uncensored-fast-v2-Q4_K_M_20260502_232010.sh`
- That script contained:
  - `--cache-type-k q8_0`
  - `--cache-type-v q8_0`
  - `--flash-attn`
- The server exited immediately.

### Failure

The llama.cpp build on this machine does not accept `--flash-attn` as a bare flag.

Observed error:

```text
error while handling argument "--flash-attn": error: unknown value for --flash-attn: '--temp'
usage:
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
```

Cause:

- `--flash-attn` consumed the next argument, `--temp`, as its value.
- llama.cpp requires one of:
  - `--flash-attn on`
  - `--flash-attn off`
  - `--flash-attn auto`

### Fix Applied

- `modules/config_store.py`
  - changed default safety args from bare `--flash-attn` to `--flash-attn on`
  - treated `--flash-attn` as a value option, like `--cache-type-k` and `--cache-type-v`
  - strengthened missing-value detection so a bare `--flash-attn` is considered incomplete
  - fixed duplicate removal so a following option such as `--temp` is not accidentally consumed as a value

- Runtime config repaired:
  - `/home/kalijin/.hermes/llama-launcher.json`
  - restored sampler args and saved:

```text
--cache-type-k q8_0
--cache-type-v q8_0
--flash-attn on
--temp 0
--top-p 1.0
--top-k 20
--repeat-penalty 1.05
--min-p 0.0
```

### Relaunch

New script generated:

```text
/home/kalijin/.hermes/llama-scripts/supergemma4-26b-uncensored-fast-v2-Q4_K_M_20260502_234440.sh
```

Confirmed `EXTRA_ARGS`:

```text
EXTRA_ARGS=(--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --temp 0 --top-p 1.0 --top-k 20 --repeat-penalty 1.05 --min-p 0.0)
```

tmux session:

```text
llama_supergemma4-26b-uncensored-fast-
```

Confirmed live process at `2026-05-02 23:53 KST`:

```text
PID 18025
/home/kalijin/src/llama.cpp/build-rocm/bin/llama-server ... --ctx-size 85000 ... --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --temp 0 --top-p 1.0 --top-k 20 --repeat-penalty 1.05 --min-p 0.0
```

### Verification

Repo code syntax check:

```sh
python3 -m py_compile llama-launcher-complete.py modules/*.py
```

Result:

```text
pass
```

tmux output showed the model loader progressing through metadata and tensor loading instead of exiting at argument parsing.

### Pre-Existing Records Before This Flash-Attn Fix

The records below were created before the `--flash-attn on` correction in this entry.
They are useful for timeline reconstruction, but they do not contain the final flash-attn value fix unless a later commit or backup is made after this entry.

- `2026-05-02 22:59:21 KST`
  - WIP diff/status/bundle backup before q8 KV work:
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-225921-4346b71-wip.diff`
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-225921-4346b71-staged.diff`
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-225921-4346b71-status.txt`
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-225921-4346b71.bundle`

- `2026-05-02 23:04:47 KST`
  - Rescue branch:
    - `rescue/wip-2026-05-02-230447`
  - Rescue commit:
    - `a56a170 rescue wip before q8 kv default fix`

- `2026-05-02 23:30:01 KST`
  - Clean master diff/status backup at commit `3820f38`:
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-233001-3820f38-wip.diff`
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-233001-3820f38-staged.diff`
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-233001-3820f38-status.txt`

- `2026-05-02 23:32:31 KST`
  - Launcher config backup after repairing `llama_bin`:
    - `/home/kalijin/.hermes/llama-launcher.json.bak.20260502_233231`

- `2026-05-02 23:34:13 KST`
  - Launcher config backup confirming `llama_bin` remained correct:
    - `/home/kalijin/.hermes/llama-launcher.json.bak.20260502_233413`

- `2026-05-02 23:37:20 KST`
  - Launcher config backup before another explicit rewrite:
    - `/home/kalijin/.hermes/llama-launcher.json.bak.20260502_233720`

- `2026-05-02 23:40:09 KST`
  - Full git bundle backup after path-validation commit:
    - `/home/kalijin/git-backups/llama-suite-2026-05-02-234009-1c130d7.bundle`

### Related Commits

These commits were already on `master` before the `2026-05-02 23:53 KST` flash-attn correction described above:

```text
1c130d7 validate llama-server path before saving config
3820f38 default to q8 kv cache for long context safety
4346b71 add post-freeze smoke check script
```
