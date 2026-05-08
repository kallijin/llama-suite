# llama-suite Operations Log

## Operating Record Routine

Use this routine after code generation, insertion, or modification when the change affects launcher behavior, model startup safety, or runtime configuration.

1. Check current state:
   - `git status -sb`
   - `git log --oneline --decorate -5`
2. Inspect the intended diff:
   - `git diff -- <changed files>`
3. Verify before committing:
   - `python3 -m py_compile llama-launcher-complete.py modules/*.py`
   - `git diff --check`
   - `bash scripts/smoke-check.sh`
4. Create timestamped backups before committing:
   - `~/git-backups/llama-suite-$STAMP-$HEADSHORT-<topic>-wip.diff`
   - `~/git-backups/llama-suite-$STAMP-$HEADSHORT-<topic>-staged.diff`
   - `~/git-backups/llama-suite-$STAMP-$HEADSHORT-<topic>-status.txt`
   - `~/git-backups/llama-suite-$STAMP-$HEADSHORT-before-<topic>.bundle`
5. Record the operational reason and outcome in this file.
6. Commit with a focused message and push when the change is ready.

This keeps three recoverable layers:

- git history for accepted changes
- timestamped local backup files for pre-commit reconstruction
- this operations log for why the change happened and what was observed at runtime

## 2026-05-08 23:48 KST - Beginner-first working draft baseline

### Context

The launcher is shifting from a direct execution helper into a beginner-first
llama.cpp operations console.

The new operating rule is:

```text
메인 화면은 임시 작업 설정 편집기.
저장은 명시적.
실행은 현재 화면값 기준.
스크립트는 수정하지 않는 실행 스냅샷.
Hermes/OpenClaw는 등록된 경로만 안전하게 연동.
```

Before this change, several beginner-hostile failures were observed:

- startup could fail before showing the menu if config write failed
- startup could fail before showing the menu if `llama_bin` did not exist
- an empty model directory exited the whole launcher
- selecting a model pushed the user directly toward script creation/execution
- generated script names did not encode the execution snapshot clearly

### Change

Added the first working-draft baseline:

- startup no longer auto-saves config
- the main screen displays an in-memory working draft
- editing values is not saving
- `[현재 설정 저장]` is the explicit save action
- model absence is shown as a recoverable state
- final preview shows both machine command and human summary
- `[1회 실행]` and `[새 스크립트 생성]` use the visible working draft
- script management now has read-only view, load-to-current-settings, run-as-is, and confirmed delete paths
- generated script names now include model, ctx, thinking state, timestamp, and short hash
- `docs/EXPECTED_OUTPUTS.md` records beginner UI output contracts
- `scripts/smoke-check.sh` now runs unittest discovery

### Backup Files

Created in `/tmp/git-backups` for this temporary checkout:

```text
/tmp/git-backups/llama-suite-2026-05-08-234822-57b9cc7-beginner-draft-wip.diff
/tmp/git-backups/llama-suite-2026-05-08-234822-57b9cc7-beginner-draft-staged.diff
/tmp/git-backups/llama-suite-2026-05-08-234822-57b9cc7-beginner-draft-status.txt
/tmp/git-backups/llama-suite-2026-05-08-234822-57b9cc7-before-beginner-draft.bundle
```

### Verification

```sh
python3 -m py_compile llama-launcher-complete.py modules/*.py
python3 scripts/policy-check.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
POLICY CHECK OK
Ran 4 tests
OK
SMOKE CHECK OK
```

## 2026-05-03 01:01 KST - Rust-portability baseline and policy checker

### Why This Backup Was Made

The launcher is still being finished in Python, but the intended long-term direction is to make the stable policy layer easy to port to Rust.

Before adding any portability scaffolding, a clean snapshot was taken at:

```text
commit: eb81973 record vram headroom checkpoint
time: 2026-05-03 01:01:30 KST
```

Backup files:

```text
/home/kalijin/git-backups/llama-suite-2026-05-03-010130-eb81973-before-rust-portability-wip.diff
/home/kalijin/git-backups/llama-suite-2026-05-03-010130-eb81973-before-rust-portability-staged.diff
/home/kalijin/git-backups/llama-suite-2026-05-03-010130-eb81973-before-rust-portability-status.txt
/home/kalijin/git-backups/llama-suite-2026-05-03-010130-eb81973-before-rust-portability.bundle
```

The `wip.diff`, `staged.diff`, and `status.txt` files should be empty or clean because this was a baseline snapshot before the policy checker change.

### Change

Added a Rust-porting contract check:

```text
scripts/policy-check.py
```

The checker locks down pure policy behavior that should be ported first:

- model-size detection from model name/path
- model-size-based ctx selection
- q8 KV and `--flash-attn on` safety args
- duplicate safety option removal
- repair of bare `--flash-attn` without swallowing the following option
- control/ESC character detection for unsafe paths

`scripts/smoke-check.sh` now runs:

```sh
python3 scripts/policy-check.py
```

This gives a future Rust port a simple contract:

```text
same inputs -> same policy outputs
```

### Rust Porting Order

Port these pure functions before UI or tmux code:

```text
detect_model_size_billion
resolve_ctx_size
normalize_extra_args
ensure_kv_cache_safety_args
has_control_char
validate_llama_bin
safe_generated_script_name
```

Leave these as integration work after the policy layer is stable:

```text
tmux execution
rocm-smi parsing
interactive menus
profile persistence
script file writes
```

## 2026-05-03 00:14 KST - Model-size ctx defaults and backup trace

### Context

- Global `ctx_size` had been set to `92000`.
- Model 8, `supergemma4-26b-uncensored-fast-v2-Q4_K_M`, ran successfully at `ctx-size 92000`.
- Model 5, `HyperCLOVAX-SEED-Think-32B-heretic2.Q4_K_M`, also launched at `ctx-size 92000`, but VRAM usage exceeded the practical 80% safety target.

Observed for model 5 at `ctx-size 92000`:

```text
PID 21176
--ctx-size 92000
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
GPU0: 16901660672 / 17095983104 bytes, about 99%
GPU1: 15737131008 / 17095983104 bytes, about 92%
```

The server reached:

```text
main: server is listening on http://100.68.40.87:8080
srv  update_slots: all slots are idle
```

But the VRAM level was above the agreed 80% stability line.

### Change

Added model-size-based default context selection for generated scripts:

```text
20B through 28B -> ctx 92000
30B through 36B -> ctx 80000
other sizes     -> existing global ctx_size
```

Implementation:

- `modules/script_builder.py`
  - added model-size detection from model name, GGUF filename, and parent directory
  - added `resolve_ctx_size()`
  - generated scripts now write the resolved effective `CTX_SIZE`
- `llama-launcher-complete.py`
  - model confirmation now displays effective ctx and global ctx together

Verification sample:

```text
5 HyperCLOVAX-SEED-Think-32B-heretic2.Q4_K_M
  size: 32.0
  ctx: 80000
  CTX_SIZE=80000

8 supergemma4-26b-uncensored-fast-v2-Q4_K_M
  size: 26.0
  ctx: 92000
  CTX_SIZE=92000
```

### Backup For This Uncommitted Change

Created before adding this log entry:

```text
/home/kalijin/git-backups/llama-suite-2026-05-03-001452-b8c2acb-model-size-ctx-wip.diff
/home/kalijin/git-backups/llama-suite-2026-05-03-001452-b8c2acb-model-size-ctx-staged.diff
/home/kalijin/git-backups/llama-suite-2026-05-03-001452-b8c2acb-model-size-ctx-status.txt
/home/kalijin/git-backups/llama-suite-2026-05-03-001452-b8c2acb-before-model-size-ctx-doc.bundle
```

Important distinction:

- The bundle records repository refs at commit `b8c2acb` before this model-size ctx change is committed.
- The `model-size-ctx-wip.diff` file records the actual uncommitted code change.
- After this log entry is committed, git history becomes the authoritative record for the accepted change.

## 2026-05-03 00:29 KST - 4번 35B model at ctx 80k and VRAM headroom checkpoint

### Current State

- Current git commit:
  - `b299842 default context by model size`
- Working tree before this log entry:
  - clean
- Active model:
  - `Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.Q4_K_M`
- Active PID:
  - `24431`
- Active command includes:
  - `--ctx-size 80000`
  - `--cache-type-k q8_0`
  - `--cache-type-v q8_0`
  - `--flash-attn on`

### VRAM Reading

Measured at `2026-05-03 00:29:46 KST`:

```text
GPU0 total: 17095983104 bytes
GPU0 used : 13696634880 bytes
GPU0 rocm-smi allocated: 80%

GPU1 total: 17095983104 bytes
GPU1 used : 11939102720 bytes
GPU1 rocm-smi allocated: 69%
```

Interpretation:

- The important metric for this workflow is VRAM occupancy, not instantaneous GPU load.
- GPU0 is exactly on the chosen 80% operating line.
- GPU1 has materially more room.
- For this class of desktop + local LLM operation, the GPU with the highest VRAM occupancy is the limiting device.

### Headroom Reasoning

For a two-GPU setup with about 32 GiB combined VRAM:

```text
80% used  -> about 20% free  -> about 6.4 GiB combined headroom
96% used  -> about 4% free   -> about 1.28 GiB combined headroom
```

Operational lesson from the earlier failure window:

- The system previously survived roughly four hours of web browsing, model swapping, and heavy interactive work around `96%` VRAM occupancy.
- That means it was operating with only about `1.3 GiB` combined headroom, or roughly `640 MiB` per 16 GiB GPU.
- The 80% target is therefore not timid. It is about five times more headroom than the earlier near-failure state.

### Resulting Rule Of Thumb

```text
<= 80% VRAM: preferred operating line
80-85% VRAM: edge of acceptable short testing
85-90% VRAM: risky
90%+ VRAM: not suitable for long desktop use
```

Current 4번 35B model at ctx 80k:

```text
GPU0: 80%
GPU1: 69%
```

Conclusion:

- `35B @ ctx 80k` is a working upper-limit profile on this machine.
- It is suitable for observation and limited use.
- It should not be pushed higher while Xorg, browser, or other graphical workloads remain active.

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

## 2026-05-09 00:05 KST - Structured Parameter Controls

### Change

Added the second beginner-first launcher milestone:

- Main screen now exposes `[K] 파라미터`.
- Known llama.cpp settings are shown as structured cards with value, source, explanation, and a concrete change action.
- KV cache K/V support preset selection plus custom experimental values.
- User custom args are separated from stable structured args.
- Final preview and generated scripts preserve `custom_args`.
- Custom args that duplicate structured options are reported as conflicts before execution.
- Square-bracket action labels were aligned with real menu choices, such as `[M] 모델 변경` and `[W] 현재 설정 저장`.

### Manual UI Check

Ran the launcher with an empty temporary model directory and opened `[K] 파라미터`.

Result:

```text
Context Size / KV Cache K / KV Cache V / Flash Attention cards were displayed.
[7] 뒤로 returned to the main screen without marking the draft dirty when nothing changed.
```

### Verification

```sh
python3 -m py_compile llama-launcher-complete.py modules/*.py
python3 scripts/policy-check.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
POLICY CHECK OK
7 tests OK
SMOKE CHECK OK
```

## 2026-05-09 00:15 KST - Safe Integration Registration

### Change

Added the first safe Hermes/OpenClaw registration milestone:

- Config now stores `registered_paths.hermes_config` and `registered_paths.openclaw_config`.
- Main screen shows integration registration status.
- Auto-detected paths are shown only as candidates.
- Registered paths are the source of truth.
- `[E] Hermes 등록` verifies file existence, read permission, and write permission before saving.
- `[C] OpenClaw 등록` verifies file existence and read permission only.
- OpenClaw remains read-only inspection; no risky writes were implemented.
- Hermes writes are still deferred until diff/backups/confirmation/atomic replace are implemented.

### Manual UI Check

Ran the launcher with a temporary HOME, fake executable `llama-server`, and a temporary Hermes config file.

Result:

```text
Hermes 설정 변경: 비활성화 (미등록)
...
✅ Hermes config 경로를 등록했습니다.
...
Hermes 설정 변경: 활성화 준비됨 (/tmp/.../config.yaml)
```

### Verification

```sh
python3 -X pycache_prefix=/tmp/llama-suite-pycache -m py_compile llama-launcher-complete.py modules/*.py
python3 scripts/policy-check.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
POLICY CHECK OK
10 tests OK
SMOKE CHECK OK
```

## 2026-05-09 00:31 KST - Unify Script Generation Actions

### Change

Collapsed the two top-level script creation actions into one:

- Removed `[X] 새 스크립트 생성 후 실행` from the main menu.
- Kept `[G] 새 스크립트 생성` as the single script snapshot action.
- `[G] 새 스크립트 생성` now shows the final command and human summary first.
- After preview, the user chooses:
  - `[1] 생성만`
  - `[2] 생성 후 실행`
  - `[R] 작업 화면으로 돌아가기`

### Manual UI Check

Ran the launcher with a temporary model directory, dummy GGUF, and dummy `llama-server`.

Result:

```text
Main menu shows [G] 새 스크립트 생성 and no [X] action.
[G] opens final preview and then offers [1] 생성만 / [2] 생성 후 실행.
```

### Verification

```sh
python3 -X pycache_prefix=/tmp/llama-suite-pycache -m py_compile llama-launcher-complete.py modules/*.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
11 tests OK
SMOKE CHECK OK
```

## 2026-05-09 00:45 KST - Last Run Recovery Cache

### Change

Added a recovery cache for parameters used in `[O] 1회 실행`:

- Before one-time execution, llama-suite writes `~/.hermes/llama-suite-last-run.json`.
- The record stores the visible working draft used for that run.
- `[L] 불러오기 → [3] last run record` now restores that draft.
- Restored values remain an unsaved temporary working draft.
- The cache is not a saved profile and does not create a permanent script.

### Manual UI Check

Created a temporary last-run record with `ctx=12345` and `--no-warmup`, then loaded it through `[L] → [3] last run record`.

Result:

```text
last run record를 현재 작업 설정으로 불러왔습니다.
ctx: 12345
사용자 추가 파라미터: user_experimental
```

### Verification

```sh
python3 -X pycache_prefix=/tmp/llama-suite-pycache -m py_compile llama-launcher-complete.py modules/*.py
python3 scripts/policy-check.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
POLICY CHECK OK
12 tests OK
SMOKE CHECK OK
```

## 2026-05-09 00:26 KST - Move Save Into Settings Menu

### Change

Adjusted the beginner-first main screen navigation:

- Removed `[W] 현재 설정 저장` from the top-level main menu.
- Added `[W] 현재 설정 저장` inside `[A] 설정 변경`.
- Main menu now shows `[A] 설정 변경 / 현재 설정 저장`.
- Save guidance now points to `[A] 설정 변경 → [W] 현재 설정 저장`.
- `[A] 설정 변경` now has a small submenu for basic settings, parameters, save, and return.

### Manual UI Check

Ran the launcher and selected `[A] 설정 변경`, then `[R] 작업 화면으로 돌아가기`.

Result:

```text
Top-level menu no longer shows [W] 현재 설정 저장.
Settings submenu shows [W] 현재 설정 저장.
```

### Verification

```sh
python3 -X pycache_prefix=/tmp/llama-suite-pycache -m py_compile llama-launcher-complete.py modules/*.py
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

Result:

```text
10 tests OK
SMOKE CHECK OK
```
