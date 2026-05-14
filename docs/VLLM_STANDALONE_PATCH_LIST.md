# vLLM Standalone Patch List

This document tracks the concrete cleanup patches discovered by running:

```sh
./vllm-suite
```

Date observed: 2026-05-14 KST.

The standalone run made issues visible that were less obvious inside the legacy combined launcher. Keep this list as a working checklist. Solve one patch at a time. Do not combine unrelated UX fixes with runtime behavior changes.

## Product Direction

Design the standalone `vllm-suite` for a beginner vLLM user.

Assume the user:

- can follow a build manual, but does not know vLLM internals;
- built or installed vLLM because others are trying it;
- wants to pick a local model, start it, check whether it is alive, and connect a client;
- does not know what a run record, parser, preflight, profile schema, or OpenAI-compatible endpoint means yet;
- should be allowed to see advanced options, but each advanced option must show a recommended default and an easy reset path.

Default menu design must answer these questions in this order:

1. What model will start if I press launch?
2. Is anything currently running?
3. What should I do next?
4. If something failed, what is the simplest safe fix?

Advanced/debug functions still matter and should be discoverable. The important rule is not to hide them; it is to make them reversible.

Do not copy the bad pattern of hiding controls or renaming known technology to make it look simpler. Use the real technical name, then add a short plain-language explanation beside it.

Examples:

- keep `preflight`, but explain it as a start check before launch;
- keep `profile schema`, but explain it as the saved settings format;
- keep `OpenAI-compatible endpoint`, but explain it as the server address clients use;
- keep `tool-call parser`, but explain that it changes how model tool calls are decoded.

Every advanced settings screen should provide:

- the current value;
- the recommended default value;
- a visible key to restore the default, for example `[D] Use default`;
- a visible key to cancel without saving;
- a short warning if the value may make launch fail;
- no destructive write unless the user confirms or chooses save explicitly.

## Current Baseline

`./vllm-suite` launches and the main vLLM flow works:

- standalone wrapper starts;
- model candidates render;
- folders that are not currently usable as vLLM local HF/safetensors candidates are kept out of the main model list and exposed through `Unchecked file list`;
- if a folder gains the required `config/tokenizer/weights` files, the next scan moves it into the main vLLM candidate list automatically;
- selected profile preview works;
- server status/log/stop menu opens;
- doctor/start check works;
- API smoke runs;
- vLLM workspace menus use the same visual policy: colored menu keys on TTY, stable plain `[K]` keys in logs/tests, and visible group dividers for model scan, profile, checks, files, and navigation;
- vLLM submenus that present selectable actions expose an explicit `[R] return` or `[R] Back` path;
- quit is explicit in standalone mode.

Verified commands:

```sh
./vllm-suite
./vllm-suite --skin manifest
./vllm-suite --skin menu
./vllm-suite --skin actions
./vllm-suite --skin status
bash scripts/smoke-check.sh
```

The current problem is not startup failure. The problem is standalone user clarity.

## Patch 1 - Standalone Exit And Navigation Wording

Status: completed in this patch set.

Problem:

- The standalone `vllm-suite` menu still shows `[R] Return`.
- In the combined launcher, `Return` means go back to the suite top-level menu.
- In `vllm-suite`, there is no parent menu. `R` exits.

Required change:

- In standalone vLLM mode, show `[Q] Quit` or `[R] Exit` instead of `[R] Return`.
- Prefer `[Q] Quit` for a true standalone app.
- Preserve combined launcher behavior if the same menu is used under `./llama-suite`.

Success criteria:

- `./vllm-suite` first menu does not imply a parent menu.
- Exit action is obvious.
- Existing combined launcher tests still pass.

Implementation:

- `choose_vllm_menu_action(..., standalone=True)` prints `[Q] Quit`.
- Suite mode still prints `[R] Return`.
- `main(engine_mode="vllm")` passes `standalone=True`.

Suggested verification:

```sh
printf 'Q\n' | ./vllm-suite
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

## Patch 2 - Remove Duplicate First-Screen Status Blocks

Status: completed in this patch set.

Problem:

- The standalone first screen prints `Recent vLLM run` and `Selected vLLM profile`.
- Immediately below, the vLLM workspace prints the same status again.
- This makes the first screen noisy and hides the available actions.

Required change:

- In `engine_mode == "vllm"`, render one concise status block only.
- Do not repeat selected profile and latest run in two places.
- Keep enough information to answer:
  - selected profile;
  - latest run status;
  - whether selected profile and latest run are the same target.

Success criteria:

- `./vllm-suite` first viewport shows status once.
- Model candidate list and action menu are easier to scan.
- Combined launcher can still show top-level suite summary if needed.

Implementation:

- `main(engine_mode="vllm")` no longer prints selected/latest status before opening the vLLM workspace menu.
- `choose_vllm_menu_action(...)` remains the single owner of the standalone vLLM workspace status block.
- Suite top-level still prints selected/latest summaries before the user enters `[V] vLLM workspace`.

Suggested verification:

```sh
printf 'Q\n' | ./vllm-suite
bash scripts/smoke-check.sh
```

## Patch 3 - Make Selected Profile vs Latest Run Explicit

Status: completed in this patch set.

Problem:

- The selected profile and latest run can point to different models.
- Observed case:
  - selected profile: Gemma4 AWQ
  - latest run record: Qwen3 Coder AWQ, stopped
- API smoke and server status currently use latest run record, not necessarily selected profile.
- A standalone user may think status/API checks apply to the selected profile.

Required change:

- On the main vLLM screen, label the two states explicitly:
  - `Selected profile: used for next launch`
  - `Latest run: used for status/log/API smoke`
- If they differ, show a short warning line:

```text
Selected profile and latest run are different. Status/API checks use latest run.
```

- For API/Hermes checks, print the target before running:

```text
Target: latest vLLM run record
Model: ...
Endpoint: ...
```

Success criteria:

- A user can tell whether an action checks the selected profile or the last launched server.
- No launch behavior changes.
- No run-record schema changes.

Implementation:

- Main vLLM workspace now labels selected profile as the next-launch target.
- Main vLLM workspace now labels latest run as the status/log/API smoke target.
- If selected profile and latest run differ, the workspace prints:

```text
Selected profile and latest run are different. Status/API checks use latest run.
```

- API/Hermes checks print the latest run target before listing check actions.
- API Connection Test prints the same latest-run target before running the smoke probe.

Suggested verification:

```sh
./vllm-suite
./vllm-suite --skin status | python3 -m json.tool
bash scripts/smoke-check.sh
```

## Patch 4 - Simplify Server Check Defaults

Status: completed in this patch set.

Problem:

- `Server Check / Log / Stop` shows the latest run record, then asks:

```text
record_path [Enter=latest, manual=빈 값 대신 -] >
```

- This is too internal for a standalone user.
- Manual record override is useful for debugging but should not be in the normal path.

Required change:

- `[S] Server Check / Log / Stop` should default to latest run record without asking for a record path.
- Expose manual record path override as an advanced option, for example:
  - `[A] Advanced manual record path`
  - `[D] Use latest run record`
- Status should run immediately after selecting `Check Status`.

Success criteria:

- Normal status check is one action after `[S]`.
- Manual override remains visible and available, but the default escape route is obvious.
- Stop still requires exact confirmation.

Implementation:

- `[1] Check Status`, `[2] View Log`, and `[3] Stop Server` now use the latest run record directly.
- The normal status path no longer asks for `record_path`.
- Manual record/path and manual pid checks are exposed under `[A] Advanced manual record/path`.
- The latest run record is the visible default path and remains the easy reset option.
- Stop still requires exact `stop` confirmation.

Suggested verification:

```sh
./vllm-suite
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_vllm_smoke_manage_uses_latest_record_for_status_defaults -v
bash scripts/smoke-check.sh
```

## Patch 5 - Doctor Warning Presentation

Status: completed in this patch set.

Problem:

- `vLLM Start Check` can pass while printing long stderr warning text from dependencies.
- Observed warning:
  - `torchvision` image extension warning.
- The check is PASS, but the long warning looks like a failure to a beginner.

Required change:

- Keep stderr for diagnostics, but present it as non-blocking warning when the check passed.
- Summarize long warning text by default.
- Provide a visible advanced/detail path for full stderr when implemented.

Success criteria:

- PASS remains PASS.
- User sees a short line like:

```text
non-blocking warning from version check: torchvision image extension warning
```

- Full details are not lost in structured result or log if already available.

Implementation:

- `format_vllm_doctor_report(...)` now labels stderr from passing checks as `warning: non-blocking stderr`.
- Long stderr detail lines are summarized in the user-facing report.
- The underlying `DoctorCheck.stderr` field remains unchanged for diagnostics and tests.

Suggested verification:

```sh
./vllm-suite
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_vllm_doctor_uses_wrapper_for_version_and_python_for_torch_hip -v
bash scripts/smoke-check.sh
```

## Patch 6 - Profile Preview With Visible Advanced Defaults

Problem:

- Selected profile preview is useful but too long for a standalone first-user flow.
- The target user may only know that they built vLLM and want to load a model.
- The default preview currently teaches implementation details before answering whether launch is likely to work.
- Advanced values should not disappear. They should be shown with defaults and reset keys.
- It prints:
  - every profile field;
  - validation;
  - model inspection;
  - cache env;
  - full command;
  - preflight;
  - all editable fields and hints;
  - host guidance;
  - local model guidance.

Required change:

- Make the top of the preview immediately useful:
  - selected model name/path;
  - launch endpoint;
  - whether required model files look present;
  - whether the profile is launchable;
  - the next safe action.
- Do not rename known technical terms to make them look simpler:
  - show `preflight`, then explain it as a start check before launch;
  - show `profile schema`, then explain it as the saved settings format;
  - show `OpenAI-compatible endpoint`, then explain it as the server address clients use.
- Keep full command, editable field specs, JSON, cache env, parser details, and long guidance accessible from visible advanced/detail actions.
- Advanced values must show:
  - current value;
  - recommended default;
  - `[D] Use default` or equivalent reset key;
  - `[C] Cancel / do not save`;
  - `[S] Save` only when the user explicitly wants to write.
- Keep one command preview line available, but label it as advanced/for debugging if it is long.

Success criteria:

- `[P] Selected profile preview` starts as a practical beginner preview.
- A user can decide whether to launch without understanding vLLM internals.
- Advanced details remain visible and reversible, not hidden.
- No profile schema changes.

Suggested verification:

```sh
./vllm-suite
bash scripts/smoke-check.sh
```

## Patch 7 - Gemma4 Parser Profile Check

Problem:

- Observed selected profile:

```text
Profile : gemma4-26b-awq-auto
extra_args: --served-model-name gemma4-26b-awq-auto --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes
```

- Existing runtime notes say Gemma4 worked with:

```text
--tool-call-parser gemma4
```

- Existing notes also say Gemma4 with `--tool-call-parser hermes` caused raw tool-call markup leakage.

Required change:

- Verify whether the stored selected profile is stale, intentionally different, or wrong.
- If wrong, update the verified Gemma4 profile helper/default to use `gemma4`.
- Add or adjust a test so Gemma4 verified profile uses the expected parser.
- Do not silently rewrite arbitrary user profiles without an explicit migration/confirmation path.

Success criteria:

- Built-in/verified Gemma4 profile uses `--tool-call-parser gemma4`.
- Existing user profile files are not overwritten unexpectedly.
- UI warns if a known Gemma4 model profile uses a likely mismatched parser.

Suggested verification:

```sh
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_vllm_profile_menu_can_copy_verified_gemma4_profile_to_custom_draft -v
python3 -m unittest discover -v
bash scripts/smoke-check.sh
```

## Patch 8 - Skin Contract Should Expose Selected vs Latest Distinction

Problem:

- `./vllm-suite --skin status` currently returns status fields but the distinction between selected profile and latest run should be explicit enough for Rust skin.

Required change:

- Ensure skin status has separate structured fields:
  - `selected_profile`
  - `selected_profile_model`
  - `latest_run`
  - `latest_run_model`
  - `latest_run_endpoint`
  - `latest_run_status`
  - `status_target_policy`: likely `latest_run`
- Keep existing fields for compatibility if already emitted.

Success criteria:

- Rust skin can render selected profile and latest run separately without inference.
- JSON remains valid and stable.

Suggested verification:

```sh
./vllm-suite --skin status | python3 -m json.tool
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_skin_manifest_returns_engine_specific_contracts -v
bash scripts/smoke-check.sh
```

## Patch 9 - Move Skin Helpers Into Control Modules

Problem:

- Skin contract builders currently live in `llama-launcher-complete.py`.
- That was acceptable for the first separation baseline, but standalone engines should eventually own their control responses.

Required change:

- Add `modules/control_schema.py`.
- Add `modules/vllm/control.py`.
- Move vLLM `manifest/menu/actions/status` construction into `modules/vllm/control.py`.
- Keep behavior identical.

Success criteria:

- `./vllm-suite --skin ...` output shape remains the same.
- `llama-launcher-complete.py` becomes thinner.
- No runtime launch behavior change.

Suggested verification:

```sh
./vllm-suite --skin manifest | python3 -m json.tool
./vllm-suite --skin menu | python3 -m json.tool
./vllm-suite --skin actions | python3 -m json.tool
./vllm-suite --skin status | python3 -m json.tool
bash scripts/smoke-check.sh
```

## Patch 10 - Unified vLLM Menu Color And Group Policy

Status: completed in this patch set.

Problem:

- The vLLM candidate list is easy to scan because status badges and model entries have a consistent visual shape.
- The surrounding menus were still flat lists of `[key] label` rows.
- Without visible group dividers, beginner users have to infer which actions belong to model scan, selected profile, server checks, profile files, or navigation.

Required change:

- Add one shared menu rendering policy for the vLLM standalone menu hierarchy:
  - color the actual menu key on interactive terminals;
  - keep brackets dimmer than the key so the key is the visual target;
  - keep plain `[K]` output when color is disabled, when stdout is not a TTY, under `NO_COLOR`, or under `TERM=dumb`;
  - add visible group dividers before related action clusters.
- Apply the policy to the vLLM engine menu and its submenus:
  - selected profile workspace;
  - profile JSON/preset;
  - selected profile settings;
  - default profile policy;
  - tool-call parser selection;
  - model readiness detail actions;
  - server check/log/stop;
  - advanced manual server record/path;
  - API/Hermes checks.
- Every submenu that presents selectable actions must include an explicit `[R] return` or `[R] Back` action. Unknown input as implicit cancel is not enough for beginner users.

Success criteria:

- A piped `./vllm-suite` transcript remains grep/test friendly.
- A real terminal shows clearer key emphasis.
- Beginners can distinguish model scan, profile, server/check, file, and navigation action groups without reading every line.
- Beginners can always see how to go back from a submenu before choosing an action.
- No launch, save, scan, or skin JSON behavior changes.

Suggested verification:

```sh
printf 'Q\n' | ./vllm-suite
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_vllm_workspace_menu_groups_make_actions_scannable -v
python3 -m unittest tests.test_beginner_flow.BeginnerFlowTests.test_terminal_status_color_helpers_keep_visible_words_and_honor_no_color -v
bash scripts/smoke-check.sh
```

## Do Not Combine With These Patches

Do not mix the above patches with:

- model download behavior;
- launch command behavior changes;
- Hermes config write behavior;
- shim deletion;
- large import-path rewrites;
- Rust implementation;
- generic backend abstraction;
- public release packaging.

Keep each patch small enough that a failed test points to one responsibility.
