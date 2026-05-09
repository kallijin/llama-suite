# Patch Notes - 2026-05-09

## Scope

Beginner-first llama.cpp operations console redesign work.

The main design rule for today's work:

```text
메인 화면은 임시 작업 설정 편집기.
저장은 명시적.
실행은 현재 화면값 기준.
스크립트는 수정하지 않는 실행 스냅샷.
Hermes/OpenClaw는 등록된 경로만 안전하게 연동.
```

## Commits

```text
27fe099 Add short launcher wrapper
e4b7bd7 Show planned run summary
5bba8cd Add last run recovery cache
fa62b3b Unify script generation action
250a75c Move save action under settings
e22c2b7 Add safe integration registration
37dfacd Add structured parameter controls
64904e4 Add beginner working draft flow
```

## User-Facing Changes

- Added a short executable wrapper:

```sh
./llama-suite
```

- Main screen now acts as a working draft editor.
- Settings are not saved automatically.
- `[A] 설정 변경 / 현재 설정 저장` owns explicit save behavior.
- `[G] 새 스크립트 생성` now covers both:
  - `[1] 생성만`
  - `[2] 생성 후 실행`
- `[O] 1회 실행` writes a recovery cache before running.
- `[L] 불러오기 -> [3] last run record` restores the last one-time run parameters.
- Main screen shows `실행 예정 요약`.
- Structured parameter cards show value, source, explanation, and change action.
- Custom args are separated from structured stable args.
- Conflicts between custom args and structured args are detected.
- Hermes/OpenClaw integration now uses registered paths as the source of truth.
- OpenClaw remains read-only inspection only.

## Safety Notes

- Generated scripts are execution snapshots.
- Existing generated scripts are not silently overwritten.
- One-time run does not save profile/config and does not create a permanent script.
- One-time run parameters are recoverable through the last-run cache.
- Hermes config write support is still intentionally deferred until diff, backup, confirmation, parse validation, and atomic replace are implemented.
- The repository remains private on GitHub.

## Verification

Latest verification performed during the session:

```text
14 tests OK
SMOKE CHECK OK
```

Manual flows tested:

- Start launcher with `./llama-suite`.
- Select a model and see it reflected in `실행 예정 요약`.
- Change `Context Size` and see `ctx` update in the main summary.
- Restore parameters from `[L] 불러오기 -> [3] last run record`.
- Register a temporary Hermes config path.
- Use unified `[G] 새 스크립트 생성` flow and see `[1] 생성만` / `[2] 생성 후 실행`.

## Known Remaining Work

- Hermes config write workflow:
  - show diff
  - create `config.yaml.old0`
  - create rotating `old1`, `old2`, `old3`
  - user confirmation
  - temp file write
  - parse validation
  - atomic replace
- OpenClaw write support remains intentionally unimplemented.
- Last run record exists as a recovery cache, not a saved profile.
- A pre-existing local `.gitignore` change adds `.worktrees/`; it was not included in today's feature commits.
