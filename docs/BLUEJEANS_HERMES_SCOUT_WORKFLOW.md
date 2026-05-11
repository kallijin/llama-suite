# Bluejeans Hermes Scout Workflow

This document defines how Hermes Agent should behave when working around the future `llama-suite-bluejeans-skin` Rust skin project.

The purpose is to prevent small or medium local agents from accidentally becoming architects, inventing protocol fields, or changing core behavior silently.

## Core idea

```text
Hermes is a scout and request writer.
Codex is the periodic architect/reviewer.
llama-suite is the Python core and source of truth.
Bluejeans is the Rust skin/renderer.
GitHub issues and request docs are the handoff channel.
```

Hermes should not try to solve every architectural problem while coding. If Hermes gets blocked, uncertain, or needs a core function, it should leave a structured request in Git instead of improvising a hidden workaround.

## Mandatory read order for Hermes

Before doing Bluejeans-related work, Hermes should read:

```text
1. README.md
2. docs/AI_FIRST_CORE_HUMAN_UI.md
3. docs/BLUEJEANS_HERMES_SCOUT_WORKFLOW.md
4. docs/RUST_SKIN_UI_PROTOCOL.md, if it exists
5. docs/CROSS_REPO_AI_COLLABORATION.md, if it exists
6. protocol/fixtures/*.json, if they exist
7. relevant open GitHub issues labeled blocker/protocol/skin/core
```

If working inside the future Bluejeans repo, Hermes should additionally read:

```text
1. docs/ROLE.md
2. docs/UPSTREAM_CONTRACT.md
3. docs/AI_COLLABORATION.md
4. docs/HERMES_SCOUT_RULES.md, if it exists
5. .agents/roles.json, if it exists
```

Hermes must report which files it read before making design-affecting changes.

## Hermes role

Hermes may:

```text
- inspect llama-suite GitHub as a scout
- inspect Bluejeans GitHub as a scout
- read protocol docs and fixtures
- summarize missing contract information
- draft structured requests
- open or prepare GitHub issues
- report blockers
- ask for missing screen descriptor fields
- ask for missing item.kind behavior
- ask for missing fixture examples
- make small local code changes only inside explicitly assigned scope
```

Hermes must not:

```text
- act as final architect
- silently redesign Bluejeans
- silently redesign llama-suite core
- invent protocol fields without filing a request
- parse human labels as logic
- write llama-suite persistent state directly
- write model-registry/discovery/run JSON directly from Rust
- turn Rust skin into a second launcher
- bypass Python core ownership of meaning and execution
- make large cross-repo changes without explicit instruction
```

## Codex role

Codex may periodically review Bluejeans and decide direction.

Codex should:

```text
- read Hermes blocker/request issues
- decide whether a problem belongs in Bluejeans or llama-suite core
- update protocol docs when accepted
- update fixtures when protocol changes
- review Rust code for boundary violations
- ensure Rust remains a renderer only
- ensure Python core remains usable without Rust
- produce architectural correction commits when needed
```

## When Hermes gets blocked

Hermes should not improvise a hidden workaround.

Hermes should create or draft a request in this shape:

```text
Title:
[blocker] short description

Agent:
Hermes

Working repo:
llama-suite-bluejeans-skin or llama-suite

Working area:
Example: Bluejeans vLLM workspace screen

Problem:
What failed or is unclear.

Needed from owner:
What is needed from llama-suite core or Bluejeans skin.

Suggested contract:
Optional JSON or field proposal.

Can continue without this?
yes/no/partial

Recommended owner:
llama-suite core / bluejeans skin / protocol / undecided

Fixture involved:
Path or proposed fixture name.

Notes:
Short operational details.
```

## Request examples

### Skin needs core field

```text
Title:
[skin-needs-core] Bluejeans needs disabled_reason for disabled menu items

Agent:
Hermes

Working repo:
llama-suite-bluejeans-skin

Problem:
Rust skin can render enabled=false, but cannot explain to the user why the button is disabled.

Needed from owner:
llama-suite core should optionally provide disabled_reason or human_hint in screen item descriptors.

Suggested contract:
{
  "id": "vllm.profile.launch",
  "enabled": false,
  "disabled_reason": "selected vLLM profile is not READY"
}

Can continue without this?
partial

Recommended owner:
llama-suite core

Fixture involved:
protocol/fixtures/vllm_workspace_disabled_launch.screen.json
```

### Core needs skin behavior

```text
Title:
[core-needs-skin] Render dangerous_action with a visible confirmation marker

Agent:
Hermes or Codex

Working repo:
llama-suite

Problem:
Actions with requires_confirmation=true should not look like normal buttons.

Needed from owner:
Bluejeans should render dangerous_action with a marker such as [!] or a distinct border style.

Suggested behavior:
If item.kind == dangerous_action or item.requires_confirmation == true, render a visible danger marker.

Can continue without this?
yes, but UX is weaker

Recommended owner:
bluejeans skin

Fixture involved:
protocol/fixtures/vllm_launch_confirm.screen.json
```

## GitHub issue labels

Recommended labels:

```text
from:hermes
from:codex
from:core-python
from:skin-rs
to:core-python
to:skin-rs
kind:proposal
kind:blocker
kind:protocol
kind:fixture
kind:rendering
kind:font
kind:request
status:accepted
status:rejected
status:deferred
```

## Local request files

If GitHub issue creation is unavailable, Hermes may create local request files instead.

Recommended path in Bluejeans:

```text
docs/requests/YYYYMMDD-HHMM-short-title.md
```

Recommended path in llama-suite:

```text
docs/bluejeans-requests/YYYYMMDD-HHMM-short-title.md
```

Local request files must use the same structure as GitHub issues and should be committed so Codex can review them later.

## Screen protocol boundary

Hermes must remember:

```text
Python core generates screen descriptors.
Rust skin renders screen descriptors.
Rust returns item_id events.
Python core decides meaning and execution.
```

Rust may inspect:

```text
item.kind
item.enabled
item.requires_confirmation
layout hints
font profile hints
label text for rendering only
```

Rust must not inspect label text to decide meaning.

Correct:

```text
clicked item_id = vllm.discovery.scan
send event to Python core
```

Incorrect:

```text
if label contains "모델 찾기" then scan models
```

## Skin failure rule

Rust Bluejeans skin is optional.

Python launcher/core must remain canonical and directly operable.

```text
./llama-suite      = canonical Python launcher
./llama-suite-rs   = optional Rust skin, future
```

If Rust skin fails, the user must still be able to run Python launcher directly.

## Review rhythm

Hermes should leave clear requests while working.
Codex should periodically review those requests.

Suggested Codex review loop:

```text
1. Read open Bluejeans issues labeled kind:blocker or kind:protocol.
2. Read docs/requests if GitHub issue access is unavailable.
3. Decide owner: Bluejeans skin, llama-suite core, protocol, or deferred.
4. Patch the correct repo or write an upstream request.
5. Update fixtures when protocol changes.
6. Commit the decision.
```

## Final rule

When uncertain, Hermes should not guess the architecture.

Hermes should write the uncertainty down in Git so Codex can review it later.

```text
Scout. Report. Request. Do not silently redesign.
```
