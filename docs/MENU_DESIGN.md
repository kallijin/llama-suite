# llama-suite Menu Design

This document defines the intended beginner-first menu architecture for
llama-suite before the next UI routing patches.

It is a design baseline only. It does not change runtime behavior.

## Goal

llama-suite is a beta operations launcher for real local AI engines. The
current features are useful, but the menus are still too developer-centric.
A beginner should not have to open many menu items just to discover which
engine a command belongs to.

The core structural decision is:

```text
The main menu branches by engine first.
```

llama.cpp and vLLM are independent local AI engine workflows controlled by
llama-suite. vLLM must not look like an add-on under llama.cpp. llama.cpp
must not look like a legacy corner under vLLM.

## Core Rule

Core is AI-first. UI is human-first.

Internal IDs, state files, dataclasses, run records, and helper names can
remain technical and stable. Human-facing menus are render output and should
be organized by user intent and engine type.

## Main Menu

The main menu should be short and engine-separated:

```text
llama-suite local AI engine control

What do you want to run?

[1] llama.cpp engine
    GGUF models, llama-server, ctx/KV/cache, scripts

[2] vLLM engine
    HF model folders, vLLM profiles, API server, Hermes checks

[3] Shared tools
    system info, GPU/VRAM status, logs, backups, integration registration

[Q] Quit
```

## Design Principles

1. Main menu branches by engine first.
2. llama.cpp and vLLM have independent workspaces.
3. Top-level labels should answer: "what does the user want to operate?"
4. Submenus should explain what the screen does before showing actions.
5. Dangerous actions must say whether they launch, stop, delete, overwrite,
   download, or use GPU memory.
6. Saving a profile must be clearly different from launching a model.
7. Import/export model-folder profile actions must say they do not launch a
   model.
8. Advanced/internal tools should be reachable but should not dominate
   beginner screens.
9. Internal IDs and dataclasses should remain stable; UI labels are render
   output.
10. UI should not force beginners to understand implementation names before
    they can operate the launcher.

## llama.cpp Engine Workspace

```text
[1] llama.cpp engine

Current llama.cpp status
- Selected GGUF model
- Running server
- Endpoint
- Context size
- KV cache
- Script/profile state

What do you want to do?

[1] Choose GGUF model
[2] Edit llama.cpp run settings
[3] Preview command
[4] Start / stop llama-server
[5] Save or load script/profile
[6] Check server
[7] View logs
[A] Advanced llama.cpp tools
[R] Back
```

The llama.cpp workspace owns GGUF model selection, llama-server command
preview, ctx/KV/cache settings, llama.cpp scripts, and llama.cpp run/log
operations.

## vLLM Engine Workspace

```text
[2] vLLM engine

Current vLLM status
- Selected model folder
- Selected profile
- Running server
- Endpoint
- API status
- Hermes status

What do you want to do?

[1] Choose vLLM model folder
[2] Edit vLLM profile
[3] Apply safe defaults
[4] Import / save model-folder profile
[5] Preview command
[6] Start / stop vLLM server
[7] Check API / Hermes
[8] View logs
[A] Advanced vLLM tools
[R] Back
```

The vLLM workspace owns HF/local model folder selection, vLLM profile editing,
safe desktop defaults, model-directory profile hints, vLLM command preview,
vLLM server lifecycle, API checks, Hermes checks, and vLLM logs.

## Shared Tools Workspace

```text
[3] Shared tools

[1] System info
[2] GPU / VRAM status
[3] Hermes / OpenClaw registration
[4] Global logs and backups
[5] Recovery records
[6] Developer diagnostics
[R] Back
```

Shared tools are for system-wide inspection and integration tasks that are not
owned by exactly one engine.

## Current Menu to Future Menu Mapping

| Current concept | Future location |
| --- | --- |
| GGUF model selection | llama.cpp engine / Choose GGUF model |
| llama.cpp ctx/KV/flash-attn settings | llama.cpp engine / Edit llama.cpp run settings |
| llama.cpp final preview | llama.cpp engine / Preview command |
| llama.cpp one-shot run | llama.cpp engine / Start / stop llama-server |
| script generation | llama.cpp engine / Save or load script/profile |
| script management | llama.cpp engine / Save or load script/profile |
| vLLM selected profile settings | vLLM engine / Edit vLLM profile |
| vLLM default profile policy | vLLM engine / Apply safe defaults |
| vLLM command preview / preflight | vLLM engine / Preview command |
| vLLM selected profile launch | vLLM engine / Start / stop vLLM server |
| vLLM latest run status/log/stop | vLLM engine / Start / stop vLLM server or View logs |
| vLLM API smoke | vLLM engine / Check API / Hermes / API Connection Test |
| Hermes smoke | vLLM engine / Check API / Hermes / Hermes Chat Test |
| Hermes tool-agent smoke | vLLM engine / Check API / Hermes / Hermes Tool / Raw Markup Check |
| Import profile from model directory | vLLM engine / Import / save model-folder profile |
| Save profile hint to model directory | vLLM engine / Import / save model-folder profile |
| run record | Shared tools / Recovery records or engine-specific View logs |
| system info | Shared tools / System info |
| GPU / VRAM status | Shared tools / GPU / VRAM status |
| Hermes/OpenClaw registration | Shared tools / Hermes / OpenClaw registration |
| developer diagnostics | Shared tools / Developer diagnostics |

## Migration Strategy

Recommended patch order:

1. Add this design document.
2. Add a new main menu shell that branches into llama.cpp / vLLM / Shared
   tools, while preserving existing actions.
3. Move existing llama.cpp actions under llama.cpp engine.
4. Move existing vLLM actions under vLLM engine.
5. Move system info, registration, logs, backups, and recovery under Shared
   tools where appropriate.
6. Only after the structure is stable, improve wording inside each submenu.
7. Do not combine large routing refactors with new runtime features.

## Non-goals for This Design Pass

- No behavior change
- No launch logic change
- No profile schema change
- No Rust skin work
- No Bluejeans work
- No model download logic
- No automatic migration
- No deletion behavior changes
- No new vLLM or llama.cpp runtime options
