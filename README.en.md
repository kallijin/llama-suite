# llama-suite

**Author:** kalijim  
**Co-author / Assistant:** ChatGPT, kalijin's assistant

[Korean original](README.md)

Korean README is primary for now. This English document is a working translation.

---

## Current Status

`llama-suite` is currently a private experimental toolkit.

It is not a public release yet, and it does not promise installation, distribution, or compatibility support for general users.
After `v0.4-scope-freeze`, stabilization, verification, and documentation take priority over new features.

### Implemented

- Basic local launcher flow
- Modular structure
- llama.cpp backend discovery and inspection path
- Model/system information menus
- Some probe and run-helper routines
- Ideas parking lot and scope freeze notes

### Planned

- Reproducibility checks for implemented features
- README/IDEAS cleanup
- Reproduction check for the root execution warning flow
- Stabilization of model profile and probe result records

### Verification

Run the quick stabilization check from the repository root:

```sh
bash scripts/smoke-check.sh
```

This check does not run models. It only checks git status, recent commits, Python compilation, key file/directory existence, and key module imports.

### Parked

- Automatic large model directory discovery
- Automatic network/system configuration inference
- External agent execution integration
- Unicode/multilingual cleanup
- Public release preparation

---

## Purpose

`llama-suite` is a small, rough, but practical toolbox for local LLM operations.

The project aims to tie `llama.cpp`, `Hermes Agent`, `tmux`, `Tailscale`, local model profiles, and test routines into one operational flow.

A polished UI or a heavy framework is not the goal.  
The goal is clear.

- Start local models reliably.
- Record per-model runtime conditions.
- Quickly verify the connection between Hermes and llama.cpp.
- Test thinking output, tool calls, context size, and VRAM pressure.
- Avoid losing both failed and successful configurations.
- Keep as much control as possible local.

---

## Basic Philosophy

### 1. Disk size is acceptable. Runtime waste is not.

More code is acceptable.  
More modules are acceptable.  
More config files and profiles are acceptable.

But the project should avoid structures that waste CPU, RAM, or VRAM immediately on startup.

VRAM belongs to the model.  
CPU is for search, log analysis, tests, and cleanup work.  
The GUI is a control panel, not the main subject.

---

### 2. Keep Core dumb and Modules independent.

Do not build a complex framework.

Core only does the following:

- Reads configuration.
- Calls modules.
- Shows results.
- Records failures.

Each feature is separated into an independent module.

Candidate modules:

- `model_scan`
- `runner_tmux`
- `probes`
- `hermes_sync`
- `profiles`
- `local_search`
- `clean_search`
- `web_sidecar`
- `mcp_bridge`

A failing module can be discarded.  
Core must survive.

---

### 3. CLI is the final recovery path.

Even if the GUI dies, the web panel breaks, or the browser gets confused, the CLI must remain alive.

Basic operations must be possible from the terminal.

- Model selection
- Server start/stop
- tmux log inspection
- `/health` check
- `/v1/models` check
- no-thinking test
- tool-call test
- Hermes config sync

The GUI is convenience, not a lifeline.

---

### 4. The XWin GUI may be ugly.

Fancy graphics are not the goal.

A control panel that can open under KDE, GNOME, XFCE, or a simple Xorg environment is enough.  
It may look like an old plain X11 utility.

Avoid:

- Electron
- QtWebEngine
- Heavy QML/QtQuick
- Unnecessary animation
- Transparent/blurred UI
- Decorative elements that consume VRAM

What is needed is function, not decoration.

---

### 5. Per-model records are an asset.

Local models behave differently from each other.

Even the same GGUF can produce different results depending on:

- llama.cpp version
- ROCm/Vulkan/backend
- chat template
- Jinja handling
- reasoning options
- tool-call parser
- context size
- KV cache
- Hermes request style

Therefore, success/failure records should be kept per model.

Example:

```text
EXAONE 4.0 32B Q4_K_M
- ctx 95000: possible timeout / wedge
- ctx 92000: direct + Hermes passed 3 repeated runs
- no-thinking: pass
- direct tool-call: pass
- Hermes terminal tool: fail / text imitation
- chatml template: fail / <|im_end|> leak
```
