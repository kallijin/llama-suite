from __future__ import annotations

import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.backends import get_backend, resolve_backend_server_bin
from modules.config_store import (
    normalize_extra_args,
)


DEFAULT_SCRIPTS_DIR = Path(os.path.expanduser("~/.hermes/llama-scripts"))


def safe_generated_script_name(text: str, limit: int = 64) -> str:
    allowed = []
    for c in text[:limit]:
        if c.isalnum() or c in ("-", "_", "."):
            allowed.append(c)
        else:
            allowed.append("_")
    name = "".join(allowed).strip("_")
    return name or "model"


def resolve_llama_bin(cfg: dict[str, Any]) -> str:
    backend = get_backend(cfg.get("backend"))
    return resolve_backend_server_bin(backend, cfg.get("llama_bin"))


def generate_script(
    model_name: str,
    model_path: str,
    cfg: dict[str, Any],
    scripts_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    target_dir = Path(scripts_dir) if scripts_dir is not None else DEFAULT_SCRIPTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = safe_generated_script_name(model_name)
    script_name = f"{safe_name}_{ts}.sh"
    script_path = target_dir / script_name

    bin_path = resolve_llama_bin(cfg)
    model_id = Path(model_path).name if cfg.get("alias_by_file", True) else model_name
    extra_args = normalize_extra_args(cfg.get("extra_args", []))
    extra_args_shell = " ".join(shlex.quote(x) for x in extra_args)

    enable_thinking = "true" if bool(cfg.get("enable_thinking", False)) else "false"
    reasoning = str(cfg.get("reasoning", "off"))
    reasoning_budget = int(cfg.get("reasoning_budget", 0))

    cmd_lines = [
        '    -m "$MODEL_PATH" \\',
        '    --host "$HOST" \\',
        '    --port "$PORT" \\',
        '    --ctx-size "$CTX_SIZE" \\',
    ]

    if cfg.get("jinja", True):
        cmd_lines.append('    --jinja \\')

    if cfg.get("alias_by_file", True):
        cmd_lines.append('    --alias "$MODEL_ID" \\')

    # Qwen thinking-only 응답 방지용. reasoning=auto/on으로 바꾸면 사용자가 의도한 대로 따라간다.
    if reasoning in {"off", "auto", "on"}:
        cmd_lines.append('    --reasoning "$REASONING_MODE" \\')
    cmd_lines.append('    --reasoning-budget "$REASONING_BUDGET" \\')
    cmd_lines.append('    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \\')
    cmd_lines.append('    "${EXTRA_ARGS[@]}"')

    cmd_block = "\n".join(cmd_lines)

    script_content = f"""#!/usr/bin/env bash
# 🦙 LLAMA.CPP 실행 스크립트
# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 생성기: llama-launcher.py

set -euo pipefail

SERVER_BIN={shlex.quote(bin_path)}
MODEL={shlex.quote(model_name)}
MODEL_ID={shlex.quote(model_id)}
MODEL_PATH={shlex.quote(model_path)}
HOST={shlex.quote(str(cfg['host']))}
PORT={int(cfg['port'])}
CTX_SIZE={int(cfg['ctx_size'])}
REASONING_MODE={shlex.quote(reasoning)}
REASONING_BUDGET={reasoning_budget}
CHAT_TEMPLATE_KWARGS='{{"enable_thinking":{enable_thinking}}}'
EXTRA_ARGS=({extra_args_shell})

echo "🚀 Starting $MODEL"
echo "   model id : $MODEL_ID"
echo "   endpoint : http://$HOST:$PORT/v1"
echo "   ctx      : $CTX_SIZE"
echo "   reasoning: $REASONING_MODE, budget=$REASONING_BUDGET, template=$CHAT_TEMPLATE_KWARGS"
echo

exec "$SERVER_BIN" \\
{cmd_block}
"""

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_name, str(script_path)
