from __future__ import annotations

import os
import re
import shlex
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.llama_cpp.backends import get_backend, resolve_backend_server_bin
from modules.llama_cpp.config_store import (
    normalize_extra_args,
)


DEFAULT_SCRIPTS_DIR = Path(os.path.expanduser("~/.hermes/llama-scripts"))
MODEL_SIZE_CTX_RULES = [
    (20.0, 28.0, 92000),
    (30.0, 36.0, 80000),
]


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


def detect_model_size_billion(model_name: str, model_path: str) -> float | None:
    text = f"{model_name} {Path(model_path).name} {Path(model_path).parent.name}"
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z0-9])", text)
    if not matches:
        return None
    try:
        return max(float(x) for x in matches)
    except ValueError:
        return None


def resolve_ctx_size(model_name: str, model_path: str, cfg: dict[str, Any]) -> int:
    model_size = detect_model_size_billion(model_name, model_path)
    if model_size is not None:
        for min_size, max_size, ctx_size in MODEL_SIZE_CTX_RULES:
            if min_size <= model_size <= max_size:
                return ctx_size
    return int(cfg["ctx_size"])


def script_snapshot_hash(model_name: str, model_path: str, cfg: dict[str, Any]) -> str:
    payload = "|".join(
        [
            model_name,
            model_path,
            str(resolve_ctx_size(model_name, model_path, cfg)),
            str(cfg.get("host")),
            str(cfg.get("port")),
            str(cfg.get("reasoning", "off")),
            str(cfg.get("reasoning_budget", 0)),
            str(bool(cfg.get("enable_thinking", False))),
            " ".join(normalize_extra_args(cfg.get("extra_args", []))),
            " ".join(normalize_extra_args(cfg.get("custom_args", []))),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:6]


def generated_script_name(model_name: str, model_path: str, cfg: dict[str, Any], timestamp: str | None = None) -> str:
    safe_name = safe_generated_script_name(model_name, limit=44)
    ctx_size = resolve_ctx_size(model_name, model_path, cfg)
    thinking = "thinkon" if bool(cfg.get("enable_thinking", False)) else "thinkoff"
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = script_snapshot_hash(model_name, model_path, cfg)
    return f"{safe_name}__ctx{ctx_size}__{thinking}__{stamp}__{short_hash}.sh"


def collision_free_script_path(target_dir: Path, script_name: str) -> Path:
    script_path = target_dir / script_name
    if not script_path.exists():
        return script_path

    stem = script_path.stem
    suffix = script_path.suffix
    counter = 2
    while True:
        candidate = target_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_llama_command(model_name: str, model_path: str, cfg: dict[str, Any]) -> list[str]:
    bin_path = resolve_llama_bin(cfg)
    model_id = Path(model_path).name if cfg.get("alias_by_file", True) else model_name
    ctx_size = resolve_ctx_size(model_name, model_path, cfg)
    command = [
        bin_path,
        "-m",
        model_path,
        "--host",
        str(cfg["host"]),
        "--port",
        str(cfg["port"]),
        "--ctx-size",
        str(ctx_size),
    ]
    if cfg.get("jinja", True):
        command.append("--jinja")
    if cfg.get("alias_by_file", True):
        command.extend(["--alias", model_id])
    reasoning = str(cfg.get("reasoning", "off"))
    if reasoning in {"off", "auto", "on"}:
        command.extend(["--reasoning", reasoning])
    command.extend(
        [
            "--reasoning-budget",
            str(int(cfg.get("reasoning_budget", 0))),
            "--chat-template-kwargs",
            '{"enable_thinking":' + ("true" if bool(cfg.get("enable_thinking", False)) else "false") + "}",
        ]
    )
    command.extend(normalize_extra_args(cfg.get("extra_args", [])))
    command.extend(normalize_extra_args(cfg.get("custom_args", [])))
    return command


def command_preview(model_name: str, model_path: str, cfg: dict[str, Any]) -> str:
    return shlex.join(build_llama_command(model_name, model_path, cfg))


def parse_generated_script(path: str | os.PathLike[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    target = Path(path)
    for line in target.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key not in {
            "SERVER_BIN",
            "MODEL",
            "MODEL_ID",
            "MODEL_PATH",
            "HOST",
            "PORT",
            "CTX_SIZE",
            "REASONING_MODE",
            "REASONING_BUDGET",
            "CHAT_TEMPLATE_KWARGS",
            "EXTRA_ARGS",
            "CUSTOM_ARGS",
        }:
            continue
        if key == "EXTRA_ARGS":
            inside = raw.removeprefix("(").removesuffix(")")
            fields["extra_args"] = normalize_extra_args(inside)
            continue
        if key == "CUSTOM_ARGS":
            inside = raw.removeprefix("(").removesuffix(")")
            fields["custom_args"] = normalize_extra_args(inside)
            continue
        try:
            parsed = shlex.split(raw)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = raw.strip("'").strip('"')
        fields[key] = value

    cfg: dict[str, Any] = {}
    if fields.get("SERVER_BIN"):
        cfg["llama_bin"] = fields["SERVER_BIN"]
    if fields.get("HOST"):
        cfg["host"] = fields["HOST"]
    if fields.get("PORT"):
        cfg["port"] = int(fields["PORT"])
    if fields.get("CTX_SIZE"):
        cfg["ctx_size"] = int(fields["CTX_SIZE"])
    if fields.get("REASONING_MODE"):
        cfg["reasoning"] = fields["REASONING_MODE"]
    if fields.get("REASONING_BUDGET"):
        cfg["reasoning_budget"] = int(fields["REASONING_BUDGET"])
    if "extra_args" in fields:
        cfg["extra_args"] = fields["extra_args"]
    if "custom_args" in fields:
        cfg["custom_args"] = fields["custom_args"]
    if fields.get("CHAT_TEMPLATE_KWARGS"):
        cfg["enable_thinking"] = '"enable_thinking":true' in str(fields["CHAT_TEMPLATE_KWARGS"]).replace(" ", "")

    return {
        "model_name": fields.get("MODEL"),
        "model_path": fields.get("MODEL_PATH"),
        "cfg": cfg,
    }


def generate_script(
    model_name: str,
    model_path: str,
    cfg: dict[str, Any],
    scripts_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    target_dir = Path(scripts_dir) if scripts_dir is not None else DEFAULT_SCRIPTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    script_name = generated_script_name(model_name, model_path, cfg)
    script_path = collision_free_script_path(target_dir, script_name)
    script_name = script_path.name

    bin_path = resolve_llama_bin(cfg)
    model_id = Path(model_path).name if cfg.get("alias_by_file", True) else model_name
    ctx_size = resolve_ctx_size(model_name, model_path, cfg)
    extra_args = normalize_extra_args(cfg.get("extra_args", []))
    extra_args_shell = " ".join(shlex.quote(x) for x in extra_args)
    custom_args_shell = " ".join(shlex.quote(x) for x in normalize_extra_args(cfg.get("custom_args", [])))

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
    cmd_lines.append('    "${EXTRA_ARGS[@]}" \\')
    cmd_lines.append('    "${CUSTOM_ARGS[@]}"')

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
CTX_SIZE={ctx_size}
REASONING_MODE={shlex.quote(reasoning)}
REASONING_BUDGET={reasoning_budget}
CHAT_TEMPLATE_KWARGS='{{"enable_thinking":{enable_thinking}}}'
EXTRA_ARGS=({extra_args_shell})
CUSTOM_ARGS=({custom_args_shell})

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
