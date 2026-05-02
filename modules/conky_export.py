from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONKY_CONFIG_PATH = Path(os.path.expanduser("~/.hermes/llama-conky-export.json"))


@dataclass
class ConkyExportConfig:
    enabled: bool
    path: str | None
    updated_at: str | None


@dataclass
class ConkyExportResult:
    ok: bool
    path: str
    warning: str | None = None


def default_conky_config() -> ConkyExportConfig:
    return ConkyExportConfig(
        enabled=False,
        path=None,
        updated_at=None,
    )


def load_conky_config() -> ConkyExportConfig:
    cfg = default_conky_config()
    if not CONKY_CONFIG_PATH.exists():
        return cfg

    try:
        with CONKY_CONFIG_PATH.open() as f:
            data = json.load(f)
    except Exception:
        return cfg

    if not isinstance(data, dict):
        return cfg

    return ConkyExportConfig(
        enabled=bool(data.get("enabled", cfg.enabled)),
        path=data.get("path") if data.get("path") else None,
        updated_at=data.get("updated_at") if data.get("updated_at") else None,
    )


def save_conky_config(cfg: ConkyExportConfig) -> None:
    CONKY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONKY_CONFIG_PATH.open("w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def default_conky_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "llama-suite" / "conky.txt"
    return Path.home() / ".cache" / "llama-suite" / "conky.txt"


def render_conky_text(state: dict[str, Any]) -> str:
    lines = [
        "llama-suite",
        f"model: {_value(state.get('model') or state.get('model_name'))}",
        f"server: {_value(state.get('server') or state.get('endpoint'))}",
        f"status: {_value(state.get('status'))}",
        f"backend: {_value(state.get('backend'))}",
        f"ctx: {_value(state.get('ctx') or state.get('ctx_size'))}",
        f"profile: {_value(state.get('profile'))}",
        f"updated: {_value(state.get('updated') or _now_iso())}",
    ]
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_conky_snapshot(
    state: dict[str, Any],
    path: str | os.PathLike[str] | None = None,
) -> ConkyExportResult:
    target = Path(path) if path is not None else default_conky_path()
    try:
        atomic_write_text(target, render_conky_text(state))
    except Exception as e:
        return ConkyExportResult(
            ok=False,
            path=str(target),
            warning=f"{type(e).__name__}: {e}",
        )
    return ConkyExportResult(ok=True, path=str(target))


def live_export_if_enabled(state: dict[str, Any]) -> ConkyExportResult | None:
    cfg = load_conky_config()
    if not cfg.enabled:
        return None

    target = Path(cfg.path) if cfg.path else default_conky_path()
    result = write_conky_snapshot(state, path=target)
    if result.ok:
        cfg.path = str(target)
        cfg.updated_at = _now_iso()
        try:
            save_conky_config(cfg)
        except Exception as e:
            return ConkyExportResult(
                ok=False,
                path=str(target),
                warning=f"{type(e).__name__}: {e}",
            )
    return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
