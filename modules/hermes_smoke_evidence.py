from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_HERMES_EVIDENCE_ROOT = "~/.local/state/llama-suite/evidence/hermes"
HERMES_SMOKE_EVIDENCE_SCHEMA = "llama-suite.hermes-smoke-evidence.v1"

_REDACTION_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-[REDACTED]"),
    (re.compile(r"ghp_[A-Za-z0-9_]{8,}"), "ghp_[REDACTED]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "github_pat_[REDACTED]"),
    (re.compile(r"Bearer\s+[^\s'\";,]+", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"(?i)([\"'](?:api[_-]?key|password|token|authorization|secret)[\"']\s*:\s*[\"'])[^\"']+([\"'])"), r"\1[REDACTED]\2"),
    (re.compile(r"(?i)\b(api[_-]?key|password|token|authorization|secret)\b\s*[:=]\s*['\"]?[^'\"\s,;}]+"), r"\1=[REDACTED]"),
)


@dataclass(frozen=True)
class HermesSmokeEvidenceResult:
    ok: bool
    evidence_path: str | None
    messages: list[str]


def redact_sensitive_text(text: str) -> str:
    redacted = str(text or "")
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def write_hermes_smoke_evidence(
    result: Any,
    *,
    evidence_root: str | Path | None = None,
    timestamp: str | None = None,
    max_chars: int = 4000,
) -> HermesSmokeEvidenceResult:
    raw_markup_detected = bool(getattr(result, "raw_markup_detected", False))
    ok = bool(getattr(result, "ok", False))
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    status = str(getattr(result, "status", "") or "not_run")
    smoke_kind = str(getattr(result, "smoke_kind", "") or "unknown")
    if ok and not raw_markup_detected:
        return HermesSmokeEvidenceResult(True, None, ["not saved: smoke passed without raw markup"])
    if status in {"not_run", "unsupported"} and not raw_markup_detected and not stdout and not stderr:
        return HermesSmokeEvidenceResult(True, None, [f"not saved: {status} smoke produced no runtime output"])

    root = Path(evidence_root or DEFAULT_HERMES_EVIDENCE_ROOT).expanduser()
    safe_kind = _safe_filename_part(smoke_kind)
    created_at, file_stamp = _timestamp_values(timestamp)
    evidence_path = root / f"hermes-{safe_kind}-{file_stamp}.json"
    tmp_path = evidence_path.with_name(f".{evidence_path.name}.tmp")
    max_len = max(0, int(max_chars))
    payload = {
        "schema": HERMES_SMOKE_EVIDENCE_SCHEMA,
        "created_at": created_at,
        "smoke_kind": smoke_kind,
        "status": status,
        "ok": ok,
        "returncode": getattr(result, "returncode", None),
        "raw_markup_detected": raw_markup_detected,
        "raw_markup_patterns": list(getattr(result, "raw_markup_patterns", None) or []),
        "messages": list(getattr(result, "messages", None) or []),
        "command_excerpt": _excerpt(_command_text(getattr(result, "command", None) or []), max_len),
        "stdout_excerpt": _excerpt(stdout, max_len),
        "stderr_excerpt": _excerpt(stderr, max_len),
        "notes": [],
    }

    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, evidence_path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return HermesSmokeEvidenceResult(False, None, [f"save failed: {exc}"])

    return HermesSmokeEvidenceResult(True, str(evidence_path), [f"saved: {evidence_path}"])


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _excerpt(text: str, max_chars: int) -> str:
    redacted = redact_sensitive_text(text)
    if len(redacted) <= max_chars:
        return redacted
    marker = "\n[truncated]"
    if max_chars <= len(marker):
        return redacted[:max_chars]
    return redacted[: max_chars - len(marker)] + marker


def _timestamp_values(timestamp: str | None) -> tuple[str, str]:
    if timestamp:
        safe = _safe_filename_part(timestamp)
        return timestamp, safe
    now = datetime.now().astimezone()
    return now.isoformat(timespec="seconds"), now.strftime("%Y%m%d-%H%M%S")


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "unknown")).strip("-._")
    return safe or "unknown"
