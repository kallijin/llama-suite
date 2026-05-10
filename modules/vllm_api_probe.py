from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from modules.vllm_runner import latest_vllm_run_record


@dataclass(frozen=True)
class VllmApiProbeCheck:
    name: str
    ok: bool
    message: str
    status_code: int | None = None


@dataclass(frozen=True)
class VllmApiSmokeResult:
    ok: bool
    base_url: str | None
    model_id: str | None
    checks: list[VllmApiProbeCheck]
    messages: list[str]


def run_vllm_api_smoke(
    *,
    latest_record: Any = None,
    opener: Any = None,
    timeout: float = 15.0,
) -> VllmApiSmokeResult:
    latest = latest_record or latest_vllm_run_record()
    if not latest.ok or latest.record is None:
        return VllmApiSmokeResult(
            ok=False,
            base_url=None,
            model_id=None,
            checks=[],
            messages=["latest vLLM run record is missing or invalid", *latest.messages],
        )

    record = latest.record
    if not record.host or not record.port:
        return VllmApiSmokeResult(
            ok=False,
            base_url=None,
            model_id=None,
            checks=[],
            messages=["latest vLLM run record does not include host/port"],
        )

    model_id = _model_from_command(record.command)
    if not model_id:
        return VllmApiSmokeResult(
            ok=False,
            base_url=f"http://{record.host}:{record.port}/v1",
            model_id=None,
            checks=[],
            messages=["latest vLLM run record command does not include a model id"],
        )

    base_url = f"http://{record.host}:{record.port}/v1"
    open_url = opener or request.urlopen
    models_check, served_model_id = _get_models(base_url, open_url, timeout)
    chat_model_id = served_model_id or model_id
    checks = [
        models_check,
        _post_chat_completion(base_url, chat_model_id, open_url, timeout),
    ]
    ok = all(check.ok for check in checks)
    messages = ["vLLM API smoke completed" if ok else "vLLM API smoke failed"]
    return VllmApiSmokeResult(ok=ok, base_url=base_url, model_id=chat_model_id, checks=checks, messages=messages)


def _get_models(base_url: str, opener: Any, timeout: float) -> tuple[VllmApiProbeCheck, str | None]:
    req = request.Request(f"{base_url}/models", method="GET")
    check, payload = _json_request_check("GET /v1/models", req, opener, timeout, required_key="data")
    if not check.ok:
        return check, None
    served_model_id = _first_served_model_id(payload)
    if served_model_id:
        return VllmApiProbeCheck(check.name, True, f"{check.message}; using served model id {served_model_id}", check.status_code), served_model_id
    return check, None


def _post_chat_completion(base_url: str, model_id: str, opener: Any, timeout: float) -> VllmApiProbeCheck:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    check, _payload = _json_request_check("POST /v1/chat/completions", req, opener, timeout, required_key="choices")
    return check


def _json_request_check(name: str, req: request.Request, opener: Any, timeout: float, *, required_key: str) -> tuple[VllmApiProbeCheck, dict[str, Any] | None]:
    try:
        response = opener(req, timeout=timeout)
        status_code = _status_code(response)
        data = response.read()
        if hasattr(response, "close"):
            response.close()
        payload = json.loads(data.decode("utf-8") if isinstance(data, bytes) else str(data))
    except Exception as exc:
        return VllmApiProbeCheck(name=name, ok=False, message=f"{name} failed: {exc}"), None

    if status_code is not None and not (200 <= status_code < 300):
        return VllmApiProbeCheck(name=name, ok=False, message=f"{name} returned HTTP {status_code}", status_code=status_code), None
    if not isinstance(payload, dict) or required_key not in payload:
        return VllmApiProbeCheck(name=name, ok=False, message=f"{name} response missing {required_key}", status_code=status_code), None
    return VllmApiProbeCheck(name=name, ok=True, message=f"{name} passed", status_code=status_code), payload


def _first_served_model_id(payload: dict[str, Any] | None) -> str | None:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict):
            model_id = str(item.get("id") or "").strip()
            if model_id:
                return model_id
    return None


def _status_code(response: Any) -> int | None:
    if hasattr(response, "getcode"):
        code = response.getcode()
        return int(code) if code is not None else None
    code = getattr(response, "status", None)
    return int(code) if code is not None else None


def _model_from_command(command: list[str]) -> str | None:
    try:
        serve_index = command.index("serve")
    except ValueError:
        return None
    if serve_index + 1 >= len(command):
        return None
    return command[serve_index + 1]
