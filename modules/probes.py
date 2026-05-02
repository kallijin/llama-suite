import json
import urllib.error
import urllib.request
from typing import Any


def http_get_json(url: str, timeout: int = 5) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def show_status(cfg: dict[str, Any], running_servers: list[str]) -> None:
    print("\n  ── 서버 상태 ──")
    print("  모델 교체 직후에는 준비까지 30초~5분 이상 걸릴 수 있습니다.")
    print("  모델 크기, ctx-size, VRAM 여유, backend 상태에 따라 시간이 달라집니다.")
    print("  이때 503은 서버 사망이 아니라 loading/busy 상태일 수 있습니다.")
    print()
    print("  기다리는 게 속 터지면 술담배만 쳐드시지 마시고 시스템에 투자하세요.")
    print("  로컬 AI는 결국 VRAM, RAM, 냉각, 스토리지로 굴러갑니다.")

    lines = running_servers
    if not lines:
        print("  실행 중인 llama-server 없음")
    else:
        print(f"  실행 중인 llama-server: {len(lines)}개")
        for line in lines:
            print(f"    {line}")

    base = f"http://{cfg['host']}:{cfg['port']}"
    print(f"\n  health: {base}/health")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            print(f"    {r.read().decode('utf-8', errors='replace').strip()}")
    except Exception as e:
        print(f"    ⚠️  실패: {e}")

    print(f"\n  models: {base}/v1/models")
    try:
        data = http_get_json(f"{base}/v1/models")
        for m in data.get("data", []):
            print(f"    - {m.get('id')}")
    except Exception as e:
        print(f"    ⚠️  실패: {e}")


def get_first_model_id(base: str) -> str | None:
    data = http_get_json(f"{base}/v1/models")
    models = data.get("data", [])
    if not models:
        return None
    return models[0].get("id")


def quick_no_think_test(cfg: dict[str, Any]) -> str:
    """content/reasoning_content 분리 상태를 빠르게 확인."""
    base = f"http://{cfg['host']}:{cfg['port']}"
    model = None
    try:
        model = get_first_model_id(base)
    except Exception as e:
        print(f"  ⚠️  모델 목록 확인 실패: {e}")
        return "unknown"

    if not model:
        print("  ⚠️  모델 ID를 찾지 못했습니다.")
        return "unknown"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "반드시 OK-LLAMA 라고만 출력해."}],
        "max_tokens": 64,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  ⚠️  채팅 테스트 실패: {e}")
        return "unknown"

    msg = result.get("choices", [{}])[0].get("message", {})
    finish = result.get("choices", [{}])[0].get("finish_reason")
    usage = result.get("usage", {})

    print("\n  ── no-thinking 테스트 ──")
    print(f"  model            : {model}")
    print(f"  finish_reason    : {finish}")
    print(f"  content          : {msg.get('content')!r}")
    print(f"  reasoning_content: {msg.get('reasoning_content')!r}")
    print(f"  usage            : {usage}")

    if msg.get("content") and not msg.get("reasoning_content"):
        print("  ✅ 정상: content로 답하고 reasoning_content는 비어 있습니다.")
        return "pass"
    else:
        print("  ⚠️  비정상: thinking/reasoning 출력이 아직 살아 있을 수 있습니다.")
        return "fail"


def build_tools_payload(model: str, tool_choice: str | dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call the report_status tool with status set to ok. Do not write prose.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "report_status",
                    "description": "Report a short readiness status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Short status string.",
                            }
                        },
                        "required": ["status"],
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
        "max_tokens": 128,
        "temperature": 0,
    }


def post_chat_completion(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def classify_tools_result(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

    if tool_calls:
        status = "pass"
    else:
        imitation_markers = ("report_status", "function", "tool", "json")
        status = "text-imitation" if any(marker in content_text.lower() for marker in imitation_markers) else "fail"

    details = {
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": tool_calls,
        "content": content,
        "usage": result.get("usage", {}),
    }
    return status, details


def run_tools_choice_probe(
    base: str,
    model: str,
    label: str,
    tool_choice: str | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    try:
        result = post_chat_completion(base, build_tools_payload(model, tool_choice))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        status = "incompatible" if label == "named_choice" else "unknown"
        return status, {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return "unknown", {"error": str(e)}

    status, details = classify_tools_result(result)
    if label == "named_choice" and status == "text-imitation":
        status = "fail"
    return status, details


def print_tools_choice_result(label: str, status: str, details: dict[str, Any]) -> None:
    print(f"\n  ── tools 테스트: {label} ──")
    print(f"  status       : {status}")
    if "error" in details:
        print(f"  error        : {details['error']}")
        return
    print(f"  finish_reason: {details.get('finish_reason')}")
    print(f"  tool_calls   : {details.get('tool_calls')!r}")
    print(f"  content      : {details.get('content')!r}")
    print(f"  usage        : {details.get('usage')}")


def quick_tools_test(cfg: dict[str, Any]) -> dict[str, str]:
    """tool_choice 방식별로 현재 llama.cpp 조합의 structured tool_calls를 확인."""
    base = f"http://{cfg['host']}:{cfg['port']}"
    statuses = {
        "direct_tools_required": "unknown",
        "direct_tools_auto": "unknown",
        "direct_tools_named_choice": "unknown",
    }

    try:
        model = get_first_model_id(base)
    except Exception as e:
        print(f"  ⚠️  모델 목록 확인 실패: {e}")
        return statuses

    if not model:
        print("  ⚠️  모델 ID를 찾지 못했습니다.")
        return statuses

    print("\n  ── tools 테스트 ──")
    print(f"  model: {model}")

    probes: list[tuple[str, str | dict[str, Any]]] = [
        ("required", "required"),
        ("auto", "auto"),
        ("named_choice", {"type": "function", "function": {"name": "report_status"}}),
    ]

    for label, tool_choice in probes:
        status, details = run_tools_choice_probe(base, model, label, tool_choice)
        statuses[f"direct_tools_{label}"] = status
        print_tools_choice_result(label, status, details)

    return statuses


def print_no_thinking_failure_report() -> None:
    print("\n❌ no-thinking 실패\n")
    print("이 모델은 현재 실행 조합에서 reasoning/thinking 출력이 완전히 분리되거나 억제되지 않는 상태입니다.\n")
    print("이 상태로 에이전트에 투입하면 불필요한 reasoning 토큰이 상위 에이전트 컨텍스트에 전달되어 응답 형성, 도구 선택, 작업 흐름이 불안정해질 수 있습니다.\n")
    print("또한 유료 API 모델, 외부 툴체인, 원격 에이전트와 결합될 경우 불필요한 토큰 전달로 인해 추가 비용이 발생할 수 있습니다.")


def print_tools_success_report() -> None:
    print("\n✅ Tools 호출 확인\n")
    print("현재 llama.cpp 실행 조합에서 tools 능력이 정상적으로 발현되는 상태입니다.\n")
    print("이 모델은 현재 설정 기준으로 structured tool-call 응답을 생성할 수 있으며, 에이전트에서 도구 호출 작업에 사용할 수 있는 상태입니다.")


def print_tools_partial_report() -> None:
    print("\n⚠️ Tools 부분 확인\n")
    print("이 모델은 현재 실행 조합에서 강제 tool-call은 가능하지만, auto tool selection은 확인되지 않은 상태입니다.")
    print("상위 에이전트가 도구 선택을 모델 자율 판단에 맡기면 실제 도구 호출 대신 도구 사용법을 설명하는 텍스트를 생성할 수 있습니다.")


def print_tools_failure_report() -> None:
    print("\n❌ Tools 호출 실패\n")
    print("현재 llama.cpp 실행 조합에서 tools 능력이 정상적으로 발현되지 않는 상태입니다.\n")
    print("이 결과는 원본 모델의 tools 능력이나 품질을 단정하지 않습니다.")
    print("모델 공개 이후의 GGUF 변환, 양자화, 튜닝, chat template, 서버 옵션, tool-call parser, 상위 런처/에이전트 연동 방식 중 하나가 원인일 수 있습니다.\n")
    print("이 상태의 모델은 에이전트 작업에 투입했을 때 런처가 모델 탑재를 거부하거나, 탑재되더라도 실제 도구를 호출하지 않고 도구 실행처럼 보이는 텍스트만 생성할 수 있습니다.\n")
    print("이 판정은 tools/tool-call 동작에 한정됩니다.")
    print("일반적인 질의응답, 추론, 창작, 채팅 품질은 원본 모델 개발사가 공개한 공식 평가를 우선 신뢰합니다.")


def quick_agent_readiness_test(cfg: dict[str, Any]) -> None:
    """no-thinking과 tools 상태를 묶어 에이전트 투입 위험을 빠르게 확인."""
    no_thinking_status = quick_no_think_test(cfg)
    tools_status = quick_tools_test(cfg)
    required_status = tools_status["direct_tools_required"]
    auto_status = tools_status["direct_tools_auto"]
    named_choice_status = tools_status["direct_tools_named_choice"]

    print("\n  ── 에이전트 투입 위험 리포트 ──")

    if no_thinking_status == "fail":
        print_no_thinking_failure_report()

    if required_status == "pass" and auto_status == "pass":
        print_tools_success_report()
    elif required_status == "pass" and auto_status in {"fail", "text-imitation"}:
        print_tools_partial_report()
    elif required_status in {"fail", "text-imitation"}:
        print_tools_failure_report()
    elif named_choice_status == "incompatible":
        print("\n⚠️ named tool_choice 호환성 주의\n")
        print("특정 function 객체를 지정하는 tool_choice 형식이 현재 실행 조합에서 호환되지 않을 수 있습니다.")
        print("상위 agent가 어떤 tool_choice 형식을 쓰는지에 따라 결과가 달라질 수 있습니다.")

    print("\n  ── 판정 항목 ──")
    print(f"  no_thinking              : {no_thinking_status}")
    print(f"  direct_tools_required    : {required_status}")
    print(f"  direct_tools_auto        : {auto_status}")
    print(f"  direct_tools_named_choice: {named_choice_status}")
    print("  upper_agent_tools        : unknown")

    print("\n  ── 종합 판정 ──")
    if no_thinking_status == "unknown" or "unknown" in tools_status.values():
        print("  ⚠️ 판정 불가")
    elif no_thinking_status == "pass" and required_status == "pass" and auto_status == "pass":
        print("  ✅ 에이전트 기본 적합성 확인")
        print("  ✅ direct tools 안정 후보")
    elif required_status == "pass" and auto_status in {"fail", "text-imitation"}:
        print("  ⚠️ 에이전트 투입 주의")
        print("  ⚠️ 강제 호출은 가능하지만 자율 도구 선택은 약함")
    else:
        print("  ⚠️ 에이전트 투입 주의")
