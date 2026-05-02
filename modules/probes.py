import json
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


def quick_no_think_test(cfg: dict[str, Any]) -> None:
    """content/reasoning_content 분리 상태를 빠르게 확인."""
    base = f"http://{cfg['host']}:{cfg['port']}"
    model = None
    try:
        data = http_get_json(f"{base}/v1/models")
        models = data.get("data", [])
        if models:
            model = models[0].get("id")
    except Exception as e:
        print(f"  ⚠️  모델 목록 확인 실패: {e}")
        return

    if not model:
        print("  ⚠️  모델 ID를 찾지 못했습니다.")
        return

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
        return

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
    else:
        print("  ⚠️  비정상: thinking/reasoning 출력이 아직 살아 있을 수 있습니다.")
