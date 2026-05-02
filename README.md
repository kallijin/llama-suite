# llama-suite

**Author:** kalijim  
**Co-author / Assistant:** ChatGPT, kalijin의 아우

[English version](README.en.md)

[Change log](CHANGELOG.md)

---

## 현재 상태

`llama-suite`는 현재 private experimental toolkit이다.

아직 public release가 아니며, 일반 사용자를 위한 설치/배포/호환성 보장을 하지 않는다.
v0.4-scope-freeze 이후에는 새 기능보다 안정화, 검증, 문서 정리를 우선한다.

### Implemented

- 로컬 launcher 기본 흐름
- 모듈 분리 구조
- llama.cpp backend 탐지/검사 계열
- 모델/시스템 정보 확인 메뉴
- 일부 probe와 실행 보조 루틴
- 아이디어 parking lot과 scope freeze 기록

### Planned

- 구현된 기능의 재현성 검증
- README/IDEAS 기준 정리
- root 실행 경고 흐름의 재현 확인
- 모델 프로필과 probe 결과 기록 방식 안정화

### Verification

안정화용 빠른 확인은 repo root에서 다음 명령으로 실행한다.

```sh
bash scripts/smoke-check.sh
```

이 검사는 모델을 실행하지 않고 git 상태, 최근 로그, Python compile, 주요 파일/디렉터리 존재, 주요 모듈 import만 확인한다.

### Parked

- 대용량 모델 디렉터리 자동 탐색
- 네트워크/시스템 설정 자동 추정
- 외부 agent 실행 연동
- 유니코드/다국어 정리
- public release 준비

---

## 목적

`llama-suite`는 로컬 LLM 운용을 위한 작고 투박하지만 강한 도구 상자다.

이 프로젝트는 `llama.cpp`, `Hermes Agent`, `tmux`, `Tailscale`, 로컬 모델 프로파일, 테스트 루틴을 하나의 운용 흐름으로 묶는 것을 목표로 한다.

화려한 UI나 무거운 프레임워크가 목적이 아니다.  
목적은 명확하다.

- 로컬 모델을 안정적으로 띄운다.
- 모델별 실행 조건을 기록한다.
- Hermes와 llama.cpp 연결을 빠르게 검증한다.
- thinking 출력, tool-call, context 크기, VRAM 압박을 테스트한다.
- 실패한 설정과 성공한 설정을 다시 잃어버리지 않는다.
- 모든 것을 가능한 한 로컬에서 통제한다.

---

## 기본 철학

### 1. 디스크 덩치는 허용한다. 런타임 낭비는 허용하지 않는다.

코드가 많아지는 것은 괜찮다.  
모듈이 많아지는 것도 괜찮다.  
설정 파일과 프로파일이 늘어나는 것도 괜찮다.

하지만 실행하자마자 쓸데없이 CPU, RAM, VRAM을 잡아먹는 구조는 피한다.

VRAM은 모델에게 준다.  
CPU는 검색, 로그 분석, 테스트, 정제 작업에 쓴다.  
GUI는 조작판일 뿐, 주인공이 아니다.

---

### 2. Core는 멍청하게, Module은 독립적으로.

복잡한 프레임워크를 만들지 않는다.

Core는 다음 일만 한다.

- 설정을 읽는다.
- 모듈을 호출한다.
- 결과를 보여준다.
- 실패를 기록한다.

각 기능은 독립 모듈로 분리한다.

예상 모듈:

- `model_scan`
- `runner_tmux`
- `probes`
- `hermes_sync`
- `profiles`
- `local_search`
- `clean_search`
- `web_sidecar`
- `mcp_bridge`

안 되는 모듈은 버리면 된다.  
Core는 살아남아야 한다.

---

### 3. CLI는 최후의 복구 경로다.

GUI가 죽어도, 웹 패널이 망가져도, 브라우저가 꼬여도, CLI는 살아 있어야 한다.

기본 조작은 터미널에서 가능해야 한다.

- 모델 선택
- 서버 시작/중지
- tmux 로그 확인
- `/health` 확인
- `/v1/models` 확인
- no-thinking 테스트
- tool-call 테스트
- Hermes config 동기화

GUI는 편의 기능이지 생명줄이 아니다.

---

### 4. XWin GUI는 못생겨도 된다.

화려한 그래픽은 목표가 아니다.

KDE, GNOME, XFCE, 단순 Xorg 환경 어디서든 뜰 수 있는 조작판이면 충분하다.  
옛날 X11 흑백 유틸처럼 못생겨도 된다.

금지 대상:

- Electron
- QtWebEngine
- 무거운 QML/QtQuick
- 불필요한 애니메이션
- 투명/블러 UI
- VRAM을 잡아먹는 장식

필요한 것은 눈호강이 아니라 기능이다.

---

### 5. 모델별 기록은 자산이다.

로컬 모델은 서로 다르게 동작한다.

같은 GGUF라도 다음 요소에 따라 결과가 달라진다.

- llama.cpp 버전
- ROCm/Vulkan/backend
- chat template
- Jinja 처리
- reasoning 옵션
- tool-call parser
- context size
- KV cache
- Hermes request 방식

따라서 모델별로 성공/실패 기록을 남긴다.

예시:

```text
EXAONE 4.0 32B Q4_K_M
- ctx 95000: timeout / wedge 가능
- ctx 92000: direct + Hermes 3회 반복 통과
- no-thinking: pass
- direct tool-call: pass
- Hermes terminal tool: fail / text imitation
- chatml template: fail / <|im_end|> leak
```
