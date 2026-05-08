# Expected Outputs

이 파일은 beginner-first UI의 출력 계약이다.
문구가 바뀔 수는 있지만, 아래 의미는 유지해야 한다.

## Final Run Preview

```text
[1] 최종 실행 명령
<actual llama-server command>

[2] 실행 요약
현재 실행할 모델은 XXXXX.gguf 입니다.
다른 모델을 사용하려면 [M] 모델 변경을 선택하세요.

사용될 endpoint는 http://HOST:PORT/v1 입니다.
주소를 바꾸려면 [A] 설정 변경을 선택하세요.

사용될 주요 파라미터는 다음과 같습니다.
- Context Size: 80000
- KV/기타 추가 파라미터: --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on

저장하려면 [A] 설정 변경 → [W] 현재 설정 저장을 선택하세요.
```

## No Script Exists

```text
저장된 스크립트가 없습니다.
```

## Unified Script Generation

```text
[G] 새 스크립트 생성이 선택되어 있습니다.
현재 설정을 바탕으로 새로운 실행 스크립트가 생성됩니다.
기존 스크립트는 삭제되거나 덮어쓰이지 않습니다.

생성 후 동작을 선택하세요.
[1] 생성만
[2] 생성 후 실행
[R] 작업 화면으로 돌아가기
```

## Script Exists And Matches Profile

```text
기존 스크립트: <name> (modern)
기존 스크립트를 수정하려면 [S] 스크립트 관리 -> [3] 현재 설정으로 불러오기를 사용하세요.
```

## Script Exists But Differs From Profile

```text
기존 스크립트: <name> (modern|old)
스크립트는 실행 스냅샷입니다.
기존 스크립트는 삭제되거나 덮어쓰이지 않습니다.
```

## Script Loaded Into Main Settings

```text
스크립트 설정을 현재 작업 설정으로 불러왔습니다.
기존 스크립트에서 불러온 임시 작업 설정입니다.
스크립트 파일은 수정되지 않았습니다.
```

## Hermes Config Not Registered

```text
Hermes 설정 변경: 비활성화
Hermes 설정 변경: 비활성화 (미등록) - config 경로가 아직 등록되지 않았습니다.
자동 탐지는 후보일 뿐이고, 등록된 경로만 공식 연결 대상입니다.
```

## Hermes Config Registered

```text
Hermes 설정 변경: 활성화 준비됨 (/path/to/config.yaml) - 확인됨: /path/to/config.yaml
실제 수정 기능은 diff/백업/사용자 확인/atomic replace 흐름이 준비된 뒤에만 수행됩니다.
```

## OpenClaw Read-Only Registered

```text
OpenClaw inspection: 활성화 준비됨 (/path/to/config.yaml) - 확인됨: /path/to/config.yaml
OpenClaw는 현재 읽기 전용 inspection만 수행합니다. 위험한 쓰기 작업은 하지 않습니다.
```

## Custom Args Conflict

```text
사용자 추가 파라미터 상태: user_experimental
이 값은 llama-suite가 안정값으로 보장하지 않습니다.
현재 등록된 llama-server 바이너리에서 지원되는지 확인해야 합니다.

충돌: 사용자 추가 파라미터가 구조화 설정과 같은 옵션을 다시 지정합니다.
충돌 옵션: --cache-type-k
구조화 설정을 쓰거나, 실험 옵션에서 중복 옵션을 제거하세요.
```

## Structured Parameter Card

```text
Context Size:
  80000
  출처: 모델 크기 기반 자동 선택
  설명: 모델이 한 번에 다룰 수 있는 대화/문서 길이입니다.
  바꾸려면 [1] 변경을 선택하세요.

KV Cache K:
  q8_0
  출처: llama-suite 안정성 기본값
  설명: 긴 context 사용 시 안정성과 VRAM 사용량에 영향을 주는 KV cache K 설정입니다.
  바꾸려면 [2] 변경을 선택하세요.
```

## Unsaved Working Draft One-Time Run

```text
이번 실행에는 현재 화면에 보이는 임시 설정이 사용됩니다.
이 값들은 아직 저장된 프로필에 반영되지 않았습니다.
프로그램 종료 시 저장되지 않은 변경값은 사라집니다.
저장하려면 [A] 설정 변경 → [W] 현재 설정 저장을 선택하세요.
```

## Script Read-Only View

```text
이 화면은 읽기 전용입니다.
선택한 스크립트 파일은 여기서 직접 수정되지 않습니다.
스크립트의 설정을 바꾸고 싶다면 [3] 현재 설정으로 불러오기를 선택하세요.
```
