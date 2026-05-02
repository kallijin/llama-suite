#!/usr/bin/env python3
"""
🦙 LLAMA.CPP 모델 실행기 (스크립트 생성 모드)
- 모델/설정 선택 → .sh 스크립트 파일 생성 (~/.hermes/llama-scripts/)
- 스크립트 백그라운드 실행
- 기존 스크립트 목록 표시 + 삭제 관리
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ─── 설정 ──────────────────────────────────────────────
MODELS_DIR = "/mnt/data_main/downloads/models"
LLAMA_SERVER_BIN = "./build/bin/llama-server"  # llama.cpp 디렉토리 기준 상대경로
CONFIG_PATH = os.path.expanduser("~/.hermes/llama-launcher.json")
SCRIPTS_DIR = os.path.expanduser("~/.hermes/llama-scripts")

# ─── 함수들 ────────────────────────────────────────────

def get_running_model():
    """현재 실행 중인 llama-server 의 모델 이름 반환"""
    try:
        result = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "llama-server" in line or "llama_server" in line:
                parts = line.split("-m")
                if len(parts) > 1:
                    model_path = parts[1].strip().split()[0]
                    model_name = Path(model_path).parent.name
                    return model_name
    except Exception:
        pass
    return None


def get_model_list():
    """GGUF 파일들을 찾아 모델 폴더 이름으로 그룹화"""
    models = {}
    if not os.path.isdir(MODELS_DIR):
        return models
    for entry in os.scandir(MODELS_DIR):
        if entry.is_dir():
            for f in os.listdir(entry.path):
                if f.endswith(".gguf"):
                    full_path = os.path.join(entry.path, f)
                    models[entry.name] = full_path
    return dict(sorted(models.items()))


def load_config():
    defaults = {
        "ctx_size": 4096,
        "host": "127.0.0.1",
        "port": 8080,
        "last_model": None,
        "llama_bin": LLAMA_SERVER_BIN,
        "kv_cache_type": "q8_0",
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            defaults.update(saved)
    return defaults


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def print_header():
    print("\n" + "=" * 56)
    print("  🦙  LLAMA.CPP 모델 실행기")
    print("=" * 56)


def change_settings(cfg):
    """설정 변경 메뉴"""
    print("\n  ── 설정 변경 ──")
    print(f"  현재: ctx={cfg['ctx_size']}, kv={cfg.get('kv_cache_type', 'q8_0')}, host={cfg['host']}:{cfg['port']}\n")

    presets = ["2k", "4k", "8k", "16k", "32k", "62k", "95k", "120k"]
    preset_map = {p: int(p.replace("k", "000")) for p in presets}

    print("  Context size 프리셋:")
    for i, p in enumerate(presets, 1):
        marker = " ◀ 현재" if cfg["ctx_size"] == preset_map[p] else ""
        print(f"    [{i}] {p:>4s}{marker}")
    print(f"    [C] 커스텀 입력 (현재: {cfg['ctx_size']})")

    choice = input("  선택 > ").strip().upper()
    if choice == "C":
        val = input(f"  숫자 입력 [{cfg['ctx_size']}] > ").strip()
        if val:
            cfg["ctx_size"] = int(val)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                cfg["ctx_size"] = preset_map[presets[idx]]
        except ValueError:
            pass

    val = input(f"\n  Host [{cfg['host']}] > ").strip()
    if val:
        cfg["host"] = val

    val = input(f"  Port [{cfg['port']}] > ").strip()
    if val:
        cfg["port"] = int(val)

    val = input(f"  llama-server 경로 [{cfg.get('llama_bin', LLAMA_SERVER_BIN)}] > ").strip()
    if val:
        cfg["llama_bin"] = val

    val = input(f"  KV cache type f16/q8_0/q4_0/off [{cfg.get('kv_cache_type', 'q8_0')}] > ").strip().lower()
    if val == "off":
        cfg["kv_cache_type"] = ""
    elif val in {"f16", "q8_0", "q4_0"}:
        cfg["kv_cache_type"] = val

    save_config(cfg)
    print("  ✅ 설정 저장됨!")
    return cfg


def list_scripts():
    """기존 스크립트 목록 반환"""
    scripts = []
    if not os.path.isdir(SCRIPTS_DIR):
        return scripts
    for f in sorted(os.listdir(SCRIPTS_DIR), key=lambda x: os.path.getmtime(os.path.join(SCRIPTS_DIR, x)), reverse=True):
        if f.endswith(".sh"):
            full_path = os.path.join(SCRIPTS_DIR, f)
            scripts.append((f[:-3], full_path))  # (이름, 경로)
    return scripts


def get_latest_script(model_name):
    """해당 모델의 최신 스크립트 파일 반환"""
    if not os.path.isdir(SCRIPTS_DIR):
        return None
    # 모델 이름으로 정렬된 스크립트 목록
    scripts = sorted(
        [f for f in os.listdir(SCRIPTS_DIR) if model_name in f and f.endswith('.sh')],
        key=lambda x: os.path.getmtime(os.path.join(SCRIPTS_DIR, x)),
        reverse=True
    )
    if scripts:
        return os.path.join(SCRIPTS_DIR, scripts[0]), scripts[0]
    return None


def run_existing_script(script_path):
    """기존 스크립트 파일 실행 (wezterm + tmux)"""
    try:
        subprocess.run(
            ["wezterm", "start", "--", "tmux", "new-session", "-d", "bash", script_path],
            capture_output=True, text=True
        )
        print(f"  ✅ 기존 스크립트 실행됨! ({script_path})")
    except FileNotFoundError:
        subprocess.Popen(["bash", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✅ 백그라운드 실행됨 ({script_path})")


def run_script(script_path):
    """스크립트 백그라운드 실행 (wezterm + tmux)"""
    # 기존 서버 종료 확인
    running = get_running_model()
    if running:
        confirm_kill = input(f"  🔴 '{running}' 실행 중. 교체할까요? (y/n) > ").strip().lower()
        if confirm_kill == "y":
            try:
                subprocess.run(["pkill", "-f", "llama-server"], check=False)
                print("  🛑 기존 서버 종료됨")
                time.sleep(2)
            except Exception as e:
                print(f"  ⚠️  종료 실패: {e}")
        else:
            return

    # wezterm에서 실행
    try:
        result = subprocess.run(
            ["wezterm", "start", "--", "tmux", "new-session", "-d", "bash", script_path],
            capture_output=True, text=True
        )
        print(f"  ✅ 스크립트 실행됨! ({script_path})")
    except FileNotFoundError:
        # wezterm 없으면 직접 백그라운드 실행
        subprocess.Popen(["bash", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✅ 백그라운드 실행됨 ({script_path})")


def show_scripts():
    """스크립트 목록 표시"""
    scripts = list_scripts()
    if not scripts:
        print("\n  📜 저장된 스크립트가 없습니다.\n")
        return

    print(f"\n  📜 저장된 스크립트 ({len(scripts)}개)\n")
    for i, (name, path) in enumerate(scripts, 1):
        # 파일에서 모델명 추출
        try:
            with open(path) as f:
                content = f.read()
            model_line = [l for l in content.splitlines() if "MODEL=" in l]
            model_info = model_line[0].split('"')[1] if model_line else "?"
        except Exception:
            model_info = name

        # 실행 상태 확인
        running_marker = ""
        try:
            with open(path) as f:
                for line in f:
                    if "PID=" in line and "$(" not in line:
                        pid_str = line.split("=")[1].strip().rstrip('"')
                        if pid_str.isdigit():
                            os.kill(int(pid_str), 0)
                            running_marker = " 🔴 실행 중"
                        break
        except (OSError, ValueError):
            pass

        print(f"    [{i}] {model_info}{running_marker}")

    print()


def delete_script(index):
    """스크립트 삭제"""
    scripts = list_scripts()
    if 0 <= index < len(scripts):
        name, path = scripts[index]
        # 실행 중이면 먼저 종료
        try:
            with open(path) as f:
                for line in f:
                    if "PID=" in line and "$(" not in line:
                        pid_str = line.split("=")[1].strip().rstrip('"')
                        if pid_str.isdigit():
                            os.kill(int(pid_str), 15)
                            time.sleep(0.5)
                            try:
                                os.kill(int(pid_str), 9)
                            except OSError:
                                pass
                        break
        except (OSError, ValueError):
            pass

        os.remove(path)
        print(f"  🗑️  '{name}' 삭제됨!")


def generate_script(model_name, model_path, cfg):
    """실행 스크립트 생성"""
    os.makedirs(SCRIPTS_DIR, exist_ok=True)

    # 타임스탬프 기반 파일명
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c == '-' else '_' for c in model_name[:40])
    v_suffix = f"-v{model_name[40:60]}" if len(model_name) > 40 else ""
    script_name = f"{safe_name}{v_suffix}_{ts}.sh"
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    # llama-server 절대 경로 확인
    bin_path = cfg.get("llama_bin", LLAMA_SERVER_BIN)
    if not os.path.isabs(bin_path):
        # 현재 디렉토리가 llama.cpp인지 확인
        candidate = os.path.abspath(os.path.join(os.getcwd(), bin_path))
        if os.path.exists(candidate):
            bin_path = candidate
        else:
            # src/llama.cpp 에서 시도
            alt = os.path.expanduser("~/src/llama.cpp/build/bin/llama-server")
            if os.path.exists(alt):
                bin_path = alt

    kv_cache_type = str(cfg.get("kv_cache_type", "q8_0")).strip()
    cache_args = ""
    if kv_cache_type:
        cache_args = f"""    --cache-type-k "$KV_CACHE_TYPE" \\
    --cache-type-v "$KV_CACHE_TYPE" \\
"""

    script_content = f"""#!/bin/bash
# 🦙 LLAMA.CPP 실행 스크립트
# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
MODEL="{model_name}"
PATH="{model_path}"
HOST="{cfg['host']}"
PORT={cfg['port']}
CTX_SIZE={cfg['ctx_size']}
KV_CACHE_TYPE="{kv_cache_type}"

echo "🚀 Starting $MODEL ..."
{bin_path} \\
    -m "$PATH" \\
    --host "$HOST" \\
    --port "$PORT" \\
    --ctx-size "$CTX_SIZE" \\
{cache_args}\
    --jinja

# PID 저장 (종료 시 활용)
echo "PID=$!" > "{script_path}.pid" 2>/dev/null
"""

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_name, script_path


def manage_scripts():
    """스크립트 관리 메뉴"""
    while True:
        show_scripts()
        scripts = list_scripts()

        if not scripts:
            break

        print("  [D] 번호로 삭제")
        print("  [A] 모두 삭제")
        print("  [B] 돌아가기\n")

        choice = input("  선택 > ").strip().upper()

        if choice == "B":
            break

        if choice == "A":
            confirm = input("  정말 모두 삭제할까요? (y/n) > ").strip().lower()
            if confirm == "y":
                for _, path in scripts:
                    try:
                        os.remove(path)
                        pid_file = path + ".pid"
                        if os.path.exists(pid_file):
                            os.remove(pid_file)
                    except OSError:
                        pass
                print("  🗑️  모두 삭제됨!")

        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scripts):
                delete_script(idx)
            else:
                print("  ⚠️  유효하지 않은 번호입니다.")


def main():
    cfg = load_config()
    models = get_model_list()

    if not models:
        print(f"\n⚠️  {MODELS_DIR} 에서 GGUF 파일을 찾을 수 없습니다.")
        sys.exit(1)

    while True:
        print_header()
        print(f"  모델 목록 ({len(models)}개)\n")

        # 실제 실행 중인 모델 감지
        running = get_running_model()
        if running and running in models:
            print(f"  🔴 실행 중: {running}\n")

        numbered = list(enumerate(models.items(), 1))
        for i, (name, path) in numbered:
            marker = ""
            if name == running:
                marker = " ◀ 실행 중"
            elif name == cfg.get("last_model"):
                marker = " ◀ 최근 사용"
            print(f"  [{i}] {name}{marker}")

        # 기존 스크립트 수 표시
        existing_scripts = list_scripts()
        script_info = f" ({len(existing_scripts)}개)" if existing_scripts else ""

        print(f"\n  [A] 설정 변경")
        print(f"  [S] 스크립트 관리{script_info}")
        print(f"  [R] 모델 목록 새로고침")
        print(f"  [Q] 종료\n")

        choice = input("  선택 > ").strip()

        if not choice:
            continue

        # ── 종료 ──
        if choice.upper() == "Q":
            print("\n👋 안녕!\n")
            break

        # ── 설정 변경 ──
        if choice.upper() == "A":
            change_settings(cfg)
            continue

        # ── 스크립트 관리 ──
        if choice.upper() == "S":
            manage_scripts()
            continue

        # ── 새로고침 ──
        if choice.upper() == "R":
            models = get_model_list()
            print("  ✅ 목록 새로고침!")
            input("\n  (계속하려면 Enter)")
            continue

        # ── 번호 선택 ──
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(numbered):
                model_name, model_path = numbered[idx][1]
            else:
                print("\n⚠️  유효하지 않은 번호입니다.")
                input("\n  (계속하려면 Enter)")
                continue
        except ValueError:
            # 이름 검색 (부분 일치)
            matches = [m for m in models if choice.lower() in m.lower()]
            if len(matches) == 1:
                model_name = matches[0]
                model_path = models[model_name]
            elif len(matches) > 1:
                print(f"\n  ⚠️  여러 개 일치:")
                for m in matches[:5]:
                    print(f"    - {m}")
                if len(matches) > 5:
                    print(f"    ... 외 {len(matches) - 5}개")
                input("\n  (계속하려면 Enter)")
                continue
            else:
                print("\n⚠️  일치하는 모델이 없습니다.")
                input("\n  (계속하려면 Enter)")
                continue

        # ── 스크립트 생성 + 실행 ──
        cfg["last_model"] = model_name
        save_config(cfg)

        # 기존 스크립트 확인
        result = get_latest_script(model_name)
        existing_script, existing_name = (result if result else (None, None))

        print(f"\n  📦 모델 : {model_name}")
        print(f"  ⚙️  설정 : ctx={cfg['ctx_size']}, {cfg['host']}:{cfg['port']}")

        if existing_script:
            print(f"  📝 기존 스크립트: {existing_name}")
            choice = input("\n  [E] 기존 스크립트 실행하기 / [N] 새 스크립트 만들기 > ").strip().upper()
            if choice == "E":
                run_existing_script(existing_script)
                input("\n  (계속하려면 Enter)")
                continue
        else:
            choice = "N"

        if choice == "N":
            confirm = input("\n  새 스크립트 생성하고 실행할까요? (y/n) > ").strip().lower()
            if confirm != "y":
                continue

            script_name, script_path = generate_script(model_name, model_path, cfg)
            print(f"  📝 스크립트 생성됨: {script_path}")

            run_confirm = input("  지금 실행할까요? (y/n) > ").strip().lower()
            if run_confirm == "y":
                run_script(script_path)

            input("\n  (계속하려면 Enter)")


if __name__ == "__main__":
    main()
