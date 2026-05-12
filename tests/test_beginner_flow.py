import importlib.util
import os
import subprocess
import sys
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from modules.config_store import ensure_kv_cache_safety_args
from modules.script_builder import command_preview, generated_script_name, parse_generated_script


ROOT = Path(__file__).resolve().parents[1]


def load_launcher_module():
    path = ROOT / "llama-launcher-complete.py"
    spec = importlib.util.spec_from_file_location("llama_launcher_complete", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BeginnerFlowTests(unittest.TestCase):
    def sample_cfg(self, server_bin: str) -> dict:
        return {
            "ctx_size": 80000,
            "host": "127.0.0.1",
            "port": 8080,
            "llama_bin": server_bin,
            "jinja": True,
            "alias_by_file": True,
            "reasoning": "off",
            "reasoning_budget": 0,
            "enable_thinking": False,
            "extra_args": ensure_kv_cache_safety_args([]),
            "custom_args": [],
        }

    def test_script_name_contains_snapshot_fields(self) -> None:
        cfg = self.sample_cfg("/bin/echo")

        name = generated_script_name(
            "Huihui-Qwen35B",
            "/models/Huihui-Qwen35B/model.gguf",
            cfg,
            timestamp="20260508_231012",
        )

        self.assertIn("Huihui-Qwen35B", name)
        self.assertIn("__ctx80000__", name)
        self.assertIn("__thinkoff__", name)
        self.assertIn("__20260508_231012__", name)
        self.assertTrue(name.endswith(".sh"))

    def test_short_launcher_wrapper_exists(self) -> None:
        wrapper = ROOT / "llama-suite"

        self.assertTrue(wrapper.is_file())
        self.assertTrue(os.access(wrapper, os.X_OK))

    def test_generated_script_loads_back_into_draft_shape(self) -> None:
        from modules.script_builder import generate_script

        with TemporaryDirectory() as directory:
            cfg = self.sample_cfg("/bin/echo")
            _script_name, script_path = generate_script(
                "Dummy-7B",
                "/models/Dummy-7B/model.gguf",
                cfg,
                scripts_dir=directory,
            )
            snapshot = parse_generated_script(script_path)

        self.assertEqual(snapshot["model_name"], "Dummy-7B")
        self.assertEqual(snapshot["model_path"], "/models/Dummy-7B/model.gguf")
        self.assertEqual(snapshot["cfg"]["ctx_size"], 80000)
        self.assertEqual(snapshot["cfg"]["reasoning"], "off")

    def test_generated_script_preserves_custom_args(self) -> None:
        from modules.script_builder import generate_script

        with TemporaryDirectory() as directory:
            cfg = self.sample_cfg("/bin/echo")
            cfg["custom_args"] = ["--no-warmup"]
            _script_name, script_path = generate_script(
                "Dummy-7B",
                "/models/Dummy-7B/model.gguf",
                cfg,
                scripts_dir=directory,
            )
            snapshot = parse_generated_script(script_path)

        self.assertEqual(snapshot["cfg"]["custom_args"], ["--no-warmup"])
        self.assertIn("--no-warmup", command_preview("Dummy-7B", "/models/Dummy-7B/model.gguf", cfg))

    def test_final_preview_includes_command_and_human_summary(self) -> None:
        launcher = load_launcher_module()
        draft = self.sample_cfg("/bin/echo")
        draft.update(
            {
                "model_name": "Dummy-7B",
                "model_path": "/models/Dummy-7B/model.gguf",
                "dirty": True,
            }
        )

        text = launcher.final_preview_text(draft)

        self.assertIn("[1] 최종 실행 명령", text)
        self.assertIn("[2] 실행 요약", text)
        self.assertIn("현재 실행할 모델은 model.gguf 입니다.", text)
        self.assertIn("이번 실행에는 현재 화면에 보이는 임시 설정이 사용됩니다.", text)
        self.assertIn("[A] 설정 변경 → [W] 현재 설정 저장", text)

    def test_custom_arg_conflict_is_reported_in_preview(self) -> None:
        launcher = load_launcher_module()
        draft = self.sample_cfg("/bin/echo")
        draft.update(
            {
                "model_name": "Dummy-7B",
                "model_path": "/models/Dummy-7B/model.gguf",
                "custom_args": ["--cache-type-k", "tbq3_0"],
            }
        )

        text = launcher.final_preview_text(draft)

        self.assertIn("사용자 추가 파라미터가 구조화 설정과 충돌", text)
        self.assertIn("--cache-type-k", text)
        self.assertIn("[K] 파라미터", text)

    def test_parameter_overview_shows_sources_and_explanations(self) -> None:
        launcher = load_launcher_module()
        draft = self.sample_cfg("/bin/echo")
        draft["param_sources"] = {
            "ctx_size": "모델 크기 기반 자동 선택",
            "cache_type_k": "llama-suite 안정성 기본값",
        }

        from io import StringIO
        import contextlib

        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            launcher.show_parameter_overview(draft)
        text = stdout.getvalue()

        self.assertIn("Context Size:", text)
        self.assertIn("출처: 모델 크기 기반 자동 선택", text)
        self.assertIn("설명: 모델이 한 번에 다룰 수 있는 대화/문서 길이입니다.", text)
        self.assertIn("[1] 변경", text)

    def test_planned_run_summary_shows_selected_runtime_options(self) -> None:
        launcher = load_launcher_module()
        draft = self.sample_cfg("/bin/echo")
        draft.update(
            {
                "model_name": "Dummy-7B",
                "model_path": "/models/Dummy-7B/model.gguf",
                "dirty": True,
                "custom_args": ["--no-warmup"],
            }
        )

        text = "\n".join(launcher.planned_run_summary_lines(draft, "Running-Model"))

        self.assertIn("실행 예정 요약", text)
        self.assertIn("실행 중: Running-Model", text)
        self.assertIn("실행될 모델: Dummy-7B", text)
        self.assertIn("ctx=80000", text)
        self.assertIn("kv-k=q8_0", text)
        self.assertIn("사용자 추가 파라미터: user_experimental", text)
        self.assertIn("현재 설정 저장 상태: 저장 안 됨", text)

    def test_integration_status_requires_registered_path(self) -> None:
        launcher = load_launcher_module()
        cfg = {"registered_paths": {"hermes_config": None, "openclaw_config": None}}

        text = launcher.integration_status_line(cfg, "hermes_config", "Hermes 설정 변경", require_writable=True)

        self.assertIn("Hermes 설정 변경: 비활성화", text)
        self.assertIn("config 경로가 아직 등록되지 않았습니다", text)

    def test_registered_hermes_path_enables_safe_write_readiness(self) -> None:
        launcher = load_launcher_module()
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("endpoint: http://127.0.0.1:8080/v1\n")
            cfg = {"registered_paths": {"hermes_config": str(config_path)}}

            text = launcher.integration_status_line(cfg, "hermes_config", "Hermes 설정 변경", require_writable=True)

        self.assertIn("Hermes 설정 변경: 활성화 준비됨", text)
        self.assertIn("확인됨:", text)

    def test_openclaw_status_is_read_only_inspection(self) -> None:
        launcher = load_launcher_module()
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("model: local\n")
            cfg = {"registered_paths": {"openclaw_config": str(config_path)}}

            text = launcher.integration_status_line(cfg, "openclaw_config", "OpenClaw inspection", require_writable=False)

        self.assertIn("OpenClaw inspection: 활성화 준비됨", text)

    def test_draft_from_config_tolerates_bad_startup_values(self) -> None:
        launcher = load_launcher_module()

        draft = launcher.draft_from_config(
            {
                "ctx_size": "bad",
                "port": "bad",
                "reasoning_budget": "bad",
                "param_sources": ["bad"],
            },
            {},
        )

        self.assertEqual(draft["ctx_size"], 95000)
        self.assertEqual(draft["port"], 8080)
        self.assertEqual(draft["reasoning_budget"], 0)
        self.assertEqual(draft["param_sources"], {})

    def test_launcher_starts_without_models_or_valid_llama_bin(self) -> None:
        with TemporaryDirectory() as home:
            model_dir = Path(home) / "models"
            env = dict(os.environ)
            env["HOME"] = home
            env["LLAMA_MODELS_DIR"] = str(model_dir)
            completed = subprocess.run(
                [sys.executable, "llama-launcher-complete.py"],
                cwd=ROOT,
                input="q\n",
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("llama-suite local AI engine control", completed.stdout)
        self.assertIn("llama.cpp / vLLM 로컬 AI 엔진 관제판", completed.stdout)
        self.assertIn("GGUF 파일을 찾을 수 없습니다", completed.stdout)
        self.assertIn("[L] llama.cpp workspace", completed.stdout)
        self.assertIn("[E] Hermes 등록", completed.stdout)
        self.assertIn("Hermes 설정 변경: 비활성화", completed.stdout)
        self.assertIn("실행 예정 요약", completed.stdout)
        self.assertIn("Recent vLLM run:", completed.stdout)
        self.assertIn("Selected vLLM profile: custom-draft / (empty model) / http://127.0.0.1:8000/v1", completed.stdout)
        self.assertIn("Selected vLLM profile path:", completed.stdout)
        self.assertIn("custom-draft.json", completed.stdout)
        self.assertIn("Backend workflow bridge:", completed.stdout)
        self.assertIn("llama.cpp actions: [L] GGUF model selection / params / preview / run / scripts", completed.stdout)
        self.assertIn("vLLM actions: [V] profile / materials / command preview / preflight / launch / scripts / status / API smoke", completed.stdout)
        self.assertIn("Selected vLLM profile actions are under [V]", completed.stdout)
        self.assertIn("llama.cpp GGUF models: 0 found", completed.stdout)
        self.assertIn("full list: [L] llama.cpp workspace", completed.stdout)
        self.assertIn("[L] llama.cpp workspace", completed.stdout)
        self.assertIn("[V] vLLM workspace", completed.stdout)
        self.assertIn("[S] llama.cpp 스크립트 관리", completed.stdout)
        self.assertNotIn("[K] llama.cpp 파라미터", completed.stdout)
        self.assertNotIn("[P] llama.cpp 최종 미리보기", completed.stdout)
        self.assertNotIn("[O] llama.cpp 1회 실행", completed.stdout)
        self.assertNotIn("[G] llama.cpp 새 스크립트 생성", completed.stdout)
        self.assertNotIn("[B] vLLM profile", completed.stdout)
        self.assertNotIn("[W] vLLM API smoke", completed.stdout)
        self.assertNotIn("[Y] vLLM smoke launch", completed.stdout)
        self.assertNotIn("[Z] vLLM latest run status/log/stop", completed.stdout)
        self.assertNotIn("[V] vLLM doctor", completed.stdout)
        self.assertNotIn("\n  [W] 현재 설정 저장", completed.stdout)
        self.assertNotIn("[X] 새 스크립트 생성 후 실행", completed.stdout)

    def test_main_screen_labels_model_list_as_llama_cpp_gguf(self) -> None:
        with TemporaryDirectory() as home:
            model_dir = Path(home) / "models"
            model_dir.mkdir()
            (model_dir / "tiny.gguf").write_text("not a real model")
            env = dict(os.environ)
            env["HOME"] = home
            env["LLAMA_MODELS_DIR"] = str(model_dir)
            completed = subprocess.run(
                [sys.executable, "llama-launcher-complete.py"],
                cwd=ROOT,
                input="q\n",
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("llama.cpp GGUF models: 1 found", completed.stdout)
        self.assertIn("selected: (none)", completed.stdout)
        self.assertIn("full list: [L] llama.cpp workspace", completed.stdout)
        self.assertNotIn("llama.cpp GGUF model list (1 found)", completed.stdout)
        self.assertNotIn("[ 1] models", completed.stdout)
        self.assertIn("[S] llama.cpp 스크립트 관리", completed.stdout)
        self.assertIn("Selected vLLM profile actions are under [V]", completed.stdout)

    def test_main_backend_submenus_dispatch_to_existing_actions(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from unittest.mock import patch

        stdout = StringIO()
        with patch("builtins.input", side_effect=["1"]), contextlib.redirect_stdout(stdout):
            load_action = launcher.choose_llama_cpp_menu_action({}, {}, None)
        with patch("builtins.input", side_effect=["4"]), contextlib.redirect_stdout(stdout):
            llama_action = launcher.choose_llama_cpp_menu_action()
        with patch("builtins.input", side_effect=["9"]), contextlib.redirect_stdout(stdout):
            vllm_action = launcher.choose_vllm_menu_action()

        self.assertEqual(load_action, "LOAD")
        self.assertEqual(llama_action, "K")
        self.assertEqual(vllm_action, "VLLM_DOCTOR")
        output = stdout.getvalue()
        self.assertIn("llama.cpp workspace", output)
        self.assertIn("GGUF 모델과 llama.cpp 실행 흐름 전용", output)
        self.assertIn("llama.cpp GGUF model list: none found", output)
        self.assertIn("[2] llama.cpp GGUF 모델 변경", output)
        self.assertIn("[4] llama.cpp 파라미터", output)
        self.assertIn("[8] llama.cpp 스크립트 관리", output)
        self.assertIn("vLLM workspace", output)
        self.assertIn("vLLM beta launch path", output)
        self.assertIn("[1] Load Verified Gemma4 Profile", output)
        self.assertIn("[2] Profile Preview / Run Check", output)
        self.assertIn("[3] Start AI Model", output)
        self.assertIn("[4] Server Check / Log / Stop", output)
        self.assertIn("[5] API Connection Test", output)
        self.assertIn("[6] Hermes Config Sync", output)
        self.assertIn("[7] Hermes Chat Test", output)
        self.assertIn("[8] Hermes Tool Test / Raw Markup Check", output)
        self.assertIn("[9] vLLM Start Check", output)
        self.assertIn("[10] Profile Settings", output)
        self.assertIn("[A] Advanced Profile / JSON", output)
        self.assertNotIn("[9] vLLM doctor", output)

    def test_vllm_workspace_stays_open_after_submenu_action(self) -> None:
        with TemporaryDirectory() as home:
            model_dir = Path(home) / "models"
            model_dir.mkdir()
            env = dict(os.environ)
            env["HOME"] = home
            env["LLAMA_MODELS_DIR"] = str(model_dir)
            completed = subprocess.run(
                [sys.executable, "llama-launcher-complete.py"],
                cwd=ROOT,
                input="v\n2\n\nr\nq\n",
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertGreaterEqual(completed.stdout.count("── vLLM workspace ──"), 2)
        self.assertIn("Profile Preview / Run Check", completed.stdout)

    def test_vllm_server_check_menu_explains_latest_suite_server_scope(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecordResult
        from unittest.mock import patch

        stdout = StringIO()
        latest = VllmRunRecordResult(False, None, None, ["no latest run"])

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch("builtins.input", side_effect=["R"]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_manage()

        output = stdout.getvalue()
        self.assertIn("Server Check / Log / Stop", output)
        self.assertIn("Controls only the last vLLM server started by llama-suite.", output)
        self.assertIn("Uses the latest llama-suite vLLM run record.", output)
        self.assertIn("[1] Check Status", output)
        self.assertIn("[2] View Log", output)
        self.assertIn("[3] Stop Server", output)

    def test_vllm_server_check_runs_model_response_check_when_port_ready(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from modules.vllm_api_probe import VllmApiSmokeResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="gemma4",
            run_id="run-ready",
            pid=1234,
            command=["/bin/vllm", "serve", "/models/gemma4"],
            env_preview={},
            log_path="/tmp/ready.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-12T19:00:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "gemma4", "pid": 1234, "run_id": "run-ready", "log_path": "/tmp/ready.log", "alive": True, "log_exists": True, "port_listening": True, "messages": []})())
        mocked_api = Mock(return_value=VllmApiSmokeResult(True, "http://127.0.0.1:8000/v1", "gemma4", [], ["vLLM API smoke completed"]))
        stdout = StringIO()

        with (
            patch.object(launcher, "latest_vllm_run_record", return_value=latest),
            patch.object(launcher, "check_vllm_run_status", mocked_status),
            patch.object(launcher, "run_vllm_api_smoke", mocked_api),
            patch("builtins.input", side_effect=["1", ""]),
            contextlib.redirect_stdout(stdout),
        ):
            launcher.show_vllm_smoke_manage()

        mocked_api.assert_called_once_with(latest_record=latest)
        output = stdout.getvalue()
        self.assertIn("Model response check", output)
        self.assertIn("PASS: model answered through /v1/chat/completions", output)

    def test_vllm_server_check_reports_loading_when_port_not_ready(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="gemma4",
            run_id="run-loading",
            pid=1234,
            command=["/bin/vllm", "serve", "/models/gemma4"],
            env_preview={},
            log_path="/tmp/loading.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-12T19:00:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "gemma4", "pid": 1234, "run_id": "run-loading", "log_path": "/tmp/loading.log", "alive": True, "log_exists": True, "port_listening": False, "messages": []})())
        mocked_api = Mock()
        stdout = StringIO()

        with (
            patch.object(launcher, "latest_vllm_run_record", return_value=latest),
            patch.object(launcher, "check_vllm_run_status", mocked_status),
            patch.object(launcher, "run_vllm_api_smoke", mocked_api),
            patch("builtins.input", side_effect=["1", ""]),
            contextlib.redirect_stdout(stdout),
        ):
            launcher.show_vllm_smoke_manage()

        mocked_api.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("server process is alive, but the API port is not ready yet", output)
        self.assertIn("현재 모델을 VRAM에 탑재 중입니다", output)

    def test_vllm_start_check_screen_uses_easy_english_title(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from unittest.mock import Mock, patch

        stdout = StringIO()
        with (
            patch.object(launcher, "run_vllm_doctor", Mock(return_value=object())),
            patch.object(launcher, "format_vllm_doctor_report", Mock(return_value="wrapper: ok")),
            contextlib.redirect_stdout(stdout),
        ):
            launcher.show_vllm_doctor()

        output = stdout.getvalue()
        self.assertIn("vLLM Start Check", output)
        self.assertIn("Check wrapper, Python, Torch HIP, and ROCm GPU detection before launch.", output)
        self.assertNotIn("── vLLM doctor ──", output)

    def test_main_script_management_is_labeled_llama_cpp_only(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from unittest.mock import Mock, patch

        stdout = StringIO()
        with (
            patch.object(launcher, "list_scripts", Mock(return_value=[("tiny", "/tmp/tiny.sh")])),
            patch.object(launcher, "get_running_servers", Mock(return_value=[])),
            patch.object(launcher, "read_script_field", Mock(side_effect=lambda _path, field: "tiny" if field == "MODEL" else "")),
            patch.object(launcher, "script_is_modern", Mock(return_value=True)),
            contextlib.redirect_stdout(stdout),
        ):
            launcher.show_scripts()

        output = stdout.getvalue()
        self.assertIn("저장된 llama.cpp 스크립트", output)
        self.assertIn("llama.cpp GGUF 실행 스크립트 전용", output)
        self.assertIn("vLLM 스크립트 preview/save는 [V] vLLM workspace", output)

    def test_recent_vllm_run_summary_line_renders_no_record(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_runner import VllmLatestRunSummary

        line = launcher.recent_vllm_run_summary_line(VllmLatestRunSummary(False, None, None, None, "UNKNOWN", ["no record"]))

        self.assertEqual(line, "  Recent vLLM run: no run record")

    def test_recent_vllm_run_startup_warnings_ignore_missing_record(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_runner import VllmLatestRunSummary

        warnings = launcher.recent_vllm_run_startup_warnings(
            VllmLatestRunSummary(False, None, None, None, "UNKNOWN", ["no vLLM run records found under /tmp/runs"])
        )

        self.assertEqual(warnings, [])

    def test_recent_vllm_run_startup_warnings_report_invalid_record(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_runner import VllmLatestRunSummary

        warnings = launcher.recent_vllm_run_startup_warnings(
            VllmLatestRunSummary(False, None, None, None, "UNKNOWN", ["invalid run record schema"])
        )

        self.assertIn("latest vLLM run record could not be loaded cleanly", warnings)
        self.assertIn("invalid run record schema", warnings)

    def test_recent_vllm_run_summary_line_renders_ready_record(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_runner import VllmLatestRunSummary

        line = launcher.recent_vllm_run_summary_line(
            VllmLatestRunSummary(
                True,
                "smoke-qwen-0.5b",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "http://127.0.0.1:8000/v1",
                "READY",
                [],
            )
        )

        self.assertEqual(
            line,
            "  Recent vLLM run: smoke-qwen-0.5b / Qwen/Qwen2.5-0.5B-Instruct / http://127.0.0.1:8000/v1 / READY",
        )

    def test_latest_vllm_run_summary_prefers_profile_snapshot_model(self) -> None:
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult, VllmSmokeStatusResult, latest_vllm_run_summary

        record = VllmRunRecord(
            backend="vllm",
            preset_id="custom-large",
            run_id="vllm-custom-large-20260511-010203",
            pid=123,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "stale-command-model"],
            env_preview={},
            log_path="/tmp/vllm.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-11T01:02:03+09:00",
            status_hint="started",
            profile_snapshot={"model": "/mnt/data_main/downloads/models/local-large-q4-hf"},
        )

        summary = latest_vllm_run_summary(
            latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
            status_check=lambda **kwargs: VllmSmokeStatusResult(
                True,
                kwargs["pid"],
                kwargs["run_id"],
                kwargs["log_path"],
                "custom-large",
                True,
                True,
                True,
                [],
            ),
        )

        self.assertEqual(summary.model, "/mnt/data_main/downloads/models/local-large-q4-hf")
        self.assertEqual(summary.status, "READY")

    def test_selected_vllm_profile_summary_line_renders_current_draft(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmPreflightCheck, VllmPreflightReport, VllmProfile

        line = launcher.selected_vllm_profile_summary_line(
            VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", host="100.64.1.2", port=8010),
            "tailscale-qwen",
        )

        self.assertEqual(
            line,
            "  Selected vLLM profile: tailscale-qwen / Qwen/Qwen2.5-0.5B-Instruct / http://100.64.1.2:8010/v1",
        )
        self.assertIn("tailscale-qwen.json", launcher.selected_vllm_profile_path_line("tailscale-qwen"))

    def test_selected_vllm_profile_summary_uses_served_model_name(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, verified_gemma4_26b_awq_vllm_profile

        line = launcher.selected_vllm_profile_summary_line(
            verified_gemma4_26b_awq_vllm_profile(),
            "verified-gemma4-26b-awq-auto",
        )

        self.assertIn("gemma4-26b-awq-auto", line)
        self.assertIn(VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, line)
        self.assertIn("http://127.0.0.1:8000/v1", line)

    def test_vllm_workspace_shows_selected_profile_and_latest_run(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, verified_gemma4_26b_awq_vllm_profile
        from modules.vllm_runner import VllmLatestRunSummary
        from unittest.mock import patch

        stdout = StringIO()
        with patch("builtins.input", side_effect=["R"]), contextlib.redirect_stdout(stdout):
            action = launcher.choose_vllm_menu_action(
                verified_gemma4_26b_awq_vllm_profile(),
                "gemma4-26b-awq-auto",
                VllmLatestRunSummary(True, "gemma4-26b-awq-auto", VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, "http://127.0.0.1:8000/v1", "READY", []),
            )

        output = stdout.getvalue()
        self.assertEqual(action, "")
        self.assertIn("Selected vLLM profile:", output)
        self.assertIn("Selected vLLM profile path:", output)
        self.assertIn("gemma4-26b-awq-auto.json", output)
        self.assertIn("gemma4-26b-awq-auto", output)
        self.assertIn(VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, output)
        self.assertIn("Recent vLLM run:", output)
        self.assertIn("READY", output)

    def test_vllm_api_smoke_get_models_and_chat_success(self) -> None:
        from modules.vllm_api_probe import run_vllm_api_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class FakeResponse:
            def __init__(self, payload: dict, status: int = 200):
                self.payload = payload
                self.status = status
                self.closed = False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

            def close(self):
                self.closed = True

            def getcode(self):
                return self.status

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        requests = []

        def fake_opener(req, timeout):
            requests.append((req.full_url, req.get_method(), getattr(req, "data", None), timeout))
            if req.full_url.endswith("/models"):
                return FakeResponse({"data": [{"id": "Qwen/Qwen2.5-0.5B-Instruct"}]})
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        result = run_vllm_api_smoke(
            latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
            opener=fake_opener,
            timeout=0.25,
        )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(result.model_id, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual([check.name for check in result.checks], ["GET /v1/models", "POST /v1/chat/completions"])
        self.assertTrue(all(check.ok for check in result.checks))
        self.assertEqual(requests[0][0], "http://127.0.0.1:8000/v1/models")
        self.assertEqual(requests[0][1], "GET")
        self.assertEqual(requests[1][0], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(requests[1][1], "POST")
        self.assertIn(b"Qwen/Qwen2.5-0.5B-Instruct", requests[1][2])
        self.assertIn(b'"max_tokens": 8', requests[1][2])

    def test_vllm_api_smoke_uses_served_model_id_from_models_response(self) -> None:
        from modules.vllm_api_probe import run_vllm_api_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class FakeResponse:
            def __init__(self, payload: dict, status: int = 200):
                self.payload = payload
                self.status = status

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

            def close(self):
                pass

            def getcode(self):
                return self.status

        record = VllmRunRecord(
            backend="vllm",
            preset_id="qwen2.5-14b-awq",
            run_id="run-latest",
            pid=1234,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "/mnt/data_main/downloads/models/Qwen2.5-14B-Instruct-AWQ", "--served-model-name", "qwen2.5-14b-awq"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        requests = []

        def fake_opener(req, timeout):
            requests.append((req.full_url, req.get_method(), getattr(req, "data", None), timeout))
            if req.full_url.endswith("/models"):
                return FakeResponse({"data": [{"id": "qwen2.5-14b-awq"}]})
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        result = run_vllm_api_smoke(
            latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
            opener=fake_opener,
            timeout=0.25,
        )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.model_id, "qwen2.5-14b-awq")
        self.assertIn(b"qwen2.5-14b-awq", requests[1][2])
        self.assertNotIn(b"/mnt/data_main/downloads/models", requests[1][2])

    def test_vllm_api_smoke_default_timeout_allows_larger_local_models(self) -> None:
        from modules.vllm_api_probe import run_vllm_api_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class FakeResponse:
            def __init__(self, payload: dict):
                self.payload = payload
                self.status = 200

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

            def close(self):
                pass

        record = VllmRunRecord(
            backend="vllm",
            preset_id="qwen2.5-14b-awq",
            run_id="run-latest",
            pid=1234,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "/mnt/data_main/downloads/models/Qwen2.5-14B-Instruct-AWQ"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        timeouts = []

        def fake_opener(req, timeout):
            timeouts.append(timeout)
            if req.full_url.endswith("/models"):
                return FakeResponse({"data": [{"id": "qwen2.5-14b-awq"}]})
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        result = run_vllm_api_smoke(
            latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
            opener=fake_opener,
        )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(timeouts, [60.0, 60.0])

    def test_vllm_api_smoke_connection_failure_returns_structured_failure(self) -> None:
        from modules.vllm_api_probe import run_vllm_api_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )

        def failing_opener(_req, timeout):
            raise OSError("connection refused")

        result = run_vllm_api_smoke(
            latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
            opener=failing_opener,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(len(result.checks), 2)
        self.assertTrue(all(not check.ok for check in result.checks))
        self.assertIn("connection refused", "\n".join(check.message for check in result.checks))

    def test_vllm_api_smoke_missing_latest_record_returns_structured_failure(self) -> None:
        from modules.vllm_api_probe import run_vllm_api_smoke
        from modules.vllm_runner import VllmRunRecordResult

        result = run_vllm_api_smoke(latest_record=VllmRunRecordResult(False, None, None, ["no vLLM run records found"]))

        self.assertFalse(result.ok)
        self.assertIsNone(result.base_url)
        self.assertIsNone(result.model_id)
        self.assertEqual(result.checks, [])
        self.assertIn("latest vLLM run record is missing or invalid", "\n".join(result.messages))
        self.assertIn("no vLLM run records found", "\n".join(result.messages))

    def test_hermes_vllm_sync_plan_updates_registered_config_from_ready_latest_run(self) -> None:
        from modules.hermes_integration import build_hermes_vllm_sync_plan
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = True
            port_listening = True
            messages = ["ready"]

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("endpoint: http://127.0.0.1:8080/v1\nmodel: old-model\n")
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="vllm-qwen2.5-14b-awq-test",
                pid=123,
                command=["/home/kalijin/bin/vllm-rocm", "serve", "/models/qwen", "--served-model-name", "served-qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )

            plan = build_hermes_vllm_sync_plan(
                str(config_path),
                latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
                status_check=lambda **kwargs: Status(),
            )

        self.assertTrue(plan.ok, plan.messages)
        self.assertEqual(plan.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(plan.model_id, "served-qwen")
        self.assertIn("endpoint: http://127.0.0.1:8000/v1", plan.updated_text)
        self.assertIn("model: served-qwen", plan.updated_text)
        self.assertIn("context_length: 64000", plan.updated_text)
        self.assertIn("context_length: 64000", "\n".join(plan.messages))

    def test_hermes_vllm_sync_plan_refuses_non_ready_latest_run(self) -> None:
        from modules.hermes_integration import build_hermes_vllm_sync_plan
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = False
            port_listening = False
            messages = ["stopped"]

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("model: old\n")
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="vllm-qwen2.5-14b-awq-test",
                pid=123,
                command=["vllm", "serve", "/models/qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )

            plan = build_hermes_vllm_sync_plan(
                str(config_path),
                latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
                status_check=lambda **kwargs: Status(),
            )

        self.assertFalse(plan.ok)
        self.assertTrue(any("not READY" in message for message in plan.messages))

    def test_hermes_vllm_sync_write_requires_confirmation_and_creates_backup(self) -> None:
        from modules.hermes_integration import HermesVllmSyncPlan, write_hermes_vllm_sync_plan

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("model: old\n")
            plan = HermesVllmSyncPlan(
                ok=True,
                config_path=str(config_path),
                base_url="http://127.0.0.1:8000/v1",
                model_id="served-qwen",
                run_id="run-1",
                original_text="model: old\n",
                updated_text="base_url: http://127.0.0.1:8000/v1\nmodel: served-qwen\n",
                messages=[],
            )

            cancelled = write_hermes_vllm_sync_plan(plan, confirmed=False, timestamp="20260511-030000")
            written = write_hermes_vllm_sync_plan(plan, confirmed=True, timestamp="20260511-030000")

            backup_path = Path(str(config_path) + ".bak.20260511-030000")
            self.assertFalse(cancelled.ok)
            self.assertTrue(written.ok)
            self.assertTrue(backup_path.is_file())
            self.assertEqual(backup_path.read_text(), "model: old\n")
            self.assertIn("model: served-qwen", config_path.read_text())

    def test_hermes_vllm_sync_updates_json_configs(self) -> None:
        from modules.hermes_integration import update_hermes_config_text

        updated = update_hermes_config_text(
            '{"temperature": 0.2}\n',
            base_url="http://127.0.0.1:8000/v1",
            model_id="served-qwen",
            config_path="config.json",
        )
        data = json.loads(updated)
        self.assertEqual(data["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(data["model"], "served-qwen")
        self.assertEqual(data["context_length"], 64000)
        self.assertEqual(data["auxiliary"]["compression"]["context_length"], 64000)
        self.assertEqual(data["custom_providers"][0]["name"], "llama-suite vLLM")
        self.assertEqual(data["custom_providers"][0]["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(data["custom_providers"][0]["model"], "served-qwen")
        self.assertEqual(data["custom_providers"][0]["models"]["served-qwen"]["context_length"], 64000)
        self.assertEqual(data["temperature"], 0.2)

    def test_hermes_vllm_sync_preserves_nested_model_block(self) -> None:
        from modules.hermes_integration import update_hermes_config_text

        original = (
            "model:\n"
            "  api_key: local\n"
            "  base_url: http://127.0.0.1:8080/v1\n"
            "  default: old-default\n"
            "  model: old-model\n"
            "  provider: custom\n"
            "  context_length: 4096\n"
            "  name: old-name\n"
            "providers:\n"
            "  other:\n"
            "    model: should-not-change\n"
            "auxiliary:\n"
            "  compression:\n"
            "    provider: auto\n"
            "    context_length: 2048\n"
            "  vision:\n"
            "    model: vision-model\n"
        )

        updated = update_hermes_config_text(
            original,
            base_url="http://127.0.0.1:8000/v1",
            model_id="served-qwen",
            config_path="config.yaml",
        )

        self.assertIn("model:\n", updated)
        self.assertIn("  base_url: http://127.0.0.1:8000/v1", updated)
        self.assertIn("  default: served-qwen", updated)
        self.assertIn("  model: served-qwen", updated)
        self.assertIn("  context_length: 64000", updated)
        self.assertIn("  name: served-qwen", updated)
        self.assertIn("    model: should-not-change", updated)
        self.assertIn("  compression:\n    provider: auto\n    context_length: 64000", updated)
        self.assertIn("  vision:\n    model: vision-model", updated)
        self.assertIn("custom_providers:", updated)
        self.assertIn("  - name: llama-suite vLLM", updated)
        self.assertIn("    base_url: http://127.0.0.1:8000/v1", updated)
        self.assertIn("    model: served-qwen", updated)
        self.assertIn("      served-qwen:\n        context_length: 64000", updated)

    def test_hermes_vllm_sync_updates_existing_llama_suite_custom_provider(self) -> None:
        from modules.hermes_integration import update_hermes_config_text

        original = (
            "model: old\n"
            "custom_providers:\n"
            "  - name: llama-suite vLLM\n"
            "    base_url: http://127.0.0.1:9000/v1\n"
            "    api_key: local\n"
            "    model: old-vllm\n"
            "    models:\n"
            "      old-vllm:\n"
            "        context_length: 4096\n"
            "  - name: other\n"
            "    base_url: http://127.0.0.1:1234/v1\n"
        )

        updated = update_hermes_config_text(
            original,
            base_url="http://127.0.0.1:8000/v1",
            model_id="served-qwen",
            config_path="config.yaml",
        )

        self.assertEqual(updated.count("name: llama-suite vLLM"), 1)
        self.assertIn("  - name: llama-suite vLLM\n    base_url: http://127.0.0.1:8000/v1", updated)
        self.assertIn("    model: served-qwen", updated)
        self.assertIn("      served-qwen:\n        context_length: 64000", updated)
        self.assertIn("  - name: other\n    base_url: http://127.0.0.1:1234/v1", updated)

    def test_hermes_vllm_sync_preserves_unindented_custom_provider_list_style(self) -> None:
        from modules.hermes_integration import update_hermes_config_text

        original = (
            "model: old\n"
            "custom_providers:\n"
            "- name: existing\n"
            "  base_url: http://127.0.0.1:1234/v1\n"
            "platform_toolsets:\n"
            "  cli:\n"
            "  - terminal\n"
        )

        updated = update_hermes_config_text(
            original,
            base_url="http://127.0.0.1:8000/v1",
            model_id="served-qwen",
            config_path="config.yaml",
        )

        self.assertIn("custom_providers:\n- name: existing", updated)
        self.assertIn("- name: llama-suite vLLM\n  base_url: http://127.0.0.1:8000/v1", updated)
        self.assertIn("  model: served-qwen", updated)
        self.assertIn("platform_toolsets:\n  cli:\n  - terminal", updated)

    def test_hermes_vllm_smoke_plan_uses_ready_latest_run(self) -> None:
        from modules.hermes_runner import build_hermes_vllm_smoke_plan
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = True
            port_listening = True
            messages = ["ready"]

        with TemporaryDirectory() as directory:
            hermes_bin = Path(directory) / "hermes"
            hermes_bin.write_text("#!/usr/bin/env bash\nexit 0\n")
            hermes_bin.chmod(0o755)
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="run-1",
                pid=123,
                command=["vllm", "serve", "/models/qwen", "--served-model-name", "served-qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )

            plan = build_hermes_vllm_smoke_plan(
                hermes_bin=str(hermes_bin),
                latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
                status_check=lambda **kwargs: Status(),
            )

        self.assertTrue(plan.ok, plan.messages)
        self.assertEqual(plan.model_id, "served-qwen")
        self.assertIn("--provider", plan.command)
        self.assertIn("custom", plan.command)
        self.assertIn("--model", plan.command)
        self.assertIn("served-qwen", plan.command)
        self.assertEqual(plan.smoke_kind, "chat")

    def test_hermes_raw_markup_detector_uses_tag_like_patterns(self) -> None:
        from modules.hermes_runner import detect_agent_raw_markup

        leaked = detect_agent_raw_markup("<|tool_call>call:terminal{\"cmd\":\"pwd\"}<tool_call|>")
        self.assertIn("<|tool_call>", leaked)
        self.assertIn("<tool_call|>", leaked)
        self.assertIn("call:terminal", leaked)
        self.assertEqual(detect_agent_raw_markup("I thought about it and answered plainly."), [])

    def test_hermes_vllm_smoke_requires_confirmation_and_checks_marker(self) -> None:
        from modules.hermes_runner import run_hermes_vllm_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = True
            port_listening = True
            messages = ["ready"]

        class Completed:
            returncode = 0
            stdout = "llama-suite-ok\n"
            stderr = ""

        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return Completed()

        with TemporaryDirectory() as directory:
            hermes_bin = Path(directory) / "hermes"
            hermes_bin.write_text("#!/usr/bin/env bash\nexit 0\n")
            hermes_bin.chmod(0o755)
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="run-1",
                pid=123,
                command=["vllm", "serve", "/models/qwen", "--served-model-name", "served-qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )
            latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])

            cancelled = run_hermes_vllm_smoke(
                confirmed=False,
                hermes_bin=str(hermes_bin),
                latest_record=latest,
                status_check=lambda **kwargs: Status(),
                runner=fake_runner,
            )
            result = run_hermes_vllm_smoke(
                confirmed=True,
                hermes_bin=str(hermes_bin),
                latest_record=latest,
                status_check=lambda **kwargs: Status(),
                runner=fake_runner,
                timeout=3,
            )

        self.assertFalse(cancelled.ok)
        self.assertEqual(calls, [(result.command, {"capture_output": True, "text": True, "timeout": 3})])
        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.smoke_kind, "chat")
        self.assertFalse(result.raw_markup_detected)
        self.assertEqual(result.returncode, 0)

    def test_hermes_vllm_chat_smoke_fails_on_raw_markup_leak(self) -> None:
        from modules.hermes_runner import run_hermes_vllm_smoke
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = True
            port_listening = True
            messages = ["ready"]

        class Completed:
            returncode = 0
            stdout = "llama-suite-ok\n<|tool_call>call:terminal{\"cmd\":\"pwd\"}<tool_call|>\n"
            stderr = ""

        def fake_runner(command, **kwargs):
            return Completed()

        with TemporaryDirectory() as directory:
            hermes_bin = Path(directory) / "hermes"
            hermes_bin.write_text("#!/usr/bin/env bash\nexit 0\n")
            hermes_bin.chmod(0o755)
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="run-1",
                pid=123,
                command=["vllm", "serve", "/models/qwen", "--served-model-name", "served-qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )
            result = run_hermes_vllm_smoke(
                confirmed=True,
                hermes_bin=str(hermes_bin),
                latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
                status_check=lambda **kwargs: Status(),
                runner=fake_runner,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "fail")
        self.assertTrue(result.raw_markup_detected)
        self.assertIn("call:terminal", result.raw_markup_patterns or [])
        self.assertTrue(any("raw tool-call markup leaked" in message for message in result.messages))

    def test_hermes_tool_agent_smoke_is_separate_and_unsupported(self) -> None:
        from modules.hermes_runner import build_hermes_vllm_tool_agent_smoke_plan, run_hermes_vllm_tool_agent_smoke

        plan = build_hermes_vllm_tool_agent_smoke_plan()
        result = run_hermes_vllm_tool_agent_smoke(confirmed=True)

        self.assertFalse(plan.ok)
        self.assertEqual(plan.smoke_kind, "tool_agent")
        self.assertFalse(result.ok)
        self.assertEqual(result.smoke_kind, "tool_agent")
        self.assertEqual(result.status, "unsupported")
        self.assertTrue(any("unsupported" in message for message in result.messages))

    def test_hermes_smoke_evidence_redacts_sensitive_text(self) -> None:
        from modules.hermes_smoke_evidence import redact_sensitive_text

        text = (
            "sk-abc123456789 ghp_abc123456789 github_pat_abc123456789 "
            "Bearer abc123 api_key: abc password=abc token: abc "
            "\"secret\": \"abc\" authorization: abc"
        )

        redacted = redact_sensitive_text(text)

        self.assertNotIn("sk-abc123456789", redacted)
        self.assertNotIn("ghp_abc123456789", redacted)
        self.assertNotIn("github_pat_abc123456789", redacted)
        self.assertNotIn("Bearer abc123", redacted)
        self.assertNotIn("api_key: abc", redacted)
        self.assertNotIn("password=abc", redacted)
        self.assertNotIn("token: abc", redacted)
        self.assertNotIn('"secret": "abc"', redacted)
        self.assertNotIn("authorization: abc", redacted)
        self.assertIn("sk-[REDACTED]", redacted)
        self.assertIn("Bearer [REDACTED]", redacted)

    def test_hermes_smoke_evidence_writes_redacted_json_for_raw_markup(self) -> None:
        from modules.hermes_runner import HermesSmokeResult
        from modules.hermes_smoke_evidence import HERMES_SMOKE_EVIDENCE_SCHEMA, write_hermes_smoke_evidence

        with TemporaryDirectory() as directory:
            result = HermesSmokeResult(
                ok=False,
                command=["hermes", "chat", "--token", "ghp_abc123456789"],
                stdout="<|tool_call>call:terminal{}<tool_call|> sk-abc123456789 " + ("x" * 80),
                stderr="password: abc123",
                returncode=0,
                messages=["raw tool-call markup leaked"],
                smoke_kind="chat",
                status="fail",
                raw_markup_detected=True,
                raw_markup_patterns=["<|tool_call>", "call:terminal"],
            )
            written = write_hermes_smoke_evidence(
                result,
                evidence_root=directory,
                timestamp="20260511-213000",
                max_chars=48,
            )

            self.assertTrue(written.ok, written.messages)
            self.assertIsNotNone(written.evidence_path)
            path = Path(str(written.evidence_path))
            self.assertEqual(path.parent, Path(directory))
            self.assertEqual(path.name, "hermes-chat-20260511-213000.json")
            payload = json.loads(path.read_text())

        self.assertEqual(payload["schema"], HERMES_SMOKE_EVIDENCE_SCHEMA)
        self.assertEqual(payload["smoke_kind"], "chat")
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["raw_markup_patterns"], ["<|tool_call>", "call:terminal"])
        self.assertNotIn("ghp_abc123456789", payload["command_excerpt"])
        self.assertNotIn("sk-abc123456789", payload["stdout_excerpt"])
        self.assertNotIn("password: abc123", payload["stderr_excerpt"])
        self.assertLessEqual(len(payload["stdout_excerpt"]), 48)

    def test_hermes_smoke_evidence_skips_clean_success(self) -> None:
        from modules.hermes_runner import HermesSmokeResult
        from modules.hermes_smoke_evidence import write_hermes_smoke_evidence

        with TemporaryDirectory() as directory:
            result = HermesSmokeResult(
                ok=True,
                command=["hermes", "chat"],
                stdout="llama-suite-ok\n",
                stderr="",
                returncode=0,
                messages=["Hermes chat smoke completed"],
                smoke_kind="chat",
                status="pass",
                raw_markup_detected=False,
                raw_markup_patterns=[],
            )
            written = write_hermes_smoke_evidence(result, evidence_root=directory, timestamp="20260511-213000")

            self.assertTrue(written.ok, written.messages)
            self.assertIsNone(written.evidence_path)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(any("not saved" in message for message in written.messages))

    def test_hermes_smoke_evidence_can_force_save_clean_success(self) -> None:
        from modules.hermes_runner import HermesSmokeResult
        from modules.hermes_smoke_evidence import write_hermes_smoke_evidence

        with TemporaryDirectory() as directory:
            result = HermesSmokeResult(
                ok=True,
                command=["direct-vllm-api", "tools=get_weather"],
                stdout="has_tool_calls: True\n",
                stderr="",
                returncode=0,
                messages=["direct vLLM tool-call probe produced structured tool_calls"],
                smoke_kind="direct-vllm-tool",
                status="pass",
                raw_markup_detected=False,
                raw_markup_patterns=[],
            )
            written = write_hermes_smoke_evidence(result, evidence_root=directory, timestamp="20260511-223000", force=True)

            self.assertTrue(written.ok, written.messages)
            self.assertIsNotNone(written.evidence_path)
            payload = json.loads(Path(str(written.evidence_path)).read_text())

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["smoke_kind"], "direct-vllm-tool")
        self.assertIn("has_tool_calls: True", payload["stdout_excerpt"])

    def test_hermes_smoke_evidence_write_failure_is_structured(self) -> None:
        from modules.hermes_runner import HermesSmokeResult
        from modules.hermes_smoke_evidence import write_hermes_smoke_evidence

        with TemporaryDirectory() as directory:
            blocked_root = Path(directory) / "not-a-directory"
            blocked_root.write_text("blocked")
            result = HermesSmokeResult(
                ok=False,
                command=["hermes", "chat"],
                stdout="failure",
                stderr="",
                returncode=1,
                messages=["Hermes chat smoke did not return expected marker"],
                smoke_kind="chat",
                status="fail",
                raw_markup_detected=False,
                raw_markup_patterns=[],
            )
            written = write_hermes_smoke_evidence(result, evidence_root=blocked_root, timestamp="20260511-213000")

        self.assertFalse(written.ok)
        self.assertIsNone(written.evidence_path)
        self.assertTrue(any("save failed" in message for message in written.messages))

    def test_hermes_smoke_evidence_ui_prints_not_saved_path(self) -> None:
        launcher = load_launcher_module()

        class Result:
            ok = True
            evidence_path = None
            messages = ["not saved: smoke passed without raw markup"]

        from io import StringIO
        import contextlib

        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            launcher.print_hermes_smoke_evidence_result(Result())

        self.assertIn("Hermes smoke evidence:", stdout.getvalue())
        self.assertIn("not saved: smoke passed without raw markup", stdout.getvalue())

    def test_hermes_vllm_smoke_refuses_non_ready_run(self) -> None:
        from modules.hermes_runner import build_hermes_vllm_smoke_plan
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult

        class Status:
            alive = False
            port_listening = False
            messages = ["stopped"]

        with TemporaryDirectory() as directory:
            hermes_bin = Path(directory) / "hermes"
            hermes_bin.write_text("#!/usr/bin/env bash\nexit 0\n")
            hermes_bin.chmod(0o755)
            record = VllmRunRecord(
                backend="vllm",
                preset_id="qwen2.5-14b-awq",
                run_id="run-1",
                pid=123,
                command=["vllm", "serve", "/models/qwen"],
                env_preview={},
                log_path=str(Path(directory) / "run.log"),
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-11T03:00:00+09:00",
                status_hint="started",
            )

            plan = build_hermes_vllm_smoke_plan(
                hermes_bin=str(hermes_bin),
                latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []),
                status_check=lambda **kwargs: Status(),
            )

        self.assertFalse(plan.ok)
        self.assertTrue(any("not READY" in message for message in plan.messages))

    def test_vllm_doctor_reports_missing_wrapper_and_python(self) -> None:
        from modules.vllm_doctor import run_vllm_doctor

        with TemporaryDirectory() as directory:
            missing_wrapper = str(Path(directory) / "vllm-rocm")
            missing_python = str(Path(directory) / "python")
            report = run_vllm_doctor(wrapper_path=missing_wrapper, python_path=missing_python, env={})

        summaries = "\n".join(check.summary for check in report.checks)
        self.assertFalse(report.ok)
        self.assertIn("missing:", summaries)

    def test_vllm_doctor_uses_wrapper_for_version_and_python_for_torch_hip(self) -> None:
        from modules.vllm_doctor import format_vllm_doctor_report, run_vllm_doctor

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            python = root / "python"
            marker = root / "python-args.txt"

            wrapper.write_text("#!/usr/bin/env bash\nprintf '0.20.2+rocm721\\n'\n")
            wrapper.chmod(0o755)
            python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" > \"$VLLM_TEST_MARKER\"\n"
                "case \"$LD_LIBRARY_PATH\" in\n"
                "  *rocm721*) ;;\n"
                "  *) exit 7 ;;\n"
                "esac\n"
                "printf '%s\\n' '{\"torch\":\"2.10.0+fake\",\"hip\":\"7.2.53211\",\"cuda_available\":true,\"device_count\":2,\"device0\":\"AMD Radeon RX 9060 XT\"}'\n"
            )
            python.chmod(0o755)

            report = run_vllm_doctor(
                wrapper_path=str(wrapper),
                python_path=str(python),
                env={"VLLM_TEST_MARKER": str(marker)},
            )
            python_args = marker.read_text()

        self.assertTrue(report.ok)
        summaries = "\n".join(check.summary for check in report.checks)
        self.assertIn("0.20.2+rocm721", summaries)
        self.assertIn("hip=7.2.53211", summaries)
        self.assertIn("device0=AMD Radeon RX 9060 XT", summaries)
        self.assertIn("-c", python_args)

        text = format_vllm_doctor_report(report)
        self.assertIn("Hugging Face 모델 ID", text)
        self.assertIn("로컬 Hugging Face/safetensors", text)
        self.assertIn("단일 파일 GGUF", text)
        self.assertIn("llama.cpp 백엔드", text)
        self.assertIn("tokenizer", text)

    def test_vllm_profile_defaults_are_separate_and_conservative(self) -> None:
        from modules.vllm_profiles import default_vllm_profile, validate_vllm_profile

        profile = default_vllm_profile("Qwen/Qwen2.5-0.5B-Instruct")

        self.assertEqual(profile.wrapper_path, "~/bin/vllm-rocm")
        self.assertEqual(profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(profile.host, "127.0.0.1")
        self.assertEqual(profile.port, 8000)
        self.assertEqual(profile.dtype, "auto")
        self.assertEqual(profile.tensor_parallel_size, 1)
        self.assertEqual(profile.kv_cache_dtype, "auto")
        self.assertEqual(profile.max_num_seqs, "")
        self.assertEqual(profile.max_num_batched_tokens, "")
        self.assertEqual(validate_vllm_profile(profile), [])

    def test_vllm_smoke_profile_uses_known_working_model(self) -> None:
        from modules.vllm_profiles import build_vllm_command, smoke_vllm_profile, validate_vllm_profile

        profile = smoke_vllm_profile()
        command, messages = build_vllm_command(profile)

        self.assertEqual(profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(validate_vllm_profile(profile), [])
        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", command)

    def test_vllm_builtin_preset_registry_includes_default_smoke_and_local_large_template(self) -> None:
        from modules.vllm_profiles import (
            VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH,
            VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL,
            build_vllm_command,
            builtin_vllm_profile_presets,
            future_launch_preset_id,
        )

        presets = builtin_vllm_profile_presets()
        by_id = {preset.id: preset for preset in presets}

        self.assertIn("default", by_id)
        self.assertIn("smoke-qwen-0.5b", by_id)
        self.assertIn("template-local-large-q4", by_id)
        self.assertIn("verified-gemma4-26b-awq-auto", by_id)
        self.assertEqual(by_id["default"].label, "Default vLLM profile")
        self.assertEqual(by_id["smoke-qwen-0.5b"].label, "Smoke Qwen 0.5B")
        self.assertIn("read-only", by_id["smoke-qwen-0.5b"].description)
        self.assertEqual(by_id["smoke-qwen-0.5b"].profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(by_id["template-local-large-q4"].label, "Local large Q4 template")
        self.assertIn("Read-only template", by_id["template-local-large-q4"].description)
        self.assertIn("/mnt/data_main/downloads/models/local-large-q4-hf", by_id["template-local-large-q4"].profile.model)
        self.assertEqual(by_id["template-local-large-q4"].profile.max_model_len, "")
        self.assertEqual(by_id["template-local-large-q4"].profile.gpu_memory_utilization, 0.82)
        self.assertEqual(by_id["template-local-large-q4"].profile.max_num_seqs, 1)
        self.assertEqual(by_id["template-local-large-q4"].profile.max_num_batched_tokens, "")
        verified = by_id["verified-gemma4-26b-awq-auto"].profile
        verified_command, verified_messages = build_vllm_command(verified)
        self.assertEqual(verified.model, VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH)
        self.assertEqual(verified.dtype, "bfloat16")
        self.assertEqual(verified.max_model_len, "")
        self.assertEqual(verified.gpu_memory_utilization, 0.88)
        self.assertEqual(verified.tensor_parallel_size, 2)
        self.assertEqual(verified.kv_cache_dtype, "fp8")
        self.assertEqual(verified.max_num_seqs, 1)
        self.assertIn(VERIFIED_GEMMA4_26B_AWQ_SERVED_MODEL, verified.extra_args)
        self.assertEqual(verified_messages, [])
        self.assertIsNotNone(verified_command)
        assert verified_command is not None
        self.assertNotIn("--max-model-len", verified_command)
        self.assertIn("--enable-auto-tool-choice", verified_command)
        self.assertIn("--tool-call-parser", verified_command)
        self.assertIn("gemma4", verified_command)
        self.assertEqual(future_launch_preset_id(), "smoke-qwen-0.5b")

    def test_vllm_profile_validation_reports_structured_messages(self) -> None:
        from modules.vllm_profiles import VllmProfile, validate_vllm_profile

        profile = VllmProfile(
            wrapper_path="",
            model="",
            port=70000,
            gpu_memory_utilization=1.5,
            tensor_parallel_size=0,
            max_model_len=0,
            max_num_seqs=0,
            max_num_batched_tokens=0,
        )

        errors = validate_vllm_profile(profile)

        self.assertIn("wrapper path should not be empty", errors)
        self.assertIn("model should not be empty", errors)
        self.assertIn("port should be 1-65535", errors)
        self.assertIn("gpu_memory_utilization should be between 0 and 1", errors)
        self.assertIn("tensor_parallel_size should be >= 1", errors)
        self.assertIn("max_model_len should be empty/auto or > 0", errors)
        self.assertIn("max_num_seqs should be empty or >= 1", errors)
        self.assertIn("max_num_batched_tokens should be empty or >= 1", errors)

    def test_vllm_profile_validation_handles_non_numeric_user_values(self) -> None:
        from modules.vllm_profiles import VllmProfile, validate_vllm_profile

        profile = VllmProfile(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            port="abc",  # type: ignore[arg-type]
            gpu_memory_utilization="bad",  # type: ignore[arg-type]
            tensor_parallel_size="many",  # type: ignore[arg-type]
            max_model_len="long",  # type: ignore[arg-type]
            max_num_seqs="many",  # type: ignore[arg-type]
            max_num_batched_tokens="many",  # type: ignore[arg-type]
        )

        errors = validate_vllm_profile(profile)

        self.assertIn("port should be 1-65535", errors)
        self.assertIn("gpu_memory_utilization should be between 0 and 1", errors)
        self.assertIn("tensor_parallel_size should be >= 1", errors)
        self.assertIn("max_model_len should be empty/auto or > 0", errors)
        self.assertIn("max_num_seqs should be empty or >= 1", errors)
        self.assertIn("max_num_batched_tokens should be empty or >= 1", errors)

    def test_vllm_profile_from_dict_preserves_extra_args_as_opaque_string(self) -> None:
        from modules.vllm_profiles import vllm_profile_from_dict

        profile = vllm_profile_from_dict(
            {
                "model": "local-model",
                "port": "8001",
                "max_model_len": "8192",
                "gpu_memory_utilization": "0.55",
                "tensor_parallel_size": "2",
                "kv_cache_dtype": "fp8",
                "max_num_seqs": "1",
                "max_num_batched_tokens": "4096",
                "extra_args": "--trust-remote-code --served-model-name local",
            }
        )

        self.assertEqual(profile.port, 8001)
        self.assertEqual(profile.max_model_len, 8192)
        self.assertEqual(profile.gpu_memory_utilization, 0.55)
        self.assertEqual(profile.tensor_parallel_size, 2)
        self.assertEqual(profile.kv_cache_dtype, "fp8")
        self.assertEqual(profile.max_num_seqs, 1)
        self.assertEqual(profile.max_num_batched_tokens, 4096)
        self.assertEqual(profile.extra_args, "--trust-remote-code --served-model-name local")

    def test_vllm_editable_profile_fields_are_vllm_only(self) -> None:
        from modules.vllm_profiles import editable_vllm_profile_field_specs, editable_vllm_profile_fields

        fields = editable_vllm_profile_fields()
        specs = editable_vllm_profile_field_specs()

        self.assertEqual(
            fields,
            [
                "wrapper_path",
                "model",
                "host",
                "port",
                "dtype",
                "max_model_len",
                "gpu_memory_utilization",
                "tensor_parallel_size",
                "kv_cache_dtype",
                "max_num_seqs",
                "max_num_batched_tokens",
                "vllm_cache_root",
                "hf_home",
                "transformers_cache",
                "extra_args",
            ],
        )
        self.assertEqual([spec.name for spec in specs], fields)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["model"].group, "Model")
        self.assertIn("Hugging Face model ID", by_name["model"].help)
        self.assertIn("local model directory", by_name["model"].input_hint)
        self.assertEqual(by_name["model"].example, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(by_name["max_model_len"].group, "Memory")
        self.assertIn("context length", by_name["max_model_len"].help)
        self.assertIn("Leave empty or use auto", by_name["max_model_len"].input_hint)
        self.assertEqual(by_name["max_model_len"].example, "auto")
        self.assertIn("1-65535", by_name["port"].input_hint)
        self.assertIn("per GPU", by_name["gpu_memory_utilization"].help)
        self.assertIn("not to total combined VRAM", by_name["gpu_memory_utilization"].input_hint)
        self.assertEqual(by_name["gpu_memory_utilization"].example, "0.60")
        self.assertIn("AWQ/GPTQ/Int4", by_name["dtype"].input_hint)
        self.assertEqual(by_name["extra_args"].group, "Advanced")
        self.assertNotIn("ctx_size", fields)
        self.assertNotIn("llama_bin", fields)

    def test_vllm_update_profile_field_preserves_invalid_user_input_for_validation(self) -> None:
        from modules.vllm_profiles import VllmProfile, update_vllm_profile_field, validate_vllm_profile

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        updated, messages = update_vllm_profile_field(profile, "port", "not-a-port")

        self.assertEqual(updated.port, "not-a-port")
        self.assertIn("updated vLLM profile field: port", messages)
        self.assertIn("port should be 1-65535", validate_vllm_profile(updated))

    def test_vllm_update_profile_field_rejects_unknown_field(self) -> None:
        from modules.vllm_profiles import VllmProfile, update_vllm_profile_field

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        updated, messages = update_vllm_profile_field(profile, "ctx_size", "8192")

        self.assertIs(updated, profile)
        self.assertIn("unknown vLLM profile field", "\n".join(messages))

    def test_vllm_profile_draft_store_saves_and_loads_separate_schema(self) -> None:
        from modules.vllm_profile_store import load_vllm_profile_draft, save_vllm_profile_draft
        from modules.vllm_profiles import VllmPreflightCheck, VllmPreflightReport, VllmProfile

        with TemporaryDirectory() as directory:
            profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", port="not-a-port")  # type: ignore[arg-type]
            saved = save_vllm_profile_draft(profile, store_root=directory)
            loaded = load_vllm_profile_draft(store_root=directory)
            data = json.loads(Path(saved.profile_path).read_text()) if saved.profile_path else {}

        self.assertTrue(saved.ok, saved.messages)
        self.assertEqual(data["schema"], "llama-suite.vllm-profile.v1")
        self.assertEqual(data["profile"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(data["profile"]["port"], "not-a-port")
        self.assertIn("validation messages", "\n".join(saved.messages))
        self.assertTrue(loaded.ok, loaded.messages)
        self.assertIsNotNone(loaded.profile)
        assert loaded.profile is not None
        self.assertEqual(loaded.profile.port, "not-a-port")
        self.assertIn("loaded draft has validation messages", "\n".join(loaded.messages))

    def test_vllm_profile_draft_json_preview_uses_profile_schema(self) -> None:
        from modules.vllm_profile_store import format_vllm_profile_draft_json
        from modules.vllm_profiles import VllmPreflightCheck, VllmPreflightReport, VllmProfile

        text = format_vllm_profile_draft_json(
            VllmProfile(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                kv_cache_dtype="fp8",
                max_num_seqs=1,
                max_num_batched_tokens=4096,
            ),
            profile_id="qwen-json",
        )
        payload = json.loads(text)

        self.assertEqual(payload["schema"], "llama-suite.vllm-profile.v1")
        self.assertEqual(payload["profile_id"], "qwen-json")
        self.assertEqual(payload["profile"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(payload["profile"]["kv_cache_dtype"], "fp8")
        self.assertEqual(payload["profile"]["max_num_seqs"], 1)
        self.assertEqual(payload["profile"]["max_num_batched_tokens"], 4096)
        self.assertEqual(payload["validation_messages"], [])

    def test_vllm_profile_example_json_matches_schema(self) -> None:
        payload = json.loads((ROOT / "examples" / "vllm-profile.example.json").read_text())

        self.assertEqual(payload["schema"], "llama-suite.vllm-profile.v1")
        self.assertEqual(payload["profile_id"], "example-qwen-smoke")
        self.assertEqual(payload["profile"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("kv_cache_dtype", payload["profile"])
        self.assertIn("max_num_seqs", payload["profile"])
        self.assertIn("max_num_batched_tokens", payload["profile"])

    def test_vllm_local_large_profile_example_json_matches_schema(self) -> None:
        payload = json.loads((ROOT / "examples" / "vllm-profile.local-large.example.json").read_text())

        self.assertEqual(payload["schema"], "llama-suite.vllm-profile.v1")
        self.assertEqual(payload["profile_id"], "example-local-large-q4")
        self.assertEqual(payload["profile"]["model"], "/mnt/data_main/downloads/models/local-large-q4-hf")
        self.assertEqual(payload["profile"]["max_model_len"], "")
        self.assertEqual(payload["profile"]["max_num_seqs"], 1)
        self.assertEqual(payload["profile"]["max_num_batched_tokens"], "")
        self.assertIn("--served-model-name local-large", payload["profile"]["extra_args"])

    def test_vllm_profile_json_file_import_validates_schema_and_profile_id(self) -> None:
        from modules.vllm_profile_store import format_vllm_profile_draft_json, load_vllm_profile_json_file
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            path = Path(directory) / "imported.json"
            path.write_text(
                format_vllm_profile_draft_json(
                    VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", kv_cache_dtype="fp8"),
                    profile_id="imported-qwen",
                )
            )
            result = load_vllm_profile_json_file(path)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.profile_id, "imported-qwen")
        self.assertIsNotNone(result.profile)
        assert result.profile is not None
        self.assertEqual(result.profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(result.profile.kv_cache_dtype, "fp8")
        self.assertIn("loaded", "\n".join(result.messages))

    def test_vllm_profile_json_file_validate_is_read_only(self) -> None:
        from modules.vllm_profile_store import format_vllm_profile_draft_json, validate_vllm_profile_json_file
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-values.json"
            path.write_text(
                format_vllm_profile_draft_json(
                    VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", port="bad"),  # type: ignore[arg-type]
                    profile_id="validate-only",
                )
            )
            result = validate_vllm_profile_json_file(path)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.profile_id, "validate-only")
        self.assertIsNotNone(result.profile)
        self.assertIn("vLLM profile JSON validated", "\n".join(result.messages))
        self.assertIn("loaded draft has validation messages", "\n".join(result.messages))
        self.assertIn("port should be 1-65535", "\n".join(result.messages))

    def test_vllm_profile_json_round_trip_preserves_all_profile_fields(self) -> None:
        from modules.vllm_profile_store import format_vllm_profile_draft_json, load_vllm_profile_json_file, validate_vllm_profile_json_file
        from modules.vllm_profiles import VllmProfile

        profile = VllmProfile(
            wrapper_path="~/bin/vllm-rocm",
            model="/mnt/data_main/downloads/models/local-large-q4",
            host="100.64.1.2",
            port=8010,
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.82,
            tensor_parallel_size=1,
            vllm_cache_root="/mnt/data_main/ai-cache/vllm",
            hf_home="/mnt/data_main/ai-cache/huggingface",
            transformers_cache="/mnt/data_main/ai-cache/huggingface",
            extra_args="--served-model-name local-large",
            kv_cache_dtype="fp8",
            max_num_seqs=2,
            max_num_batched_tokens=8192,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "round-trip.json"
            first_preview = format_vllm_profile_draft_json(profile, profile_id="round-trip")
            path.write_text(first_preview)
            validated = validate_vllm_profile_json_file(path)
            loaded = load_vllm_profile_json_file(path)
            assert loaded.profile is not None
            second_preview = format_vllm_profile_draft_json(loaded.profile, profile_id=loaded.profile_id or "")

        self.assertTrue(validated.ok, validated.messages)
        self.assertTrue(loaded.ok, loaded.messages)
        self.assertEqual(loaded.profile_id, "round-trip")
        self.assertEqual(loaded.profile.to_dict(), profile.to_dict())
        self.assertEqual(json.loads(second_preview), json.loads(first_preview))

    def test_vllm_local_large_template_draft_save_load_validate_and_command_preview(self) -> None:
        from modules.vllm_profile_store import load_vllm_profile_draft, save_vllm_profile_draft
        from modules.vllm_profiles import build_vllm_command, local_large_q4_vllm_profile, validate_vllm_profile

        profile = local_large_q4_vllm_profile()
        profile_id = "draft-from-template-local-large-q4"

        with TemporaryDirectory() as directory:
            saved = save_vllm_profile_draft(profile, profile_id=profile_id, store_root=directory)
            loaded = load_vllm_profile_draft(profile_id=profile_id, store_root=directory)
            assert loaded.profile is not None
            command, messages = build_vllm_command(loaded.profile)

        self.assertTrue(saved.ok, saved.messages)
        self.assertTrue(loaded.ok, loaded.messages)
        self.assertEqual(loaded.profile_id, profile_id)
        self.assertEqual(loaded.profile.to_dict(), profile.to_dict())
        self.assertEqual(validate_vllm_profile(loaded.profile), [])
        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        assert command is not None
        self.assertIn("/mnt/data_main/downloads/models/local-large-q4-hf", command)
        self.assertNotIn("--max-model-len", command)
        self.assertIn("--max-num-seqs", command)
        self.assertIn("1", command)
        self.assertIn("--served-model-name", command)

    def test_vllm_selected_profile_state_saves_and_loads_profile_id(self) -> None:
        from modules.vllm_profile_store import load_selected_vllm_profile_id, save_selected_vllm_profile_id

        with TemporaryDirectory() as directory:
            saved = save_selected_vllm_profile_id("large-q4", store_root=directory)
            loaded = load_selected_vllm_profile_id(store_root=directory)
            payload = json.loads(Path(saved.state_path).read_text())

        self.assertTrue(saved.ok, saved.messages)
        self.assertTrue(loaded.ok, loaded.messages)
        self.assertEqual(loaded.profile_id, "large-q4")
        self.assertEqual(payload["schema"], "llama-suite.vllm-selected-profile.v1")
        self.assertEqual(payload["profile_id"], "large-q4")
        self.assertIn("selected/latest.json", saved.state_path)

    def test_vllm_selected_profile_state_loads_saved_profile_draft(self) -> None:
        from modules.vllm_profile_store import load_selected_vllm_profile_draft, save_selected_vllm_profile_id, save_vllm_profile_draft
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            save_vllm_profile_draft(VllmProfile(model="Local/LargeModel"), profile_id="large-q4", store_root=directory)
            save_selected_vllm_profile_id("large-q4", store_root=directory)
            loaded = load_selected_vllm_profile_draft(store_root=directory)

        self.assertTrue(loaded.ok, loaded.messages)
        self.assertEqual(loaded.profile_id, "large-q4")
        self.assertIsNotNone(loaded.profile)
        assert loaded.profile is not None
        self.assertEqual(loaded.profile.model, "Local/LargeModel")
        self.assertIn("selected vLLM profile loaded: large-q4", "\n".join(loaded.messages))

    def test_vllm_selected_profile_state_does_not_pollute_profile_list(self) -> None:
        from modules.vllm_profile_store import list_vllm_profile_drafts, save_selected_vllm_profile_id, save_vllm_profile_draft
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            save_vllm_profile_draft(VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"), profile_id="smoke", store_root=directory)
            save_selected_vllm_profile_id("smoke", store_root=directory)
            result = list_vllm_profile_drafts(store_root=directory)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual([profile.profile_id for profile in result.profiles], ["smoke"])

    def test_vllm_profile_draft_store_reports_missing_and_invalid_schema(self) -> None:
        from modules.vllm_profile_store import load_vllm_profile_draft

        with TemporaryDirectory() as directory:
            missing = load_vllm_profile_draft(store_root=directory)
            path = Path(directory) / "custom-draft.json"
            path.write_text(json.dumps({"schema": "wrong", "profile": {}}))
            invalid = load_vllm_profile_draft(store_root=directory)

        self.assertFalse(missing.ok)
        self.assertIn("load failed", "\n".join(missing.messages))
        self.assertFalse(invalid.ok)
        self.assertIn("invalid vLLM profile draft schema", "\n".join(invalid.messages))

    def test_vllm_profile_draft_store_lists_named_profiles(self) -> None:
        from modules.vllm_profile_store import list_vllm_profile_drafts, save_vllm_profile_draft
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            save_vllm_profile_draft(VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"), profile_id="smoke", store_root=directory)
            save_vllm_profile_draft(VllmProfile(model="Local/LargeModel", port="bad"), profile_id="large candidate", store_root=directory)  # type: ignore[arg-type]
            result = list_vllm_profile_drafts(store_root=directory)
            missing = list_vllm_profile_drafts(store_root=Path(directory) / "missing")

        self.assertTrue(result.ok, result.messages)
        by_id = {profile.profile_id: profile for profile in result.profiles}
        self.assertIn("smoke", by_id)
        self.assertIn("large candidate", by_id)
        self.assertEqual(by_id["smoke"].model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("port should be 1-65535", by_id["large candidate"].validation_messages)
        self.assertFalse(missing.ok)
        self.assertIn("does not exist", "\n".join(missing.messages))

    def test_vllm_profile_draft_store_deletes_only_with_confirmation(self) -> None:
        from modules.vllm_profile_store import delete_vllm_profile_draft, load_vllm_profile_draft, save_vllm_profile_draft
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            save_vllm_profile_draft(VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"), profile_id="smoke", store_root=directory)
            cancelled = delete_vllm_profile_draft(profile_id="smoke", store_root=directory, confirmed=False)
            still_loaded = load_vllm_profile_draft(profile_id="smoke", store_root=directory)
            deleted = delete_vllm_profile_draft(profile_id="smoke", store_root=directory, confirmed=True)
            missing = load_vllm_profile_draft(profile_id="smoke", store_root=directory)

        self.assertFalse(cancelled.ok)
        self.assertIn("explicit confirmation", "\n".join(cancelled.messages))
        self.assertTrue(still_loaded.ok, still_loaded.messages)
        self.assertTrue(deleted.ok, deleted.messages)
        self.assertFalse(missing.ok)

    def test_vllm_script_preview_builds_shell_script_for_valid_profile(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_script_builder import build_vllm_script_preview

        preview = build_vllm_script_preview(
            VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", extra_args="--served-model-name qwen-smoke"),
            timestamp="2026-05-10 22:30:00",
        )

        self.assertTrue(preview.ok, preview.messages)
        self.assertIn("#!/usr/bin/env bash", preview.script_text)
        self.assertIn("set -euo pipefail", preview.script_text)
        self.assertIn("export VLLM_CACHE_ROOT=/mnt/data_main/ai-cache/vllm", preview.script_text)
        self.assertIn("export HF_HOME=/mnt/data_main/ai-cache/huggingface", preview.script_text)
        self.assertIn("export TRANSFORMERS_CACHE=/mnt/data_main/ai-cache/huggingface", preview.script_text)
        self.assertIn("exec ", preview.script_text)
        self.assertIn("serve Qwen/Qwen2.5-0.5B-Instruct", preview.script_text)
        self.assertIn("--served-model-name qwen-smoke", preview.script_text)

    def test_vllm_script_preview_returns_validation_messages_for_invalid_profile(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_script_builder import build_vllm_script_preview

        preview = build_vllm_script_preview(VllmProfile(model=""))

        self.assertFalse(preview.ok)
        self.assertEqual(preview.script_text, "")
        self.assertIn("model should not be empty", preview.messages)

    def test_vllm_script_save_writes_executable_collision_free_file(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_script_builder import save_vllm_script

        with TemporaryDirectory() as directory:
            first = save_vllm_script(
                VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"),
                script_id="qwen/smoke",
                scripts_dir=directory,
                timestamp="2026-05-10 22:40:00",
            )
            second = save_vllm_script(
                VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"),
                script_id="qwen/smoke",
                scripts_dir=directory,
                timestamp="2026-05-10 22:40:01",
            )
            first_path = Path(first.script_path) if first.script_path else Path()
            second_path = Path(second.script_path) if second.script_path else Path()
            first_exists = first_path.is_file()
            second_exists = second_path.is_file()
            first_executable = bool(first_path.stat().st_mode & 0o111)
            first_text = first_path.read_text()

        self.assertTrue(first.ok, first.messages)
        self.assertTrue(second.ok, second.messages)
        self.assertEqual(first_path.name, "qwen-smoke.sh")
        self.assertEqual(second_path.name, "qwen-smoke__2.sh")
        self.assertTrue(first_exists)
        self.assertTrue(second_exists)
        self.assertTrue(first_executable)
        self.assertIn("exec ", first_text)

    def test_vllm_script_save_refuses_invalid_profile(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_script_builder import save_vllm_script

        with TemporaryDirectory() as directory:
            result = save_vllm_script(VllmProfile(model=""), scripts_dir=directory)

        self.assertFalse(result.ok)
        self.assertIsNone(result.script_path)
        self.assertIn("model should not be empty", result.messages)

    def test_vllm_command_preview_builds_expected_command_list(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")

        command, messages = build_vllm_command(profile)

        self.assertEqual(messages, [])
        self.assertEqual(
            command,
            [
                str(Path("~/bin/vllm-rocm").expanduser()),
                "serve",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--dtype",
                "auto",
                "--gpu-memory-utilization",
                "0.7",
                "--tensor-parallel-size",
                "1",
                "--kv-cache-dtype",
                "auto",
            ],
        )

    def test_vllm_command_preview_omits_auto_max_model_len_but_includes_manual_value(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command, validate_vllm_profile

        auto_command, auto_messages = build_vllm_command(
            VllmProfile(model="local-model", max_model_len="auto")
        )
        manual_command, manual_messages = build_vllm_command(
            VllmProfile(model="local-model", max_model_len=8192)
        )

        self.assertEqual(validate_vllm_profile(VllmProfile(model="local-model", max_model_len="")), [])
        self.assertEqual(validate_vllm_profile(VllmProfile(model="local-model", max_model_len="auto")), [])
        self.assertEqual(auto_messages, [])
        self.assertIsNotNone(auto_command)
        assert auto_command is not None
        self.assertNotIn("--max-model-len", auto_command)
        self.assertEqual(manual_messages, [])
        self.assertIsNotNone(manual_command)
        assert manual_command is not None
        self.assertIn("--max-model-len", manual_command)
        self.assertIn("8192", manual_command)

    def test_vllm_command_preview_adds_optional_memory_tuning_fields_when_set(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        profile = VllmProfile(
            model="local-model",
            kv_cache_dtype="fp8",
            max_num_seqs=1,
            max_num_batched_tokens=4096,
        )

        command, messages = build_vllm_command(profile)

        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        assert command is not None
        self.assertIn("--kv-cache-dtype", command)
        self.assertIn("fp8", command)
        self.assertIn("--max-num-seqs", command)
        self.assertIn("1", command)
        self.assertIn("--max-num-batched-tokens", command)
        self.assertIn("4096", command)

    def test_vllm_command_preview_omits_empty_optional_memory_tuning_fields(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        command, messages = build_vllm_command(
            VllmProfile(
                model="local-model",
                kv_cache_dtype="",
                max_num_seqs="",
                max_num_batched_tokens="",
            )
        )

        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        assert command is not None
        self.assertNotIn("--kv-cache-dtype", command)
        self.assertNotIn("--max-num-seqs", command)
        self.assertNotIn("--max-num-batched-tokens", command)

    def test_vllm_model_source_inspection_reports_hf_local_tokenizer_and_gguf_cases(self) -> None:
        from modules.vllm_profiles import inspect_vllm_model_source, model_source_recovery_guidance_lines

        hf_checks = inspect_vllm_model_source("Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(hf_checks[0].level, "INFO")
        self.assertIn("Hugging Face model ID", hf_checks[0].message)

        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}\n")
            (model_dir / "tokenizer.json").write_text("{}\n")
            (model_dir / "model.safetensors").write_text("")

            checks = inspect_vllm_model_source(str(model_dir))

            self.assertEqual([check.level for check in checks], ["PASS", "PASS", "PASS"])
            self.assertTrue(any(check.name == "tokenizer" and "있음" in check.message for check in checks))

        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}\n")
            (model_dir / "model.safetensors").write_text("")

            checks = inspect_vllm_model_source(str(model_dir))

            self.assertTrue(any(check.name == "tokenizer" and check.level == "FAIL" for check in checks))
            self.assertTrue(any("tokenizer.json / tokenizer.model / tokenizer_config.json: 없음!" in check.message for check in checks))
            guidance = "\n".join(model_source_recovery_guidance_lines(str(model_dir)))
            self.assertIn("Do not invent tokenizer/config files", guidance)
            self.assertIn("Hugging Face:", guidance)
            self.assertIn("ModelScope:", guidance)
            self.assertIn(model_dir.name, guidance)

        gguf_checks = inspect_vllm_model_source("/mnt/data_main/downloads/models/model.gguf")
        self.assertEqual(gguf_checks[0].level, "WARN")
        self.assertIn("vLLM GGUF is experimental", gguf_checks[0].message)

    def test_vllm_profile_report_shows_recovery_guidance_for_missing_model_files(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile, format_vllm_profile_report

        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "missing-tokenizer-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n")
            (model_dir / "model.safetensors").write_text("")

            lines = format_vllm_profile_report(
                "vLLM profile",
                VllmProfile(model=str(model_dir)),
                port_check=lambda host, port: VllmPreflightCheck("port availability", True, "port is available"),
            )

        text = "\n".join(lines)
        self.assertIn("Missing model file recovery guidance:", text)
        self.assertIn("Copy missing files from the same model repo or its base model repo.", text)
        self.assertIn("https://huggingface.co/models?search=missing-tokenizer-model", text)
        self.assertIn("https://modelscope.cn/models?search=missing-tokenizer-model", text)

    def test_vllm_command_preview_expands_wrapper_path_for_execution(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command
        from unittest.mock import patch

        with TemporaryDirectory() as directory, patch.dict(os.environ, {"HOME": directory}):
            command, messages = build_vllm_command(VllmProfile(model="local-model", wrapper_path="~/bin/vllm-rocm"))

        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command[0], str(Path(directory) / "bin" / "vllm-rocm"))
        self.assertNotIn("~", command[0])

    def test_vllm_command_preview_returns_validation_messages_for_invalid_profile(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        command, messages = build_vllm_command(VllmProfile(model=""))

        self.assertIsNone(command)
        self.assertIn("model should not be empty", messages)

    def test_vllm_command_preview_splits_extra_args_with_shlex(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        profile = VllmProfile(
            model="local-model",
            extra_args="--served-model-name 'local alias' --trust-remote-code",
        )

        command, messages = build_vllm_command(profile)

        self.assertEqual(messages, [])
        self.assertIsNotNone(command)
        self.assertEqual(command[-3:], ["--served-model-name", "local alias", "--trust-remote-code"])

    def test_vllm_command_preview_reports_malformed_extra_args(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        command, messages = build_vllm_command(VllmProfile(model="local-model", extra_args="'unterminated"))

        self.assertIsNone(command)
        self.assertTrue(any("extra_args could not be parsed" in message for message in messages))

    def test_vllm_cache_env_preview_lines_are_separate_from_command(self) -> None:
        from modules.vllm_profiles import VllmProfile, cache_env_preview_lines

        lines = cache_env_preview_lines(VllmProfile(model="local-model"))

        self.assertIn("VLLM_CACHE_ROOT=/mnt/data_main/ai-cache/vllm", lines)
        self.assertIn("HF_HOME=/mnt/data_main/ai-cache/huggingface", lines)
        self.assertIn("TRANSFORMERS_CACHE=/mnt/data_main/ai-cache/huggingface", lines)

    def test_vllm_preflight_valid_profile_with_fake_wrapper_and_free_port_passes(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile, run_vllm_preflight

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = VllmProfile(
                wrapper_path=str(wrapper),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                port=54321,
            )

            report = run_vllm_preflight(
                profile,
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
            )

        self.assertTrue(report.ok)
        messages = "\n".join(check.message for check in report.checks)
        self.assertIn("profile values look valid", messages)
        self.assertIn("wrapper executable found", messages)
        self.assertIn("command preview can be built", messages)
        self.assertIn("port", messages)
        self.assertIn("127.0.0.1 = local only", messages)

    def test_vllm_preflight_invalid_profile_reports_validation_messages(self) -> None:
        from modules.vllm_profiles import VllmProfile, run_vllm_preflight

        report = run_vllm_preflight(VllmProfile(model=""))

        self.assertFalse(report.ok)
        messages = "\n".join(check.message for check in report.checks)
        self.assertIn("model should not be empty", messages)

    def test_vllm_preflight_reports_local_model_source_shape(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile, run_vllm_preflight

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n")
            (model_dir / "model.safetensors").write_text("")

            report = run_vllm_preflight(
                VllmProfile(wrapper_path=str(wrapper), model=str(model_dir), port=54324),
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
            )

        self.assertFalse(report.ok)
        checks = {check.name: check for check in report.checks}
        self.assertIn("model source inspection", checks)
        self.assertFalse(checks["model source inspection"].ok)
        self.assertIn("tokenizer.json / tokenizer.model / tokenizer_config.json: 없음!", checks["model source inspection"].message)

    def test_vllm_preflight_used_port_reports_failure(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile, run_vllm_preflight

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            used_port = 54322

            report = run_vllm_preflight(
                VllmProfile(
                    wrapper_path=str(wrapper),
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    port=used_port,
                ),
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    False,
                    f"port {port} is not available on {host}: address already in use",
                ),
            )

        self.assertFalse(report.ok)
        port_checks = [check for check in report.checks if check.name == "port availability"]
        self.assertEqual(len(port_checks), 1)
        self.assertFalse(port_checks[0].ok)
        self.assertIn("not available", port_checks[0].message)

    def test_vllm_preflight_non_executable_wrapper_reports_failure(self) -> None:
        from modules.vllm_profiles import VllmProfile, run_vllm_preflight

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o644)
            report = run_vllm_preflight(
                VllmProfile(
                    wrapper_path=str(wrapper),
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    port=54323,
                )
            )

        self.assertFalse(report.ok)
        messages = "\n".join(check.message for check in report.checks)
        self.assertIn("wrapper path is not executable", messages)

    def test_vllm_host_access_notes_cover_local_tailscale_and_exposed(self) -> None:
        from modules.vllm_profiles import host_access_note

        self.assertEqual(host_access_note("127.0.0.1"), "127.0.0.1 = local only")
        self.assertEqual(host_access_note("100.68.40.87"), "Tailscale IP = private remote access")
        self.assertEqual(host_access_note("0.0.0.0"), "0.0.0.0 = advanced/exposed")

    def test_vllm_host_guidance_mentions_access_modes(self) -> None:
        from modules.vllm_profiles import host_guidance_lines, large_model_guidance_lines, launch_confirmation_guidance_lines

        text = "\n".join(host_guidance_lines())

        self.assertIn("127.0.0.1 = local only", text)
        self.assertIn("Tailscale IP = private remote access", text)
        self.assertIn("0.0.0.0 = advanced/exposed", text)
        confirmation = "\n".join(launch_confirmation_guidance_lines())
        self.assertIn("download model files", confirmation)
        self.assertIn("GPU memory", confirmation)
        self.assertIn("torch compile / graph capture", confirmation)
        self.assertIn("host/port will be bound", confirmation)
        self.assertIn("127.0.0.1 is local-only", confirmation)
        self.assertIn("Tailscale IP is for private remote access", confirmation)
        self.assertIn("0.0.0.0 is advanced/exposed", confirmation)
        self.assertIn("explicit typed confirmation", confirmation)
        large_model = "\n".join(large_model_guidance_lines())
        self.assertIn("/mnt/data_main/downloads/models", large_model)
        self.assertIn("local large vLLM", large_model)
        self.assertIn("HF/safetensors", large_model)
        self.assertIn("AWQ/GPTQ/Int4", large_model)
        self.assertIn("not the same as llama.cpp GGUF", large_model)
        self.assertIn("applied per GPU", large_model)
        self.assertIn("GGUF Q4_K_M", large_model)
        self.assertIn("vLLM GGUF remains experimental", large_model)

    def test_vllm_profile_preview_is_read_only_and_separate(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmPreflightCheck

        text = launcher.vllm_profile_preview_text(
            port_check=lambda host, port: VllmPreflightCheck(
                "port availability",
                True,
                f"port {port} is available on {host}",
            )
        )

        self.assertIn("vLLM profile overview", text)
        self.assertIn("vLLM 전용 profile", text)
        self.assertIn("llama.cpp 파라미터와 별개", text)
        self.assertIn("Built-in smoke launch preset: smoke-qwen-0.5b", text)
        self.assertIn("Custom profile selection/save/load/launch", text)
        self.assertIn("Launch confirmation guidance:", text)
        self.assertIn("download model files", text)
        self.assertIn("GPU memory", text)
        self.assertIn("torch compile / graph capture", text)
        self.assertIn("host/port will be bound", text)
        self.assertIn("Available built-in vLLM profiles:", text)
        self.assertIn("default: Default vLLM profile", text)
        self.assertIn("smoke-qwen-0.5b: Smoke Qwen 0.5B", text)
        self.assertIn("template-local-large-q4: Local large Q4 template", text)
        self.assertIn("verified-gemma4-26b-awq-auto: Verified Gemma4 26B AWQ auto", text)
        self.assertIn("Built-in presets are read-only templates", text)
        self.assertIn("Custom drafts can be copied, edited, saved, and launched separately", text)
        self.assertIn("Preset default: Default vLLM profile", text)
        self.assertIn("Local large Q4 template preset (read-only)", text)
        self.assertIn("/mnt/data_main/downloads/models/local-large-q4-hf", text)
        self.assertIn("HF/safetensors Q4", text)
        self.assertIn("Verified local Gemma4 26B AWQ profile (read-only)", text)
        self.assertIn("vLLM beta launch / API smoke / Hermes plain chat", text)
        self.assertIn("tool-call parser는 Gemma4 전용 parser를 사용합니다", text)
        self.assertIn("Hermes parser 사용 시 raw tool-call markup leak", text)
        self.assertIn("tool-agent coding 완전 합격은 실제 Hermes tool-agent smoke 통과 뒤에만 표시합니다", text)
        self.assertIn("gemma4-26b-awq-auto", text)
        self.assertIn("--enable-auto-tool-choice --tool-call-parser gemma4", text)
        self.assertIn("vLLM-only fields:", text)
        self.assertIn("- wrapper_path: ~/bin/vllm-rocm", text)
        self.assertIn("- host: 127.0.0.1", text)
        self.assertIn("- port: 8000", text)
        self.assertIn("- dtype: auto", text)
        self.assertIn("- tensor_parallel_size: 1", text)
        self.assertIn("Validation messages:", text)
        self.assertIn("- model should not be empty", text)
        self.assertIn("Cache environment preview:", text)
        self.assertIn("VLLM_CACHE_ROOT", text)
        self.assertIn("HF_HOME", text)
        self.assertIn("TRANSFORMERS_CACHE", text)
        self.assertIn("Command preview / dry-run", text)
        self.assertIn("No runnable command preview", text)
        self.assertIn("Launch preflight:", text)
        self.assertIn("[FAIL] profile validation", text)
        self.assertIn("[PASS] host guidance", text)
        self.assertIn("127.0.0.1 = local only", text)
        self.assertIn("Tailscale IP = private remote access", text)
        self.assertIn("0.0.0.0 = advanced/exposed", text)
        self.assertIn("Local / quantized model guidance:", text)
        self.assertIn("/mnt/data_main/downloads/models", text)
        self.assertIn("vLLM GGUF remains experimental", text)

    def test_vllm_profile_preview_includes_read_only_smoke_preset(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmPreflightCheck

        text = launcher.vllm_profile_preview_text(
            port_check=lambda host, port: VllmPreflightCheck(
                "port availability",
                True,
                f"port {port} is available on {host}",
            )
        )

        self.assertIn("Smoke profile preset (read-only)", text)
        self.assertIn("이미 이 시스템에서 성공한 작은 vLLM 확인용 preset", text)
        self.assertIn("Preset smoke-qwen-0.5b: Smoke Qwen 0.5B", text)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000", text)
        self.assertIn("[PASS] profile validation", text)
        self.assertIn("[PASS] command preview", text)

    def test_vllm_custom_profile_text_is_in_memory_and_separate(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile

        text = launcher.vllm_custom_profile_text(
            VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"),
            port_check=lambda host, port: VllmPreflightCheck(
                "port availability",
                True,
                f"port {port} is available on {host}",
            ),
        )

        self.assertIn("vLLM custom profile draft", text)
        self.assertIn("Preview/preflight first", text)
        self.assertIn("llama.cpp 설정과 별개", text)
        self.assertIn("Editable vLLM fields:", text)
        self.assertIn("- Model / model: Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("Hugging Face model ID", text)
        self.assertIn("hint: Use a HF model ID or a local model directory", text)
        self.assertIn("example: Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("- max_model_len: -", text)
        self.assertIn("- Memory / max_model_len: (empty)", text)
        self.assertIn("example: auto", text)
        self.assertIn("- Cache / hf_home: /mnt/data_main/ai-cache/huggingface", text)
        self.assertIn("Command preview / dry-run", text)
        self.assertIn("Model source inspection:", text)
        self.assertIn("Hugging Face model ID; local file inspection skipped", text)
        self.assertIn("Launch preflight:", text)
        self.assertIn("[PASS] profile validation", text)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("Local / quantized model guidance:", text)
        self.assertIn("HF/safetensors", text)

    def test_vllm_profile_menu_can_edit_custom_profile_draft(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile()
        stdout = StringIO()

        with patch("builtins.input", side_effect=["3", "model", "Qwen/Qwen2.5-0.5B-Instruct"]), contextlib.redirect_stdout(stdout):
            updated = launcher.show_vllm_profile_menu(profile)

        self.assertEqual(updated.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("updated vLLM profile field: model", stdout.getvalue())
        self.assertIn("custom profile draft", stdout.getvalue())
        self.assertIn("Model - Hugging Face model ID", stdout.getvalue())
        self.assertIn("selected: Model / model", stdout.getvalue())
        self.assertIn("example: Qwen/Qwen2.5-0.5B-Instruct", stdout.getvalue())
        self.assertIn("new value for model", stdout.getvalue())
        self.assertIn("Memory", stdout.getvalue())

    def test_vllm_profile_menu_can_save_and_load_custom_profile_draft_from_list(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmProfileStoreResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        loaded_profile = VllmProfile(model="loaded-model")
        save_result = VllmProfileStoreResult(True, profile, "/tmp/custom-draft.json", ["saved"])
        list_result = VllmProfileListResult(
            True,
            [VllmStoredProfileInfo("named-profile", "/tmp/named-profile.json", "saved-model", [])],
            "/tmp/profiles",
            [],
        )
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/custom-draft.json", ["loaded"])
        mocked_save = Mock(return_value=save_result)
        mocked_list = Mock(return_value=list_result)
        mocked_load = Mock(return_value=load_result)

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", "named-profile"]), contextlib.redirect_stdout(StringIO()):
            after_save = launcher.show_vllm_profile_menu(profile)
        with (
            patch.object(launcher, "list_vllm_profile_drafts", mocked_list),
            patch.object(launcher, "load_vllm_profile_draft", mocked_load),
            patch("builtins.input", side_effect=["6", "1"]),
            contextlib.redirect_stdout(StringIO()),
        ):
            after_load = launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile, profile_id="named-profile")
        mocked_list.assert_called_once_with()
        mocked_load.assert_called_once_with(profile_id="named-profile")
        self.assertIs(after_save, profile)
        self.assertEqual(after_load.model, "loaded-model")

    def test_vllm_profile_menu_tracks_selected_profile_id(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmProfileStoreResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        loaded_profile = VllmProfile(model="loaded-model")
        save_result = VllmProfileStoreResult(True, profile, "/tmp/large-q4.json", ["saved"])
        list_result = VllmProfileListResult(
            True,
            [VllmStoredProfileInfo("large-q4", "/tmp/large-q4.json", "saved-model", [])],
            "/tmp/profiles",
            [],
        )
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/large-q4.json", ["loaded"])
        mocked_save = Mock(return_value=save_result)
        mocked_list = Mock(return_value=list_result)
        mocked_load = Mock(return_value=load_result)
        stdout = StringIO()

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", "large-q4"]), contextlib.redirect_stdout(stdout):
            saved_profile, saved_id = launcher.show_vllm_profile_menu(profile, "custom-draft", return_profile_id=True)
        with (
            patch.object(launcher, "list_vllm_profile_drafts", mocked_list),
            patch.object(launcher, "load_vllm_profile_draft", mocked_load),
            patch("builtins.input", side_effect=["6", "1"]),
            contextlib.redirect_stdout(stdout),
        ):
            loaded, loaded_id = launcher.show_vllm_profile_menu(profile, saved_id, return_profile_id=True)

        self.assertIs(saved_profile, profile)
        self.assertEqual(saved_id, "large-q4")
        self.assertEqual(loaded.model, "loaded-model")
        self.assertEqual(loaded_id, "large-q4")
        self.assertIn("selected profile: custom-draft", stdout.getvalue())
        self.assertIn("selected profile: large-q4", stdout.getvalue())
        self.assertIn("profile store root:", stdout.getvalue())
        self.assertIn("selected draft JSON path:", stdout.getvalue())
        self.assertIn("large-q4.json", stdout.getvalue())

    def test_vllm_initial_profile_selection_restores_saved_profile(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Local/LargeModel")
        result = VllmProfileStoreResult(True, profile, "/tmp/large-q4.json", ["loaded"], "large-q4")

        with patch.object(launcher, "load_selected_vllm_profile_draft", Mock(return_value=result)):
            loaded_profile, loaded_id = launcher.initial_vllm_profile_selection()

        self.assertIs(loaded_profile, profile)
        self.assertEqual(loaded_id, "large-q4")

    def test_vllm_initial_profile_selection_falls_back_to_default(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profile_store import VllmProfileStoreResult
        from unittest.mock import Mock, patch

        result = VllmProfileStoreResult(False, None, "/tmp/latest.json", ["missing"])

        with patch.object(launcher, "load_selected_vllm_profile_draft", Mock(return_value=result)):
            loaded_profile, loaded_id = launcher.initial_vllm_profile_selection()

        self.assertEqual(loaded_id, "custom-draft")
        self.assertEqual(loaded_profile.model, "")

    def test_vllm_initial_profile_selection_treats_missing_selected_state_as_quiet_default(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profile_store import VllmProfileStoreResult
        from unittest.mock import Mock, patch

        result = VllmProfileStoreResult(False, None, "/tmp/latest.json", ["no selected vLLM profile saved yet"])

        with patch.object(launcher, "load_selected_vllm_profile_draft", Mock(return_value=result)):
            loaded_profile, loaded_id, messages = launcher.initial_vllm_profile_selection_with_messages()

        self.assertEqual(loaded_id, "custom-draft")
        self.assertEqual(loaded_profile.model, "")
        self.assertEqual(messages, [])

    def test_vllm_initial_profile_selection_reports_fallback_messages(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profile_store import VllmProfileStoreResult
        from unittest.mock import Mock, patch

        result = VllmProfileStoreResult(False, None, "/tmp/latest.json", ["invalid selected vLLM profile schema"])

        with patch.object(launcher, "load_selected_vllm_profile_draft", Mock(return_value=result)):
            loaded_profile, loaded_id, messages = launcher.initial_vllm_profile_selection_with_messages()

        self.assertEqual(loaded_id, "custom-draft")
        self.assertEqual(loaded_profile.model, "")
        self.assertIn("using custom-draft defaults", "\n".join(messages))
        self.assertIn("invalid selected vLLM profile schema", "\n".join(messages))

    def test_startup_warnings_render_nonfatal_messages(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib

        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            launcher.print_startup_warnings(["selected profile failed"])

        self.assertIn("Startup warnings:", stdout.getvalue())
        self.assertIn("selected profile failed", stdout.getvalue())

    def test_vllm_profile_menu_groups_actions_by_responsibility(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        stdout = StringIO()

        with patch("builtins.input", side_effect=[""]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_profile_menu(VllmProfile())

        output = stdout.getvalue()
        self.assertIn("vLLM selected profile workspace", output)
        self.assertIn("Run / verify selected profile", output)
        self.assertIn("Choose / import profile", output)
        self.assertIn("Edit selected profile", output)
        self.assertIn("Reference", output)
        self.assertIn("[1] built-in profile preview", output)
        self.assertIn("[2] selected profile preview / dry-run / preflight", output)
        self.assertIn("[10] save selected profile script", output)
        self.assertIn("[11] launch selected vLLM profile", output)

    def test_vllm_profile_menu_save_uses_default_profile_id_and_no_duplicate_load_menu(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        store_result = VllmProfileStoreResult(True, profile, "/tmp/custom-draft.json", ["ok"])
        mocked_save = Mock(return_value=store_result)
        stdout = StringIO()

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", ""]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile, profile_id="custom-draft")
        self.assertNotIn("[5] load custom profile draft", stdout.getvalue())
        self.assertIn("[6] load saved custom profile from list", stdout.getvalue())
        self.assertIn("[8] profile JSON import/validate/preview", stdout.getvalue())
        self.assertIn("[11] launch selected vLLM profile", stdout.getvalue())

    def test_vllm_profile_menu_can_list_saved_custom_profiles(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        list_result = VllmProfileListResult(
            True,
            [
                VllmStoredProfileInfo("smoke", "/tmp/smoke.json", "Qwen/Qwen2.5-0.5B-Instruct", []),
                VllmStoredProfileInfo("large", "/tmp/large.json", "Local/LargeModel", ["port should be 1-65535"]),
            ],
            "/tmp/profiles",
            [],
        )
        mocked_list = Mock(return_value=list_result)
        stdout = StringIO()

        with patch.object(launcher, "list_vllm_profile_drafts", mocked_list), patch("builtins.input", side_effect=["5"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile, "smoke")

        mocked_list.assert_called_once_with()
        self.assertIs(result, profile)
        output = stdout.getvalue()
        self.assertIn("vLLM saved custom profiles", output)
        self.assertIn("[1] smoke: Qwen/Qwen2.5-0.5B-Instruct [valid] *selected*", output)
        self.assertIn("[2] large: Local/LargeModel [needs attention]", output)
        self.assertIn("validation: port should be 1-65535", output)

    def test_vllm_profile_menu_can_load_saved_custom_profile_from_list(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmProfileStoreResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="current-model")
        loaded_profile = VllmProfile(model="loaded-model")
        list_result = VllmProfileListResult(
            True,
            [
                VllmStoredProfileInfo("smoke", "/tmp/smoke.json", "Qwen/Qwen2.5-0.5B-Instruct", []),
                VllmStoredProfileInfo("large-q4", "/tmp/large-q4.json", "Local/LargeModel", []),
            ],
            "/tmp/profiles",
            [],
        )
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/large-q4.json", ["loaded"])
        mocked_list = Mock(return_value=list_result)
        mocked_load = Mock(return_value=load_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", mocked_list),
            patch.object(launcher, "load_vllm_profile_draft", mocked_load),
            patch("builtins.input", side_effect=["6", "2"]),
            contextlib.redirect_stdout(stdout),
        ):
            loaded, loaded_id = launcher.show_vllm_profile_menu(profile, "custom-draft", return_profile_id=True)

        mocked_list.assert_called_once_with()
        mocked_load.assert_called_once_with(profile_id="large-q4")
        self.assertEqual(loaded.model, "loaded-model")
        self.assertEqual(loaded_id, "large-q4")
        self.assertIn("[6] load saved custom profile from list", stdout.getvalue())
        self.assertIn("[1] smoke: Qwen/Qwen2.5-0.5B-Instruct [valid]", stdout.getvalue())

    def test_vllm_profile_menu_load_from_list_cancel_keeps_current_profile(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="current-model")
        list_result = VllmProfileListResult(
            True,
            [VllmStoredProfileInfo("smoke", "/tmp/smoke.json", "Qwen/Qwen2.5-0.5B-Instruct", [])],
            "/tmp/profiles",
            [],
        )
        mocked_load = Mock()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", Mock(return_value=list_result)),
            patch.object(launcher, "load_vllm_profile_draft", mocked_load),
            patch("builtins.input", side_effect=["6", "bad"]),
            contextlib.redirect_stdout(StringIO()),
        ):
            loaded, loaded_id = launcher.show_vllm_profile_menu(profile, "custom-draft", return_profile_id=True)

        mocked_load.assert_not_called()
        self.assertIs(loaded, profile)
        self.assertEqual(loaded_id, "custom-draft")

    def test_vllm_profile_menu_can_delete_saved_custom_profile_from_list(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmProfileStoreResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="current-model")
        list_result = VllmProfileListResult(
            True,
            [VllmStoredProfileInfo("large-q4", "/tmp/large-q4.json", "Local/LargeModel", [])],
            "/tmp/profiles",
            [],
        )
        delete_result = VllmProfileStoreResult(True, None, "/tmp/large-q4.json", ["deleted"])
        mocked_delete = Mock(return_value=delete_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", Mock(return_value=list_result)),
            patch.object(launcher, "delete_vllm_profile_draft", mocked_delete),
            patch("builtins.input", side_effect=["7", "1", "delete"]),
            contextlib.redirect_stdout(stdout),
        ):
            returned_profile, returned_id = launcher.show_vllm_profile_menu(profile, "large-q4", return_profile_id=True)

        mocked_delete.assert_called_once_with(profile_id="large-q4", confirmed=True)
        self.assertIs(returned_profile, profile)
        self.assertEqual(returned_id, "custom-draft")
        self.assertIn("[7] delete saved custom profile from list", stdout.getvalue())
        self.assertIn("[1] large-q4: Local/LargeModel [valid] *selected*", stdout.getvalue())

    def test_vllm_profile_menu_can_preview_profile_json(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["8", "1"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile, "json-profile")

        self.assertIs(result, profile)
        output = stdout.getvalue()
        self.assertIn("[8] profile JSON import/validate/preview", output)
        self.assertIn("selected draft JSON path:", output)
        self.assertIn("examples/vllm-profile.example.json", output)
        self.assertIn("examples/vllm-profile.local-large.example.json", output)
        self.assertIn("does not launch an editor", output)
        self.assertIn("vLLM profile JSON preview", output)
        self.assertIn('"schema": "llama-suite.vllm-profile.v1"', output)
        self.assertIn('"profile_id": "json-profile"', output)
        self.assertIn('"model": "Qwen/Qwen2.5-0.5B-Instruct"', output)

    def test_vllm_profile_menu_can_import_profile_json_file(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        current = VllmProfile(model="current-model")
        imported = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        load_result = VllmProfileStoreResult(True, imported, "/tmp/import.json", ["loaded"], "imported-qwen")
        mocked_load = Mock(return_value=load_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "load_vllm_profile_json_file", mocked_load),
            patch("builtins.input", side_effect=["8", "2", "/tmp/import.json"]),
            contextlib.redirect_stdout(stdout),
        ):
            profile, profile_id = launcher.show_vllm_profile_menu(current, "custom-draft", return_profile_id=True)

        mocked_load.assert_called_once_with("/tmp/import.json")
        self.assertIs(profile, imported)
        self.assertEqual(profile_id, "imported-qwen")
        self.assertIn("[8] profile JSON import/validate/preview", stdout.getvalue())
        self.assertIn("vLLM profile draft store", stdout.getvalue())

    def test_vllm_profile_menu_can_validate_profile_json_without_importing(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        current = VllmProfile(model="current-model")
        checked = VllmProfile(model="checked-model")
        validate_result = VllmProfileStoreResult(True, checked, "/tmp/checked.json", ["validated"], "checked-profile")
        mocked_validate = Mock(return_value=validate_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "validate_vllm_profile_json_file", mocked_validate),
            patch("builtins.input", side_effect=["8", "3", "/tmp/checked.json"]),
            contextlib.redirect_stdout(stdout),
        ):
            profile, profile_id = launcher.show_vllm_profile_menu(current, "current-profile", return_profile_id=True)

        mocked_validate.assert_called_once_with("/tmp/checked.json")
        self.assertIs(profile, current)
        self.assertEqual(profile_id, "current-profile")
        self.assertIn("[8] profile JSON import/validate/preview", stdout.getvalue())
        self.assertIn("vLLM profile draft store", stdout.getvalue())

    def test_vllm_profile_menu_can_copy_local_large_template_to_custom_draft(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        save_mock = Mock()
        launch_mock = Mock()
        stdout = StringIO()

        with (
            patch.object(launcher, "save_vllm_profile_draft", save_mock),
            patch.object(launcher, "launch_vllm_profile_once", launch_mock),
            patch("builtins.input", side_effect=["8", "4", "3"]),
            contextlib.redirect_stdout(stdout),
        ):
            profile, profile_id = launcher.show_vllm_profile_menu(VllmProfile(), "custom-draft", return_profile_id=True)

        save_mock.assert_not_called()
        launch_mock.assert_not_called()
        self.assertEqual(profile_id, "draft-from-template-local-large-q4")
        self.assertIn("/mnt/data_main/downloads/models/local-large-q4-hf", profile.model)
        self.assertEqual(profile.max_model_len, "")
        self.assertEqual(profile.max_num_seqs, 1)
        self.assertIn("[8] profile JSON import/validate/preview", stdout.getvalue())
        self.assertIn("copied built-in preset to in-memory custom draft", stdout.getvalue())
        self.assertIn("next [4] save default profile id: draft-from-template-local-large-q4", stdout.getvalue())
        self.assertIn("draft-from-template-local-large-q4.json", stdout.getvalue())
        self.assertIn("저장/launch는 하지 않았습니다", stdout.getvalue())

    def test_vllm_profile_menu_can_copy_verified_gemma4_profile_to_custom_draft(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH, VllmProfile
        from unittest.mock import Mock, patch

        stdout = StringIO()

        with (
            patch.object(launcher, "save_vllm_profile_draft", Mock()),
            patch.object(launcher, "launch_vllm_profile_once", Mock()),
            patch("builtins.input", side_effect=["8", "4", "4"]),
            contextlib.redirect_stdout(stdout),
        ):
            profile, profile_id = launcher.show_vllm_profile_menu(VllmProfile(), "custom-draft", return_profile_id=True)

        self.assertEqual(profile_id, "draft-from-verified-gemma4-26b-awq-auto")
        self.assertEqual(profile.model, VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH)
        self.assertEqual(profile.dtype, "bfloat16")
        self.assertEqual(profile.tensor_parallel_size, 2)
        self.assertEqual(profile.max_model_len, "")
        self.assertIn("--tool-call-parser gemma4", profile.extra_args)
        self.assertIn("Verified Gemma4 26B AWQ auto", stdout.getvalue())
        self.assertIn("draft-from-verified-gemma4-26b-awq-auto.json", stdout.getvalue())

    def test_vllm_selected_profile_settings_screen_shows_path_and_tokens(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(
            model="/models/gemma4",
            extra_args="--served-model-name gemma4 --tool-call-parser hermes",
        )
        stdout = StringIO()

        with patch("builtins.input", side_effect=["R"]), contextlib.redirect_stdout(stdout):
            returned_profile, returned_id = launcher.show_vllm_selected_profile_settings(profile, "gemma4")

        output = stdout.getvalue()
        self.assertIs(returned_profile, profile)
        self.assertEqual(returned_id, "gemma4")
        self.assertIn("vLLM selected profile settings", output)
        self.assertIn("selected profile id: gemma4", output)
        self.assertIn("profile path:", output)
        self.assertIn("gemma4.json", output)
        self.assertIn("[1] --served-model-name", output)
        self.assertIn("[2] gemma4", output)
        self.assertIn("[3] --tool-call-parser", output)
        self.assertIn("[4] hermes", output)
        self.assertIn("vLLM default profile policy", output)

    def test_vllm_default_policy_preview_shows_full_profile_values(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(
            model="/models/cyankiwi-gemma-4-26B-A4B-it-AWQ-4bit",
            dtype="bfloat16",
            extra_args="--served-model-name gemma4",
        )
        stdout = StringIO()

        with patch("builtins.input", side_effect=["8", "4"]), contextlib.redirect_stdout(stdout):
            returned_profile, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "gemma4")

        output = stdout.getvalue()
        self.assertIs(returned_profile, profile)
        self.assertIn("Preview Strong / Safe Defaults", output)
        self.assertIn("Preview only. Nothing is saved.", output)
        self.assertIn("Shows how each default policy would change the current profile.", output)
        self.assertIn("── Hermes Desktop Strong ──", output)
        self.assertIn("── Desktop Safe ──", output)
        self.assertIn("Profile values:", output)
        self.assertIn("max_model_len: 96000", output)
        self.assertIn("max_model_len: 80000", output)
        self.assertIn("max_num_seqs: 3", output)
        self.assertIn("max_num_batched_tokens: 1024", output)
        self.assertIn("extra_args: --served-model-name gemma4 --enable-auto-tool-choice --tool-call-parser gemma4", output)
        self.assertIn("Validation messages:", output)
        self.assertIn("Command preview / dry-run:", output)

    def test_vllm_selected_profile_settings_applies_hermes_desktop_strong_policy(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(
            model="/models/cyankiwi-gemma-4-26B-A4B-it-AWQ-4bit",
            extra_args="--served-model-name gemma4 --tool-call-parser hermes",
        )
        launch_mock = Mock()
        stdout = StringIO()

        with (
            patch.object(launcher, "launch_vllm_profile_once", launch_mock),
            patch("builtins.input", side_effect=["8", "1"]),
            contextlib.redirect_stdout(stdout),
        ):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "gemma4")

        self.assertEqual(updated.gpu_memory_utilization, 0.88)
        self.assertEqual(updated.max_model_len, 96000)
        self.assertEqual(updated.max_num_batched_tokens, 1024)
        self.assertEqual(updated.max_num_seqs, 3)
        self.assertEqual(updated.tensor_parallel_size, 2)
        self.assertEqual(updated.kv_cache_dtype, "fp8")
        self.assertIn("--enable-auto-tool-choice", updated.extra_args)
        self.assertIn("--tool-call-parser gemma4", updated.extra_args)
        self.assertNotIn("--tool-call-parser hermes", updated.extra_args)
        self.assertEqual(updated.extra_args.count("--tool-call-parser"), 1)
        launch_mock.assert_not_called()
        self.assertIn("visible thinking/reasoning output policy: off by default", stdout.getvalue())

    def test_vllm_default_profile_policy_applies_desktop_safe_values(self) -> None:
        from modules.vllm_profiles import VllmProfile, apply_vllm_default_profile_policy

        profile = VllmProfile(model="/models/unknown")
        updated, messages = apply_vllm_default_profile_policy(profile, "desktop-safe", profile_id="unknown")

        self.assertEqual(updated.gpu_memory_utilization, 0.85)
        self.assertEqual(updated.max_model_len, 80000)
        self.assertEqual(updated.max_num_batched_tokens, 1024)
        self.assertEqual(updated.max_num_seqs, 3)
        self.assertEqual(updated.tensor_parallel_size, 2)
        self.assertEqual(updated.kv_cache_dtype, "fp8")
        self.assertNotIn("--enable-auto-tool-choice", updated.extra_args)
        self.assertNotIn("--tool-call-parser", updated.extra_args)
        self.assertTrue(any("manual/none" in message for message in messages))

    def test_vllm_tool_call_parser_auto_policy_detects_model_families(self) -> None:
        from modules.vllm_profiles import VllmProfile, infer_vllm_tool_call_parser

        self.assertEqual(infer_vllm_tool_call_parser(VllmProfile(model="/models/gemma-4-26b-awq")), "gemma4")
        self.assertEqual(infer_vllm_tool_call_parser(VllmProfile(model="/models/Qwen3-Coder-30B-A3B-Instruct")), "qwen3_xml")
        self.assertEqual(infer_vllm_tool_call_parser(VllmProfile(model="/models/Nous-Hermes-3")), "hermes")
        self.assertEqual(infer_vllm_tool_call_parser(VllmProfile(model="/models/Llama-3.1-8B-Instruct")), "llama3_json")
        self.assertEqual(infer_vllm_tool_call_parser(VllmProfile(model="/models/unknown-local-model")), "")

    def test_vllm_tool_call_parser_selection_removes_duplicates_and_can_clear(self) -> None:
        from modules.vllm_profiles import VllmProfile, apply_vllm_tool_call_parser

        profile = VllmProfile(
            model="/models/gemma4",
            extra_args="--served-model-name gemma --enable-auto-tool-choice --tool-call-parser hermes --tool-call-parser gemma4",
        )
        updated, _messages = apply_vllm_tool_call_parser(profile, "gemma4")

        self.assertIn("--enable-auto-tool-choice", updated.extra_args)
        self.assertIn("--tool-call-parser gemma4", updated.extra_args)
        self.assertEqual(updated.extra_args.count("--tool-call-parser"), 1)

        cleared, _messages = apply_vllm_tool_call_parser(updated, "none")
        self.assertNotIn("--enable-auto-tool-choice", cleared.extra_args)
        self.assertNotIn("--tool-call-parser", cleared.extra_args)

    def test_vllm_selected_profile_settings_can_add_common_option(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["5", "1", "qwen-alias"]), contextlib.redirect_stdout(stdout):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "qwen")

        self.assertIn("--served-model-name qwen-alias", updated.extra_args)
        self.assertIn("added vLLM extra option: --served-model-name", stdout.getvalue())

    def test_vllm_selected_profile_settings_can_remove_extra_arg_token(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", extra_args="--served-model-name qwen --enforce-eager")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["6", "3"]), contextlib.redirect_stdout(stdout):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "qwen")

        self.assertEqual(updated.extra_args, "--served-model-name qwen")
        self.assertIn("removed vLLM extra token: --enforce-eager", stdout.getvalue())

    def test_vllm_selected_profile_settings_rejects_bad_raw_extra_args(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", extra_args="--served-model-name qwen")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["4", "'unterminated"]), contextlib.redirect_stdout(stdout):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "qwen")

        self.assertEqual(updated.extra_args, "--served-model-name qwen")
        self.assertIn("extra_args could not be parsed", stdout.getvalue())
        self.assertIn("저장하지 않았습니다", stdout.getvalue())

    def test_vllm_selected_profile_settings_save_shows_validation_and_skips_invalid_save(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="", extra_args="--served-model-name empty")
        stdout = StringIO()
        backup_mock = Mock()
        save_mock = Mock()

        with (
            patch.object(launcher, "backup_vllm_profile_draft", backup_mock),
            patch.object(launcher, "save_vllm_profile_draft", save_mock),
            patch("builtins.input", side_effect=["7"]),
            contextlib.redirect_stdout(stdout),
        ):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "empty")

        self.assertIs(updated, profile)
        backup_mock.assert_not_called()
        save_mock.assert_not_called()
        self.assertIn("Validation messages:", stdout.getvalue())
        self.assertIn("model should not be empty", stdout.getvalue())
        self.assertIn("저장하지 않았습니다", stdout.getvalue())

    def test_vllm_selected_profile_settings_save_reports_backup_path(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileBackupResult, VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        backup_result = VllmProfileBackupResult(True, "/tmp/qwen.json", "/tmp/qwen.json.20260511-010203.bak", ["vLLM profile draft backup created: /tmp/qwen.json.20260511-010203.bak"])
        save_result = VllmProfileStoreResult(True, profile, "/tmp/qwen.json", ["saved"], "qwen")
        stdout = StringIO()

        with (
            patch.object(launcher, "backup_vllm_profile_draft", Mock(return_value=backup_result)),
            patch.object(launcher, "save_vllm_profile_draft", Mock(return_value=save_result)),
            patch("builtins.input", side_effect=["7"]),
            contextlib.redirect_stdout(stdout),
        ):
            updated, _profile_id = launcher.show_vllm_selected_profile_settings(profile, "qwen")

        self.assertIs(updated, profile)
        self.assertIn("vLLM profile draft backup created", stdout.getvalue())
        self.assertIn("/tmp/qwen.json.20260511-010203.bak", stdout.getvalue())

    def test_vllm_port_conflict_guidance_reports_owner_and_actions(self) -> None:
        from modules.vllm_profiles import VllmPortOwner, VllmProfile, vllm_port_conflict_guidance_lines

        lines = vllm_port_conflict_guidance_lines(
            VllmProfile(model="local-model", port=8000),
            owner_lookup=lambda port: VllmPortOwner(1234, "python -m vllm.entrypoints.openai.api_server", "owner found"),
        )
        text = "\n".join(lines)

        self.assertIn("port 8000 is already in use", text)
        self.assertIn("owner:", text)
        self.assertIn("PID 1234 / command python -m vllm.entrypoints.openai.api_server", text)
        self.assertIn("[1] 기존 서버 재사용", text)
        self.assertIn("[2] latest run status/log 확인", text)
        self.assertIn("[3] selected profile port 변경", text)
        self.assertIn("[4] 기존 프로세스 종료 후 launch", text)

    def test_vllm_custom_launch_port_conflict_can_route_to_port_change(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmPreflightCheck, VllmPreflightReport, VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        updated_profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", port=8010)
        preflight = VllmPreflightReport([VllmPreflightCheck("port availability", False, "port 8000 is already in use")])
        prompt_mock = Mock(return_value=(updated_profile, "qwen"))
        launch_mock = Mock()
        stdout = StringIO()

        with (
            patch.object(launcher, "run_vllm_preflight", Mock(return_value=preflight)),
            patch.object(launcher, "prompt_vllm_selected_profile_fields", prompt_mock),
            patch.object(launcher, "launch_vllm_profile_once", launch_mock),
            patch("builtins.input", side_effect=["3"]),
            contextlib.redirect_stdout(stdout),
        ):
            returned_profile, returned_id = launcher.show_vllm_custom_launch(profile, "qwen")

        self.assertIs(returned_profile, updated_profile)
        self.assertEqual(returned_id, "qwen")
        prompt_mock.assert_called_once_with(profile, "qwen", ["host", "port"])
        launch_mock.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("vLLM port conflict", output)
        self.assertIn("[1] 기존 서버 재사용", output)
        self.assertIn("[3] selected profile port 변경", output)

    def test_vllm_profile_backup_helper_creates_timestamp_backup(self) -> None:
        from modules.vllm_profile_store import backup_vllm_profile_draft, default_vllm_profile_path, save_vllm_profile_draft
        from modules.vllm_profiles import VllmProfile

        with TemporaryDirectory() as directory:
            profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
            save_vllm_profile_draft(profile, profile_id="qwen", store_root=directory)
            result = backup_vllm_profile_draft(profile_id="qwen", store_root=directory, timestamp="20260511-010203")
            backup_path = Path(default_vllm_profile_path("qwen", store_root=directory) + ".20260511-010203.bak")

            self.assertTrue(result.ok, result.messages)
            self.assertEqual(result.backup_path, str(backup_path))
            self.assertTrue(backup_path.exists())
            self.assertIn("vLLM profile draft backup created", "\n".join(result.messages))

    def test_verified_gemma4_beta_profile_selection_saves_selected_profile_id(self) -> None:
        from modules.vllm_profile_store import load_selected_vllm_profile_id, load_vllm_profile_draft, save_verified_gemma4_26b_awq_beta_profile
        from modules.vllm_profiles import VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH

        with TemporaryDirectory() as directory:
            result = save_verified_gemma4_26b_awq_beta_profile(store_root=directory)
            selected = load_selected_vllm_profile_id(store_root=directory)
            loaded = load_vllm_profile_draft(profile_id="gemma4-26b-awq-auto", store_root=directory)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.profile_id, "gemma4-26b-awq-auto")
        self.assertTrue(selected.ok, selected.messages)
        self.assertEqual(selected.profile_id, "gemma4-26b-awq-auto")
        self.assertTrue(loaded.ok, loaded.messages)
        self.assertIsNotNone(loaded.profile)
        assert loaded.profile is not None
        self.assertEqual(loaded.profile.model, VERIFIED_GEMMA4_26B_AWQ_MODEL_PATH)
        self.assertIn("--served-model-name gemma4-26b-awq-auto", loaded.profile.extra_args)
        self.assertIn("--tool-call-parser gemma4", loaded.profile.extra_args)

    def test_vllm_profile_menu_delete_cancel_does_not_call_delete(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileListResult, VllmStoredProfileInfo
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="current-model")
        list_result = VllmProfileListResult(
            True,
            [VllmStoredProfileInfo("large-q4", "/tmp/large-q4.json", "Local/LargeModel", [])],
            "/tmp/profiles",
            [],
        )
        mocked_delete = Mock()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", Mock(return_value=list_result)),
            patch.object(launcher, "delete_vllm_profile_draft", mocked_delete),
            patch("builtins.input", side_effect=["7", "1", "no"]),
            contextlib.redirect_stdout(StringIO()),
        ):
            returned_profile, returned_id = launcher.show_vllm_profile_menu(profile, "large-q4", return_profile_id=True)

        mocked_delete.assert_not_called()
        self.assertIs(returned_profile, profile)
        self.assertEqual(returned_id, "large-q4")

    def test_vllm_profile_menu_can_preview_custom_script(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["9"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile)

        self.assertIs(result, profile)
        self.assertIn("vLLM custom script preview", stdout.getvalue())
        self.assertIn("exec", stdout.getvalue())
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", stdout.getvalue())

    def test_vllm_profile_menu_can_save_custom_script(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_script_builder import VllmScriptSaveResult
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        mocked_save = Mock(return_value=VllmScriptSaveResult(True, "/tmp/vllm.sh", ["saved"]))
        stdout = StringIO()

        with patch.object(launcher, "save_vllm_script", mocked_save), patch("builtins.input", side_effect=["10"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile)
        self.assertIs(result, profile)
        self.assertIn("vLLM custom script save", stdout.getvalue())
        self.assertIn("/tmp/vllm.sh", stdout.getvalue())

    def test_vllm_profile_menu_custom_launch_requires_typed_confirmation(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmPreflightCheck, VllmPreflightReport, VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        launch_result = type(
            "Launch",
            (),
            {
                "ok": False,
                "preset_id": "custom-draft",
                "pid": None,
                "run_id": None,
                "log_path": None,
                "record_path": None,
                "profile_path": None,
                "host": None,
                "port": None,
                "command": [],
                "messages": ["cancelled"],
            },
        )()
        mocked_launch = Mock(return_value=launch_result)

        preflight = VllmPreflightReport([VllmPreflightCheck("port availability", True, "port is available")])

        with (
            patch.object(launcher, "run_vllm_preflight", Mock(return_value=preflight)),
            patch.object(launcher, "launch_vllm_profile_once", mocked_launch),
            patch("builtins.input", side_effect=["11", "no"]),
            contextlib.redirect_stdout(StringIO()),
        ):
            result = launcher.show_vllm_profile_menu(profile, "large-q4")

        mocked_launch.assert_called_once_with(profile, confirmed=False, preset_id="large-q4", profile_path=launcher.default_vllm_profile_path("large-q4"))
        self.assertIs(result, profile)

    def test_vllm_launch_result_prints_profile_path(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib

        launch_result = type(
            "Launch",
            (),
            {
                "ok": True,
                "preset_id": "large-q4",
                "pid": 123,
                "run_id": "vllm-large-q4-20260511-010203",
                "log_path": "/tmp/vllm.log",
                "record_path": "/tmp/vllm.json",
                "profile_path": "/tmp/profiles/large-q4.json",
                "host": "127.0.0.1",
                "port": 8000,
                "command": ["/home/kalijin/bin/vllm-rocm", "serve", "local-model"],
                "messages": ["started"],
            },
        )()
        stdout = StringIO()

        with contextlib.redirect_stdout(stdout):
            launcher.print_vllm_launch_result(launch_result)

        output = stdout.getvalue()
        self.assertIn("profile_path: /tmp/profiles/large-q4.json", output)
        self.assertIn("record_path: /tmp/vllm.json", output)

    def test_vllm_smoke_launch_preview_is_not_read_only_profile_text(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmPreflightCheck

        text = launcher.vllm_smoke_launch_preview_text(
            port_check=lambda host, port: VllmPreflightCheck(
                "port availability",
                True,
                f"port {port} is available on {host}",
            )
        )

        self.assertIn("vLLM smoke launch preview", text)
        self.assertIn("Launch target preset: smoke-qwen-0.5b", text)
        self.assertIn("Command preview / dry-run", text)
        self.assertIn("Launch preflight:", text)
        self.assertIn("vLLM may use GPU memory immediately", text)
        self.assertIn("Custom vLLM profile launch is handled separately from [B]", text)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertNotIn("read-only registry", text)
        self.assertNotIn("no launch button", text)
        self.assertNotIn("Custom vLLM profile launch is not implemented", text)

    def test_vllm_runner_builds_launch_plan_for_valid_smoke_profile(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import build_vllm_launch_plan

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)

            readiness = build_vllm_launch_plan(
                profile,
                timestamp="20260510-190000",
                state_root=root / "runs",
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
            )

        self.assertTrue(readiness.ok, readiness.messages)
        self.assertIsNotNone(readiness.plan)
        assert readiness.plan is not None
        self.assertEqual(readiness.plan.preset_id, "smoke-qwen-0.5b")
        self.assertEqual(readiness.plan.run_id, "vllm-smoke-qwen-0.5b-20260510-190000")
        self.assertTrue(readiness.plan.log_path.endswith("vllm-smoke-qwen-0.5b-20260510-190000.log"))
        self.assertEqual(readiness.plan.host, "127.0.0.1")
        self.assertEqual(readiness.plan.port, 8000)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", readiness.plan.command)
        self.assertEqual(readiness.plan.env_preview["VLLM_CACHE_ROOT"], "/mnt/data_main/ai-cache/vllm")
        self.assertEqual(readiness.plan.env_preview["HF_HOME"], "/mnt/data_main/ai-cache/huggingface")
        self.assertEqual(readiness.plan.env_preview["TRANSFORMERS_CACHE"], "/mnt/data_main/ai-cache/huggingface")

    def test_vllm_runner_invalid_profile_returns_messages_and_no_plan(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_runner import build_vllm_launch_plan

        readiness = build_vllm_launch_plan(VllmProfile(model=""))

        self.assertFalse(readiness.ok)
        self.assertIsNone(readiness.plan)
        self.assertIn("model should not be empty", readiness.messages)

    def test_vllm_runner_preflight_failure_returns_messages_and_no_plan(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import build_vllm_launch_plan

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            readiness = build_vllm_launch_plan(
                profile,
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    False,
                    f"port {port} is not available on {host}",
                ),
            )

        self.assertFalse(readiness.ok)
        self.assertIsNone(readiness.plan)
        self.assertTrue(any("port availability:" in message for message in readiness.messages))

    def test_vllm_custom_profile_launch_requires_confirmation(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_runner import launch_vllm_profile_once

        def fake_popen(_command, **_kwargs):
            raise AssertionError("Popen should not be called")

        result = launch_vllm_profile_once(VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct"), confirmed=False, popen_factory=fake_popen)

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertEqual(result.preset_id, "custom-draft")
        self.assertIn("explicit confirmation is required", "\n".join(result.messages))

    def test_vllm_verified_beta_profile_launch_requires_confirmation(self) -> None:
        from modules.vllm_profiles import verified_gemma4_26b_awq_vllm_profile
        from modules.vllm_runner import launch_vllm_profile_once

        def fake_popen(_command, **_kwargs):
            raise AssertionError("Popen should not be called")

        result = launch_vllm_profile_once(
            verified_gemma4_26b_awq_vllm_profile(),
            confirmed=False,
            preset_id="gemma4-26b-awq-auto",
            popen_factory=fake_popen,
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertEqual(result.preset_id, "gemma4-26b-awq-auto")
        self.assertIn("explicit confirmation is required", "\n".join(result.messages))

    def test_vllm_custom_profile_launch_uses_runner_and_run_record(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile
        from modules.vllm_profile_store import load_selected_vllm_profile_draft, load_selected_vllm_profile_id, load_vllm_profile_draft
        from modules.vllm_runner import launch_vllm_profile_once, read_vllm_run_record

        class FakeProcess:
            pid = 54321

        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = VllmProfile(
                wrapper_path=str(wrapper),
                model="Qwen/Qwen2.5-0.5B-Instruct",
                port=54324,
                kv_cache_dtype="fp8",
            )
            profile_path = root / "profiles" / "custom-qwen.json"
            result = launch_vllm_profile_once(
                profile,
                confirmed=True,
                preset_id="custom-qwen",
                timestamp="20260510-224500",
                state_root=root / "runs",
                profile_path=profile_path,
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
                popen_factory=fake_popen,
            )
            record = read_vllm_run_record(str(result.record_path))
            saved_profile = load_vllm_profile_draft(profile_id="custom-qwen", store_root=profile_path.parent)
            selected = load_selected_vllm_profile_id(store_root=profile_path.parent)
            selected_profile = load_selected_vllm_profile_draft(store_root=profile_path.parent)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.pid, 54321)
        self.assertEqual(result.preset_id, "custom-qwen")
        self.assertEqual(result.run_id, "vllm-custom-qwen-20260510-224500")
        self.assertIn("vLLM custom profile launch started", "\n".join(result.messages))
        self.assertEqual(calls[0][1]["start_new_session"], True)
        self.assertEqual(calls[0][1]["env"]["VLLM_CACHE_ROOT"], "/mnt/data_main/ai-cache/vllm")
        self.assertTrue(record.ok, record.messages)
        self.assertIsNotNone(record.record)
        assert record.record is not None
        self.assertEqual(record.record.preset_id, "custom-qwen")
        self.assertEqual(record.record.profile_id, "custom-qwen")
        self.assertEqual(record.record.profile_path, str(profile_path))
        self.assertIsNotNone(record.record.profile_snapshot)
        assert record.record.profile_snapshot is not None
        self.assertEqual(record.record.profile_snapshot["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(record.record.profile_snapshot["kv_cache_dtype"], "fp8")
        self.assertTrue(saved_profile.ok, saved_profile.messages)
        self.assertIsNotNone(saved_profile.profile)
        assert saved_profile.profile is not None
        self.assertEqual(saved_profile.profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(selected.ok, selected.messages)
        self.assertEqual(selected.profile_id, "custom-qwen")
        self.assertTrue(selected_profile.ok, selected_profile.messages)
        self.assertIsNotNone(selected_profile.profile)
        assert selected_profile.profile is not None
        self.assertEqual(selected_profile.profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("launched vLLM profile saved", "\n".join(result.messages))
        self.assertIn("selected vLLM profile updated", "\n".join(result.messages))

    def test_vllm_runner_sanitizes_preset_id_for_run_id(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import build_vllm_launch_plan

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            readiness = build_vllm_launch_plan(
                profile,
                preset_id="../bad preset",
                timestamp="20260510-190000",
                state_root=directory,
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
            )

        self.assertTrue(readiness.ok, readiness.messages)
        self.assertIsNotNone(readiness.plan)
        assert readiness.plan is not None
        self.assertEqual(readiness.plan.preset_id, "../bad preset")
        self.assertEqual(readiness.plan.run_id, "vllm-bad_preset-20260510-190000")

    def test_vllm_smoke_launch_success_uses_log_redirection_and_new_session(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        calls: list[dict] = []

        class FakeProcess:
            pid = 43210

        def fake_popen(command, **kwargs):
            calls.append({"command": command, "kwargs": kwargs})
            self.assertIs(kwargs["stdout"], kwargs["stderr"])
            self.assertFalse(kwargs["stdout"].closed)
            return FakeProcess()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    timestamp="20260510-191500",
                    state_root=root / "runs",
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                    popen_factory=fake_popen,
                )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(result.pid, 43210)
        self.assertEqual(result.run_id, "vllm-smoke-qwen-0.5b-20260510-191500")
        self.assertTrue(str(result.log_path).endswith("vllm-smoke-qwen-0.5b-20260510-191500.log"))
        self.assertEqual(result.host, "127.0.0.1")
        self.assertEqual(result.port, 8000)
        self.assertEqual(result.preset_id, "smoke-qwen-0.5b")
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", result.command)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kwargs"]["start_new_session"], True)
        self.assertEqual(calls[0]["kwargs"]["env"]["VLLM_CACHE_ROOT"], "/mnt/data_main/ai-cache/vllm")

    def test_vllm_smoke_launch_popen_receives_expanded_wrapper_path(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        calls: list[list[str]] = []

        class FakeProcess:
            pid = 43211

        def fake_popen(command, **_kwargs):
            calls.append(command)
            return FakeProcess()

        with TemporaryDirectory() as directory:
            home = Path(directory)
            wrapper = home / "bin" / "vllm-rocm"
            wrapper.parent.mkdir()
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = "~/bin/vllm-rocm"
            with patch.dict(os.environ, {"HOME": directory}), patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    timestamp="20260510-192000",
                    state_root=home / "runs",
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                    popen_factory=fake_popen,
                )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(calls[0][0], str(wrapper))
        self.assertNotIn("~", calls[0][0])

    def test_vllm_smoke_launch_saves_run_record(self) -> None:
        import json
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once, read_vllm_run_record
        from unittest.mock import patch

        class FakeProcess:
            pid = 43212

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    timestamp="20260510-193000",
                    state_root=root / "runs",
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                    popen_factory=lambda command, **kwargs: FakeProcess(),
                )
            latest_path = root / "runs" / "latest.json"
            record_result = read_vllm_run_record(str(result.record_path))
            latest_data = json.loads(latest_path.read_text())
            latest_exists = latest_path.is_file()

        self.assertTrue(result.ok, result.messages)
        self.assertIsNotNone(result.record_path)
        self.assertTrue(str(result.record_path).endswith("vllm-smoke-qwen-0.5b-20260510-193000.json"))
        self.assertTrue(latest_exists)
        self.assertTrue(record_result.ok, record_result.messages)
        self.assertIsNotNone(record_result.record)
        assert record_result.record is not None
        self.assertEqual(record_result.record.schema, "llama-suite.run.v1")
        self.assertEqual(record_result.record.backend, "vllm")
        self.assertEqual(record_result.record.preset_id, "smoke-qwen-0.5b")
        self.assertEqual(record_result.record.run_id, "vllm-smoke-qwen-0.5b-20260510-193000")
        self.assertEqual(record_result.record.pid, 43212)
        self.assertEqual(record_result.record.status_hint, "started")
        self.assertEqual(record_result.record.env_preview["VLLM_CACHE_ROOT"], "/mnt/data_main/ai-cache/vllm")
        self.assertEqual(record_result.record.env_preview["HF_HOME"], "/mnt/data_main/ai-cache/huggingface")
        self.assertEqual(record_result.record.env_preview["TRANSFORMERS_CACHE"], "/mnt/data_main/ai-cache/huggingface")
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", record_result.record.command)
        self.assertEqual(latest_data["run_id"], "vllm-smoke-qwen-0.5b-20260510-193000")
        self.assertEqual(latest_data["schema"], "llama-suite.run.v1")

    def test_vllm_smoke_launch_record_write_failure_keeps_launch_ok(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import VllmRunRecordResult, launch_vllm_smoke_once
        from unittest.mock import patch

        class FakeProcess:
            pid = 43213

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile), patch(
                "modules.vllm_runner.write_vllm_run_record",
                side_effect=lambda record, state_root=None: VllmRunRecordResult(False, record, str(root / "runs" / "broken.json"), ["run record write failed: disk full"]),
            ):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    timestamp="20260510-193500",
                    state_root=root / "runs",
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                    popen_factory=lambda command, **kwargs: FakeProcess(),
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.pid, 43213)
        self.assertIn("run record write failed: disk full", "\n".join(result.messages))

    def test_vllm_latest_run_record_loads_newest_record(self) -> None:
        from modules.vllm_runner import VllmRunRecord, latest_vllm_run_record, write_vllm_run_record

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = VllmRunRecord(
                backend="vllm",
                preset_id="smoke-qwen-0.5b",
                run_id="vllm-smoke-qwen-0.5b-20260510-193000",
                pid=1,
                command=["one"],
                env_preview={"VLLM_CACHE_ROOT": "/cache/one"},
                log_path="/tmp/one.log",
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-10T19:30:00",
                status_hint="started",
            )
            second = VllmRunRecord(
                backend="vllm",
                preset_id="smoke-qwen-0.5b",
                run_id="vllm-smoke-qwen-0.5b-20260510-193100",
                pid=2,
                command=["two"],
                env_preview={"VLLM_CACHE_ROOT": "/cache/two"},
                log_path="/tmp/two.log",
                host="127.0.0.1",
                port=8000,
                started_at="2026-05-10T19:31:00",
                status_hint="started",
            )
            write_vllm_run_record(first, state_root=root)
            write_vllm_run_record(second, state_root=root)
            result = latest_vllm_run_record(state_root=root)

        self.assertTrue(result.ok, result.messages)
        self.assertIsNotNone(result.record)
        assert result.record is not None
        self.assertEqual(result.record.pid, 2)

    def test_vllm_latest_run_record_reports_missing_latest_gracefully(self) -> None:
        from modules.vllm_runner import latest_vllm_run_record

        with TemporaryDirectory() as directory:
            result = latest_vllm_run_record(state_root=directory)

        self.assertFalse(result.ok)
        self.assertIsNone(result.record)
        self.assertIn("no vLLM run records found", "\n".join(result.messages))

    def test_vllm_run_record_rejects_invalid_schema_and_backend(self) -> None:
        from modules.vllm_runner import read_vllm_run_record

        with TemporaryDirectory() as directory:
            schema_path = Path(directory) / "bad-schema.json"
            backend_path = Path(directory) / "bad-backend.json"
            base = {
                "schema": "wrong",
                "backend": "vllm",
                "preset_id": "smoke-qwen-0.5b",
                "run_id": "run-a",
                "pid": 1234,
                "command": ["cmd"],
                "env_preview": {},
                "log_path": "/tmp/vllm.log",
                "host": "127.0.0.1",
                "port": 8000,
                "started_at": "2026-05-10T19:31:00+09:00",
                "status_hint": "started",
            }
            schema_path.write_text(json.dumps(base))
            base["schema"] = "llama-suite.run.v1"
            base["backend"] = "llama.cpp"
            backend_path.write_text(json.dumps(base))

            schema_result = read_vllm_run_record(str(schema_path))
            backend_result = read_vllm_run_record(str(backend_path))

        self.assertFalse(schema_result.ok)
        self.assertIn("invalid run record schema", "\n".join(schema_result.messages))
        self.assertFalse(backend_result.ok)
        self.assertIn("invalid run record backend", "\n".join(backend_result.messages))

    def test_vllm_smoke_launch_refuses_without_confirmation(self) -> None:
        from modules.vllm_runner import launch_vllm_smoke_once

        def fake_popen(_command, **_kwargs):
            raise AssertionError("Popen should not be called")

        result = launch_vllm_smoke_once(confirmed=False, popen_factory=fake_popen)

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertEqual(result.command, [])
        self.assertIn("explicit confirmation is required", "\n".join(result.messages))

    def test_vllm_smoke_launch_refuses_invalid_profile(self) -> None:
        from modules.vllm_profiles import VllmProfile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        with patch("modules.vllm_runner.smoke_vllm_profile", return_value=VllmProfile(model="")):
            result = launch_vllm_smoke_once(confirmed=True)

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertIn("model should not be empty", result.messages)

    def test_vllm_smoke_launch_refuses_preflight_failure(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        False,
                        f"port {port} is not available on {host}",
                    ),
                )

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertTrue(any("port availability:" in message for message in result.messages))

    def test_vllm_smoke_launch_returns_failure_when_log_directory_cannot_be_created(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            blocked_root = root / "blocked"
            blocked_root.write_text("not a directory")
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    state_root=blocked_root,
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                )

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertTrue(any("launch failed:" in message for message in result.messages))

    def test_vllm_smoke_launch_returns_failure_when_popen_raises(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, smoke_vllm_profile
        from modules.vllm_runner import launch_vllm_smoke_once
        from unittest.mock import patch

        def fake_popen(_command, **_kwargs):
            raise OSError("boom")

        with TemporaryDirectory() as directory:
            wrapper = Path(directory) / "vllm-rocm"
            wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
            wrapper.chmod(0o755)
            profile = smoke_vllm_profile()
            profile.wrapper_path = str(wrapper)
            with patch("modules.vllm_runner.smoke_vllm_profile", return_value=profile):
                result = launch_vllm_smoke_once(
                    confirmed=True,
                    state_root=directory,
                    port_check=lambda host, port: VllmPreflightCheck(
                        "port availability",
                        True,
                        f"port {port} is available on {host}",
                    ),
                    popen_factory=fake_popen,
                )

        self.assertFalse(result.ok)
        self.assertIsNone(result.pid)
        self.assertTrue(any("boom" in message for message in result.messages))

    def test_vllm_popen_is_confined_to_runner_module(self) -> None:
        launcher_source = (ROOT / "llama-launcher-complete.py").read_text()
        runner_source = (ROOT / "modules" / "vllm_runner.py").read_text()

        self.assertNotIn("subprocess", launcher_source)
        self.assertNotIn("Popen", launcher_source)
        self.assertIn("subprocess.Popen", runner_source)

    def test_vllm_smoke_launch_tui_requires_launch_confirmation(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmLaunchResult
        from unittest.mock import Mock, patch

        mocked_launch = Mock(
            return_value=VllmLaunchResult(
                ok=False,
                pid=None,
                run_id=None,
                log_path=None,
                host=None,
                port=None,
                preset_id="smoke-qwen-0.5b",
                command=[],
                messages=["launch cancelled: explicit confirmation is required"],
            )
        )
        stdout = StringIO()
        with patch.object(launcher, "launch_vllm_smoke_once", mocked_launch), patch("builtins.input", return_value="no"), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_launch_once()

        mocked_launch.assert_called_once_with(confirmed=False)
        text = stdout.getvalue()
        self.assertIn("vLLM smoke launch", text)
        self.assertIn("confirmation", text)
        self.assertIn("not started", text)

    def test_vllm_smoke_status_reports_alive_and_dead_with_injected_checker(self) -> None:
        from modules.vllm_runner import check_vllm_smoke_status

        alive = check_vllm_smoke_status(pid=1234, run_id="run-a", log_path="/tmp/missing.log", alive_check=lambda pid: True)
        dead = check_vllm_smoke_status(pid=1234, run_id="run-a", log_path="/tmp/missing.log", alive_check=lambda pid: False)

        self.assertTrue(alive.ok)
        self.assertTrue(alive.alive)
        self.assertFalse(alive.log_exists)
        self.assertIn("process is alive", "\n".join(alive.messages))
        self.assertTrue(dead.ok)
        self.assertFalse(dead.alive)
        self.assertIn("process is not alive", "\n".join(dead.messages))

    def test_vllm_latest_run_helper_aliases_keep_smoke_lifecycle_compatibility(self) -> None:
        from modules.vllm_runner import check_vllm_run_status, read_vllm_run_log, stop_vllm_run

        status = check_vllm_run_status(pid=1234, run_id="run-a", log_path="/tmp/missing.log", alive_check=lambda pid: True, port_check=lambda host, port: False)
        log = read_vllm_run_log("/tmp/llama-suite-missing-vllm.log", last_lines=10)
        stop = stop_vllm_run(pid=1234, run_id="run-a", confirmed=False)

        self.assertTrue(status.ok)
        self.assertFalse(log.ok)
        self.assertFalse(stop.ok)
        self.assertIn("explicit confirmation", "\n".join(stop.messages))

    def test_vllm_smoke_status_reports_log_exists_and_optional_port(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck
        from modules.vllm_runner import check_vllm_smoke_status

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "vllm.log"
            log_path.write_text("ready\n")
            result = check_vllm_smoke_status(
                pid="1234",
                run_id="run-a",
                log_path=str(log_path),
                alive_check=lambda pid: True,
                port_check=lambda host, port: VllmPreflightCheck("port availability", True, f"port {port} is listening on {host}"),
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.log_exists)
        self.assertTrue(result.port_listening)
        self.assertIn("port 8000 listening on 127.0.0.1: True", "\n".join(result.messages))

    def test_vllm_smoke_status_uses_record_host_and_port(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck
        from modules.vllm_runner import check_vllm_smoke_status

        calls: list[tuple] = []

        result = check_vllm_smoke_status(
            pid=1234,
            run_id="run-a",
            log_path="/tmp/missing.log",
            host="100.68.40.87",
            port=8010,
            alive_check=lambda pid: True,
            port_check=lambda host, port: calls.append((host, port)) or VllmPreflightCheck("port availability", True, "ok"),
        )

        self.assertTrue(result.ok)
        self.assertEqual(calls, [("100.68.40.87", 8010)])

    def test_vllm_smoke_status_checks_tcp_listening_by_default(self) -> None:
        from modules.vllm_runner import VllmSmokeStatusPortCheck, check_vllm_smoke_status
        from unittest.mock import patch

        with patch("modules.vllm_runner.check_tcp_listening", return_value=VllmSmokeStatusPortCheck(True, "ok")) as mocked_check:
            result = check_vllm_smoke_status(
                pid=1234,
                run_id="run-a",
                log_path="/tmp/missing.log",
                host="100.68.40.87",
                port=8010,
                alive_check=lambda pid: True,
            )

        self.assertTrue(result.port_listening)
        mocked_check.assert_called_once_with("100.68.40.87", 8010)

    def test_vllm_latest_run_summary_maps_status_states(self) -> None:
        from modules.vllm_runner import VllmLatestRunSummary, VllmRunRecord, VllmRunRecordResult, latest_vllm_run_summary

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["/home/kalijin/bin/vllm-rocm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )

        def status(alive, port_listening):
            return type("Status", (), {"alive": alive, "port_listening": port_listening, "messages": []})()

        no_record = latest_vllm_run_summary(latest_record=VllmRunRecordResult(False, None, None, ["no record"]))
        ready = latest_vllm_run_summary(latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []), status_check=lambda **kwargs: status(True, True))
        starting = latest_vllm_run_summary(latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []), status_check=lambda **kwargs: status(True, False))
        stopped = latest_vllm_run_summary(latest_record=VllmRunRecordResult(True, record, "/tmp/latest.json", []), status_check=lambda **kwargs: status(False, False))

        self.assertIsInstance(no_record, VllmLatestRunSummary)
        self.assertEqual(no_record.status, "UNKNOWN")
        self.assertEqual(ready.status, "READY")
        self.assertEqual(ready.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(ready.endpoint, "http://127.0.0.1:8000/v1")
        self.assertEqual(starting.status, "STARTING")
        self.assertEqual(stopped.status, "STOPPED")

    def test_vllm_tcp_listening_helper_reports_true_false_and_errors(self) -> None:
        from modules.vllm_runner import check_tcp_listening
        from unittest.mock import patch

        class FakeSocket:
            def __init__(self, code: int | None = 0, error: Exception | None = None):
                self.code = code
                self.error = error

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return False

            def settimeout(self, _timeout):
                pass

            def connect_ex(self, _target):
                if self.error:
                    raise self.error
                return self.code

        with patch("modules.vllm_runner.socket.socket", return_value=FakeSocket(0)):
            listening = check_tcp_listening("127.0.0.1", 8000)
        with patch("modules.vllm_runner.socket.socket", return_value=FakeSocket(111)):
            closed = check_tcp_listening("127.0.0.1", 8000)
        with patch("modules.vllm_runner.socket.socket", return_value=FakeSocket(error=OSError("boom"))):
            error = check_tcp_listening("127.0.0.1", 8000)
        invalid = check_tcp_listening("127.0.0.1", "bad")

        self.assertTrue(listening.ok)
        self.assertIn("is listening", listening.message)
        self.assertFalse(closed.ok)
        self.assertIn("is not listening", closed.message)
        self.assertFalse(error.ok)
        self.assertIn("listening check failed", error.message)
        self.assertFalse(invalid.ok)
        self.assertIn("port should be a positive integer", invalid.message)

    def test_vllm_smoke_manage_uses_latest_record_for_status_defaults(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["cmd"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="100.68.40.87",
            port=8010,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "smoke-qwen-0.5b", "pid": 1234, "run_id": "run-latest", "log_path": "/tmp/latest.log", "alive": True, "log_exists": False, "port_listening": None, "messages": []})())
        stdout = StringIO()

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_run_status", mocked_status), patch("builtins.input", side_effect=["1", ""]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="1234", run_id="run-latest", log_path="/tmp/latest.log", preset_id="smoke-qwen-0.5b", host="100.68.40.87", port=8010)
        self.assertIn("latest run record", stdout.getvalue())

    def test_vllm_smoke_manage_uses_latest_record_preset_id_for_status(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="custom-draft",
            run_id="run-custom",
            pid=1234,
            command=["cmd"],
            env_preview={},
            log_path="/tmp/custom.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "custom-draft", "pid": 1234, "run_id": "run-custom", "log_path": "/tmp/custom.log", "alive": True, "log_exists": True, "port_listening": True, "messages": []})())

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_run_status", mocked_status), patch("builtins.input", side_effect=["1", ""]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="1234", run_id="run-custom", log_path="/tmp/custom.log", preset_id="custom-draft", host="127.0.0.1", port=8000)

    def test_vllm_smoke_manage_missing_latest_falls_back_to_manual_status(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecordResult
        from unittest.mock import Mock, patch

        latest = VllmRunRecordResult(False, None, None, ["no vLLM run records found under /tmp/runs"])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "smoke-qwen-0.5b", "pid": 1234, "run_id": "manual-run", "log_path": "/tmp/manual.log", "alive": True, "log_exists": False, "port_listening": None, "messages": []})())
        stdout = StringIO()

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_run_status", mocked_status), patch("builtins.input", side_effect=["1", "", "1234", "manual-run", "/tmp/manual.log", "127.0.0.1", "8000"]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="1234", run_id="manual-run", log_path="/tmp/manual.log", preset_id="smoke-qwen-0.5b", host="127.0.0.1", port="8000")
        self.assertIn("latest.json이 없거나 유효하지 않으면", stdout.getvalue())

    def test_vllm_smoke_manage_manual_override_still_works_with_latest_record(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["cmd"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "smoke-qwen-0.5b", "pid": 2222, "run_id": "manual-run", "log_path": "/tmp/manual.log", "alive": True, "log_exists": False, "port_listening": None, "messages": []})())

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_run_status", mocked_status), patch("builtins.input", side_effect=["1", "-", "2222", "manual-run", "/tmp/manual.log", "0.0.0.0", "8020"]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="2222", run_id="manual-run", log_path="/tmp/manual.log", preset_id="smoke-qwen-0.5b", host="0.0.0.0", port="8020")

    def test_vllm_smoke_manage_stop_with_latest_still_requires_confirmation(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="smoke-qwen-0.5b",
            run_id="run-latest",
            pid=1234,
            command=["cmd"],
            env_preview={},
            log_path="/tmp/latest.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_stop = Mock(return_value=type("Stop", (), {"ok": False, "preset_id": "smoke-qwen-0.5b", "pid": 1234, "run_id": "run-latest", "messages": ["cancelled"]})())

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "stop_vllm_run", mocked_stop), patch("builtins.input", side_effect=["3", "", "no"]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_stop.assert_called_once_with(pid="1234", run_id="run-latest", preset_id="smoke-qwen-0.5b", confirmed=False)

    def test_vllm_smoke_manage_uses_latest_record_preset_id_for_stop(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecord, VllmRunRecordResult
        from unittest.mock import Mock, patch

        record = VllmRunRecord(
            backend="vllm",
            preset_id="custom-draft",
            run_id="run-custom",
            pid=1234,
            command=["cmd"],
            env_preview={},
            log_path="/tmp/custom.log",
            host="127.0.0.1",
            port=8000,
            started_at="2026-05-10T19:31:00+09:00",
            status_hint="started",
        )
        latest = VllmRunRecordResult(True, record, "/tmp/latest.json", [])
        mocked_stop = Mock(return_value=type("Stop", (), {"ok": True, "preset_id": "custom-draft", "pid": 1234, "run_id": "run-custom", "messages": ["stopped"]})())

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "stop_vllm_run", mocked_stop), patch("builtins.input", side_effect=["3", "", "stop"]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_stop.assert_called_once_with(pid="1234", run_id="run-custom", preset_id="custom-draft", confirmed=True)

    def test_vllm_smoke_log_reports_missing_file(self) -> None:
        from modules.vllm_runner import read_vllm_smoke_log

        result = read_vllm_smoke_log("/tmp/llama-suite-missing-vllm.log", last_lines=10)

        self.assertFalse(result.ok)
        self.assertEqual(result.lines, [])
        self.assertIn("log file does not exist", "\n".join(result.messages))

    def test_vllm_smoke_log_returns_last_n_lines(self) -> None:
        from modules.vllm_runner import read_vllm_smoke_log

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "vllm.log"
            log_path.write_text("one\ntwo\nthree\nfour\n")
            result = read_vllm_smoke_log(str(log_path), last_lines=2)

        self.assertTrue(result.ok)
        self.assertEqual(result.lines, ["three", "four"])
        self.assertEqual(result.messages, [])

    def test_vllm_smoke_stop_refuses_without_confirmation(self) -> None:
        from modules.vllm_runner import stop_vllm_smoke

        calls: list[str] = []

        result = stop_vllm_smoke(
            pid=1234,
            run_id="run-a",
            confirmed=False,
            getpgid_func=lambda pid: calls.append("getpgid") or 999,
            killpg_func=lambda pgid, sig: calls.append("killpg"),
        )

        self.assertFalse(result.ok)
        self.assertEqual(calls, [])
        self.assertIn("explicit confirmation is required", "\n".join(result.messages))

    def test_vllm_smoke_stop_sends_sigterm_to_process_group(self) -> None:
        import signal
        from modules.vllm_runner import stop_vllm_smoke

        calls: list[tuple] = []

        result = stop_vllm_smoke(
            pid="1234",
            run_id="run-a",
            confirmed=True,
            getpgid_func=lambda pid: calls.append(("getpgid", pid)) or 9876,
            killpg_func=lambda pgid, sig: calls.append(("killpg", pgid, sig)),
            group_members_func=lambda pgid: [],
            post_signal_delay=0,
        )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(calls, [("getpgid", 1234), ("killpg", 9876, signal.SIGTERM)])
        self.assertIn("SIGTERM sent", "\n".join(result.messages))

    def test_vllm_smoke_stop_reports_residual_process_group_members(self) -> None:
        from modules.vllm_runner import stop_vllm_smoke

        result = stop_vllm_smoke(
            pid=1234,
            run_id="run-a",
            confirmed=True,
            getpgid_func=lambda pid: 9876,
            killpg_func=lambda pgid, sig: None,
            group_members_func=lambda pgid: [1234, 5678],
            post_signal_delay=0,
        )

        self.assertTrue(result.ok, result.messages)
        self.assertIn("still has live pids", "\n".join(result.messages))
        self.assertIn("5678", "\n".join(result.messages))

    def test_vllm_smoke_stop_signal_errors_return_structured_failure(self) -> None:
        from modules.vllm_runner import stop_vllm_smoke

        def raise_os_error(_pgid, _sig):
            raise OSError("permission denied")

        result = stop_vllm_smoke(
            pid=1234,
            run_id="run-a",
            confirmed=True,
            getpgid_func=lambda pid: 9876,
            killpg_func=raise_os_error,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.pid, 1234)
        self.assertIn("permission denied", "\n".join(result.messages))

    def test_script_generation_action_is_unified(self) -> None:
        launcher = load_launcher_module()
        draft = self.sample_cfg("/bin/echo")
        draft.update(
            {
                "model_name": "Dummy-7B",
                "model_path": "/models/Dummy-7B/model.gguf",
            }
        )

        from io import StringIO
        import contextlib
        from unittest.mock import patch

        stdout = StringIO()
        with patch("builtins.input", return_value="1"), contextlib.redirect_stdout(stdout):
            action = launcher.choose_script_generation_action(draft)
        text = stdout.getvalue()

        self.assertEqual(action, "create")
        self.assertIn("[G] 새 스크립트 생성이 선택되어 있습니다.", text)
        self.assertIn("[1] 생성만", text)
        self.assertIn("[2] 생성 후 실행", text)

    def test_last_run_record_restores_unsaved_test_parameters(self) -> None:
        launcher = load_launcher_module()
        with TemporaryDirectory() as directory:
            record_path = Path(directory) / "last-run.json"
            model_path = Path(directory) / "Dummy-7B.gguf"
            model_path.write_text("")
            draft = self.sample_cfg("/bin/echo")
            draft.update(
                {
                    "model_name": "Dummy-7B",
                    "model_path": str(model_path),
                    "ctx_size": 12345,
                    "custom_args": ["--no-warmup"],
                }
            )

            ok, message = launcher.write_last_run_record(draft, "one_time_run", path=record_path)
            restored: dict = {}
            loaded, loaded_message = launcher.load_last_run_record({"Dummy-7B": str(model_path)}, restored, path=record_path)

        self.assertTrue(ok, message)
        self.assertTrue(loaded, loaded_message)
        self.assertEqual(restored["model_name"], "Dummy-7B")
        self.assertEqual(restored["ctx_size"], 12345)
        self.assertEqual(restored["custom_args"], ["--no-warmup"])
        self.assertTrue(restored["dirty"])
        self.assertIn("마지막 실행 기록", restored["status"])

    def test_last_run_record_tolerates_bad_numeric_values(self) -> None:
        launcher = load_launcher_module()
        with TemporaryDirectory() as directory:
            record_path = Path(directory) / "last-run.json"
            model_path = Path(directory) / "Dummy-7B.gguf"
            model_path.write_text("")
            record_path.write_text(
                json.dumps(
                    {
                        "schema": "llama-suite-last-run-v1",
                        "saved_at": "2026-05-11T00:00:00",
                        "draft": {
                            "model_name": "Dummy-7B",
                            "model_path": str(model_path),
                            "ctx_size": "bad",
                            "port": "bad",
                            "reasoning_budget": "bad",
                            "param_sources": ["bad"],
                        },
                    }
                )
            )
            restored: dict = {}
            loaded, message = launcher.load_last_run_record({"Dummy-7B": str(model_path)}, restored, path=record_path)

        self.assertTrue(loaded, message)
        self.assertEqual(restored["ctx_size"], 95000)
        self.assertEqual(restored["port"], 8080)
        self.assertEqual(restored["reasoning_budget"], 0)
        self.assertEqual(restored["param_sources"], {})


if __name__ == "__main__":
    unittest.main()
