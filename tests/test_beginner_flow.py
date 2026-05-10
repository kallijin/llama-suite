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
        self.assertIn("GGUF 파일을 찾을 수 없습니다", completed.stdout)
        self.assertIn("[A] 설정 변경", completed.stdout)
        self.assertIn("[E] Hermes 등록", completed.stdout)
        self.assertIn("Hermes 설정 변경: 비활성화", completed.stdout)
        self.assertIn("실행 예정 요약", completed.stdout)
        self.assertIn("Recent vLLM run:", completed.stdout)
        self.assertIn("Selected vLLM profile: custom-draft / (empty model) / http://127.0.0.1:8000/v1", completed.stdout)
        self.assertIn("[K] llama.cpp 파라미터", completed.stdout)
        self.assertIn("[P] llama.cpp 최종 미리보기", completed.stdout)
        self.assertIn("[O] llama.cpp 1회 실행", completed.stdout)
        self.assertIn("[G] llama.cpp 새 스크립트 생성", completed.stdout)
        self.assertIn("[B] vLLM profile", completed.stdout)
        self.assertIn("[W] vLLM API smoke", completed.stdout)
        self.assertIn("[Y] vLLM smoke launch", completed.stdout)
        self.assertIn("[Z] vLLM latest run status/log/stop", completed.stdout)
        self.assertIn("[V] vLLM doctor", completed.stdout)
        self.assertIn("[A] 설정 변경 / 현재 설정 저장", completed.stdout)
        self.assertNotIn("\n  [W] 현재 설정 저장", completed.stdout)
        self.assertNotIn("[X] 새 스크립트 생성 후 실행", completed.stdout)

    def test_recent_vllm_run_summary_line_renders_no_record(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_runner import VllmLatestRunSummary

        line = launcher.recent_vllm_run_summary_line(VllmLatestRunSummary(False, None, None, None, "UNKNOWN", ["no record"]))

        self.assertEqual(line, "  Recent vLLM run: no run record")

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

    def test_selected_vllm_profile_summary_line_renders_current_draft(self) -> None:
        launcher = load_launcher_module()
        from modules.vllm_profiles import VllmProfile

        line = launcher.selected_vllm_profile_summary_line(
            VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct", host="100.64.1.2", port=8010),
            "tailscale-qwen",
        )

        self.assertEqual(
            line,
            "  Selected vLLM profile: tailscale-qwen / Qwen/Qwen2.5-0.5B-Instruct / http://100.64.1.2:8010/v1",
        )

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

    def test_vllm_builtin_preset_registry_includes_default_and_smoke(self) -> None:
        from modules.vllm_profiles import builtin_vllm_profile_presets, future_launch_preset_id

        presets = builtin_vllm_profile_presets()
        by_id = {preset.id: preset for preset in presets}

        self.assertIn("default", by_id)
        self.assertIn("smoke-qwen-0.5b", by_id)
        self.assertEqual(by_id["default"].label, "Default vLLM profile")
        self.assertEqual(by_id["smoke-qwen-0.5b"].label, "Smoke Qwen 0.5B")
        self.assertIn("read-only", by_id["smoke-qwen-0.5b"].description)
        self.assertEqual(by_id["smoke-qwen-0.5b"].profile.model, "Qwen/Qwen2.5-0.5B-Instruct")
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
        self.assertIn("max_model_len should be > 0", errors)
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
        self.assertIn("max_model_len should be > 0", errors)
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
        from modules.vllm_profiles import editable_vllm_profile_fields

        fields = editable_vllm_profile_fields()

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
        from modules.vllm_profiles import VllmProfile

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
            save_vllm_profile_draft(VllmProfile(model="Local/ThirtyB", port="bad"), profile_id="30b candidate", store_root=directory)  # type: ignore[arg-type]
            result = list_vllm_profile_drafts(store_root=directory)
            missing = list_vllm_profile_drafts(store_root=Path(directory) / "missing")

        self.assertTrue(result.ok, result.messages)
        by_id = {profile.profile_id: profile for profile in result.profiles}
        self.assertIn("smoke", by_id)
        self.assertIn("30b candidate", by_id)
        self.assertEqual(by_id["smoke"].model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("port should be 1-65535", by_id["30b candidate"].validation_messages)
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
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.7",
                "--tensor-parallel-size",
                "1",
                "--kv-cache-dtype",
                "auto",
            ],
        )

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
        self.assertIn("no launch button", confirmation)
        large_model = "\n".join(large_model_guidance_lines())
        self.assertIn("/mnt/data_main/downloads/models", large_model)
        self.assertIn("30B-36B vLLM", large_model)
        self.assertIn("HF/safetensors", large_model)
        self.assertIn("AWQ/GPTQ/Int4", large_model)
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

        self.assertIn("vLLM profile preview (read-only)", text)
        self.assertIn("vLLM 전용 profile", text)
        self.assertIn("llama.cpp 파라미터와 별개", text)
        self.assertIn("Selected for future launch: smoke-qwen-0.5b", text)
        self.assertIn("Read-only placeholder only", text)
        self.assertIn("launch 선택 상태는 아직 구현하지 않았습니다", text)
        self.assertIn("Future launch confirmation wording:", text)
        self.assertIn("download model files", text)
        self.assertIn("GPU memory", text)
        self.assertIn("torch compile / graph capture", text)
        self.assertIn("host/port will be bound", text)
        self.assertIn("Available built-in vLLM profiles:", text)
        self.assertIn("default: Default vLLM profile", text)
        self.assertIn("smoke-qwen-0.5b: Smoke Qwen 0.5B", text)
        self.assertIn("현재는 read-only registry", text)
        self.assertIn("Preset default: Default vLLM profile", text)
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
        self.assertIn("30B / quantized model guidance:", text)
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
        self.assertIn("In-memory only", text)
        self.assertIn("llama.cpp 설정과 별개", text)
        self.assertIn("Editable vLLM fields:", text)
        self.assertIn("- model", text)
        self.assertIn("Command preview / dry-run", text)
        self.assertIn("Launch preflight:", text)
        self.assertIn("[PASS] profile validation", text)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("30B / quantized model guidance:", text)
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

    def test_vllm_profile_menu_can_save_and_load_custom_profile_draft(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        loaded_profile = VllmProfile(model="loaded-model")
        save_result = VllmProfileStoreResult(True, profile, "/tmp/custom-draft.json", ["saved"])
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/custom-draft.json", ["loaded"])
        mocked_save = Mock(return_value=save_result)
        mocked_load = Mock(return_value=load_result)

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", "named-profile"]), contextlib.redirect_stdout(StringIO()):
            after_save = launcher.show_vllm_profile_menu(profile)
        with patch.object(launcher, "load_vllm_profile_draft", mocked_load), patch("builtins.input", side_effect=["5", "named-profile"]), contextlib.redirect_stdout(StringIO()):
            after_load = launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile, profile_id="named-profile")
        mocked_load.assert_called_once_with(profile_id="named-profile")
        self.assertIs(after_save, profile)
        self.assertEqual(after_load.model, "loaded-model")

    def test_vllm_profile_menu_tracks_selected_profile_id(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        loaded_profile = VllmProfile(model="loaded-model")
        save_result = VllmProfileStoreResult(True, profile, "/tmp/30b-q4.json", ["saved"])
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/30b-q4.json", ["loaded"])
        mocked_save = Mock(return_value=save_result)
        mocked_load = Mock(return_value=load_result)
        stdout = StringIO()

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", "30b-q4"]), contextlib.redirect_stdout(stdout):
            saved_profile, saved_id = launcher.show_vllm_profile_menu(profile, "custom-draft", return_profile_id=True)
        with patch.object(launcher, "load_vllm_profile_draft", mocked_load), patch("builtins.input", side_effect=["5", "30b-q4"]), contextlib.redirect_stdout(stdout):
            loaded, loaded_id = launcher.show_vllm_profile_menu(profile, saved_id, return_profile_id=True)

        self.assertIs(saved_profile, profile)
        self.assertEqual(saved_id, "30b-q4")
        self.assertEqual(loaded.model, "loaded-model")
        self.assertEqual(loaded_id, "30b-q4")
        self.assertIn("selected custom profile: custom-draft", stdout.getvalue())
        self.assertIn("selected custom profile: 30b-q4", stdout.getvalue())

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
        self.assertIn("Built-in / preview", output)
        self.assertIn("Custom draft", output)
        self.assertIn("Script / launch", output)

    def test_vllm_profile_menu_save_and_load_default_profile_id(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profile_store import VllmProfileStoreResult
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import Mock, patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        store_result = VllmProfileStoreResult(True, profile, "/tmp/custom-draft.json", ["ok"])
        mocked_save = Mock(return_value=store_result)
        mocked_load = Mock(return_value=store_result)

        with patch.object(launcher, "save_vllm_profile_draft", mocked_save), patch("builtins.input", side_effect=["4", ""]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_profile_menu(profile)
        with patch.object(launcher, "load_vllm_profile_draft", mocked_load), patch("builtins.input", side_effect=["5", ""]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile, profile_id="custom-draft")
        mocked_load.assert_called_once_with(profile_id="custom-draft")

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
                VllmStoredProfileInfo("30b", "/tmp/30b.json", "Local/ThirtyB", ["port should be 1-65535"]),
            ],
            "/tmp/profiles",
            [],
        )
        mocked_list = Mock(return_value=list_result)
        stdout = StringIO()

        with patch.object(launcher, "list_vllm_profile_drafts", mocked_list), patch("builtins.input", side_effect=["9"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile)

        mocked_list.assert_called_once_with()
        self.assertIs(result, profile)
        output = stdout.getvalue()
        self.assertIn("vLLM saved custom profiles", output)
        self.assertIn("[1] smoke: Qwen/Qwen2.5-0.5B-Instruct [valid]", output)
        self.assertIn("[2] 30b: Local/ThirtyB [needs attention]", output)
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
                VllmStoredProfileInfo("30b-q4", "/tmp/30b-q4.json", "Local/ThirtyB", []),
            ],
            "/tmp/profiles",
            [],
        )
        load_result = VllmProfileStoreResult(True, loaded_profile, "/tmp/30b-q4.json", ["loaded"])
        mocked_list = Mock(return_value=list_result)
        mocked_load = Mock(return_value=load_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", mocked_list),
            patch.object(launcher, "load_vllm_profile_draft", mocked_load),
            patch("builtins.input", side_effect=["10", "2"]),
            contextlib.redirect_stdout(stdout),
        ):
            loaded, loaded_id = launcher.show_vllm_profile_menu(profile, "custom-draft", return_profile_id=True)

        mocked_list.assert_called_once_with()
        mocked_load.assert_called_once_with(profile_id="30b-q4")
        self.assertEqual(loaded.model, "loaded-model")
        self.assertEqual(loaded_id, "30b-q4")
        self.assertIn("[10] load saved custom profile from list", stdout.getvalue())

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
            patch("builtins.input", side_effect=["10", "bad"]),
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
            [VllmStoredProfileInfo("30b-q4", "/tmp/30b-q4.json", "Local/ThirtyB", [])],
            "/tmp/profiles",
            [],
        )
        delete_result = VllmProfileStoreResult(True, None, "/tmp/30b-q4.json", ["deleted"])
        mocked_delete = Mock(return_value=delete_result)
        stdout = StringIO()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", Mock(return_value=list_result)),
            patch.object(launcher, "delete_vllm_profile_draft", mocked_delete),
            patch("builtins.input", side_effect=["11", "1", "delete"]),
            contextlib.redirect_stdout(stdout),
        ):
            returned_profile, returned_id = launcher.show_vllm_profile_menu(profile, "30b-q4", return_profile_id=True)

        mocked_delete.assert_called_once_with(profile_id="30b-q4", confirmed=True)
        self.assertIs(returned_profile, profile)
        self.assertEqual(returned_id, "custom-draft")
        self.assertIn("[11] delete saved custom profile from list", stdout.getvalue())

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
            [VllmStoredProfileInfo("30b-q4", "/tmp/30b-q4.json", "Local/ThirtyB", [])],
            "/tmp/profiles",
            [],
        )
        mocked_delete = Mock()

        with (
            patch.object(launcher, "list_vllm_profile_drafts", Mock(return_value=list_result)),
            patch.object(launcher, "delete_vllm_profile_draft", mocked_delete),
            patch("builtins.input", side_effect=["11", "1", "no"]),
            contextlib.redirect_stdout(StringIO()),
        ):
            returned_profile, returned_id = launcher.show_vllm_profile_menu(profile, "30b-q4", return_profile_id=True)

        mocked_delete.assert_not_called()
        self.assertIs(returned_profile, profile)
        self.assertEqual(returned_id, "30b-q4")

    def test_vllm_profile_menu_can_preview_custom_script(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
        from unittest.mock import patch

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")
        stdout = StringIO()

        with patch("builtins.input", side_effect=["6"]), contextlib.redirect_stdout(stdout):
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

        with patch.object(launcher, "save_vllm_script", mocked_save), patch("builtins.input", side_effect=["7"]), contextlib.redirect_stdout(stdout):
            result = launcher.show_vllm_profile_menu(profile)

        mocked_save.assert_called_once_with(profile)
        self.assertIs(result, profile)
        self.assertIn("vLLM custom script save", stdout.getvalue())
        self.assertIn("/tmp/vllm.sh", stdout.getvalue())

    def test_vllm_profile_menu_custom_launch_requires_typed_confirmation(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_profiles import VllmProfile
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
                "host": None,
                "port": None,
                "command": [],
                "messages": ["cancelled"],
            },
        )()
        mocked_launch = Mock(return_value=launch_result)

        with patch.object(launcher, "launch_vllm_profile_once", mocked_launch), patch("builtins.input", side_effect=["8", "no"]), contextlib.redirect_stdout(StringIO()):
            result = launcher.show_vllm_profile_menu(profile, "30b-q4")

        mocked_launch.assert_called_once_with(profile, confirmed=False, preset_id="30b-q4")
        self.assertIs(result, profile)

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
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertNotIn("read-only registry", text)
        self.assertNotIn("no launch button", text)

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

    def test_vllm_custom_profile_launch_uses_runner_and_run_record(self) -> None:
        from modules.vllm_profiles import VllmPreflightCheck, VllmProfile
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
            )
            result = launch_vllm_profile_once(
                profile,
                confirmed=True,
                preset_id="custom-qwen",
                timestamp="20260510-224500",
                state_root=root / "runs",
                port_check=lambda host, port: VllmPreflightCheck(
                    "port availability",
                    True,
                    f"port {port} is available on {host}",
                ),
                popen_factory=fake_popen,
            )
            record = read_vllm_run_record(str(result.record_path))

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

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_smoke_status", mocked_status), patch("builtins.input", side_effect=["1", ""]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="1234", run_id="run-latest", log_path="/tmp/latest.log", host="100.68.40.87", port=8010)
        self.assertIn("latest run record", stdout.getvalue())

    def test_vllm_smoke_manage_missing_latest_falls_back_to_manual_status(self) -> None:
        launcher = load_launcher_module()
        from io import StringIO
        import contextlib
        from modules.vllm_runner import VllmRunRecordResult
        from unittest.mock import Mock, patch

        latest = VllmRunRecordResult(False, None, None, ["no vLLM run records found under /tmp/runs"])
        mocked_status = Mock(return_value=type("Status", (), {"ok": True, "preset_id": "smoke-qwen-0.5b", "pid": 1234, "run_id": "manual-run", "log_path": "/tmp/manual.log", "alive": True, "log_exists": False, "port_listening": None, "messages": []})())
        stdout = StringIO()

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_smoke_status", mocked_status), patch("builtins.input", side_effect=["1", "", "1234", "manual-run", "/tmp/manual.log", "127.0.0.1", "8000"]), contextlib.redirect_stdout(stdout):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="1234", run_id="manual-run", log_path="/tmp/manual.log", host="127.0.0.1", port="8000")
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

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "check_vllm_smoke_status", mocked_status), patch("builtins.input", side_effect=["1", "-", "2222", "manual-run", "/tmp/manual.log", "0.0.0.0", "8020"]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_status.assert_called_once_with(pid="2222", run_id="manual-run", log_path="/tmp/manual.log", host="0.0.0.0", port="8020")

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

        with patch.object(launcher, "latest_vllm_run_record", return_value=latest), patch.object(launcher, "stop_vllm_smoke", mocked_stop), patch("builtins.input", side_effect=["3", "", "no"]), contextlib.redirect_stdout(StringIO()):
            launcher.show_vllm_smoke_manage()

        mocked_stop.assert_called_once_with(pid="1234", run_id="run-latest", confirmed=False)

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
        )

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(calls, [("getpgid", 1234), ("killpg", 9876, signal.SIGTERM)])
        self.assertIn("SIGTERM sent", "\n".join(result.messages))

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


if __name__ == "__main__":
    unittest.main()
