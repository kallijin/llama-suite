import importlib.util
import os
import subprocess
import sys
import unittest
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
        self.assertIn("[B] vLLM profile", completed.stdout)
        self.assertIn("[V] vLLM doctor", completed.stdout)
        self.assertIn("[A] 설정 변경 / 현재 설정 저장", completed.stdout)
        self.assertNotIn("\n  [W] 현재 설정 저장", completed.stdout)
        self.assertNotIn("[X] 새 스크립트 생성 후 실행", completed.stdout)

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
        self.assertEqual(validate_vllm_profile(profile), [])

    def test_vllm_profile_validation_reports_structured_messages(self) -> None:
        from modules.vllm_profiles import VllmProfile, validate_vllm_profile

        profile = VllmProfile(
            wrapper_path="",
            model="",
            port=70000,
            gpu_memory_utilization=1.5,
            tensor_parallel_size=0,
            max_model_len=0,
        )

        errors = validate_vllm_profile(profile)

        self.assertIn("wrapper path should not be empty", errors)
        self.assertIn("model should not be empty", errors)
        self.assertIn("port should be 1-65535", errors)
        self.assertIn("gpu_memory_utilization should be between 0 and 1", errors)
        self.assertIn("tensor_parallel_size should be >= 1", errors)
        self.assertIn("max_model_len should be > 0", errors)

    def test_vllm_profile_validation_handles_non_numeric_user_values(self) -> None:
        from modules.vllm_profiles import VllmProfile, validate_vllm_profile

        profile = VllmProfile(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            port="abc",  # type: ignore[arg-type]
            gpu_memory_utilization="bad",  # type: ignore[arg-type]
            tensor_parallel_size="many",  # type: ignore[arg-type]
            max_model_len="long",  # type: ignore[arg-type]
        )

        errors = validate_vllm_profile(profile)

        self.assertIn("port should be 1-65535", errors)
        self.assertIn("gpu_memory_utilization should be between 0 and 1", errors)
        self.assertIn("tensor_parallel_size should be >= 1", errors)
        self.assertIn("max_model_len should be > 0", errors)

    def test_vllm_profile_from_dict_preserves_extra_args_as_opaque_string(self) -> None:
        from modules.vllm_profiles import vllm_profile_from_dict

        profile = vllm_profile_from_dict(
            {
                "model": "local-model",
                "port": "8001",
                "max_model_len": "8192",
                "gpu_memory_utilization": "0.55",
                "tensor_parallel_size": "2",
                "extra_args": "--trust-remote-code --served-model-name local",
            }
        )

        self.assertEqual(profile.port, 8001)
        self.assertEqual(profile.max_model_len, 8192)
        self.assertEqual(profile.gpu_memory_utilization, 0.55)
        self.assertEqual(profile.tensor_parallel_size, 2)
        self.assertEqual(profile.extra_args, "--trust-remote-code --served-model-name local")

    def test_vllm_command_preview_builds_expected_command_list(self) -> None:
        from modules.vllm_profiles import VllmProfile, build_vllm_command

        profile = VllmProfile(model="Qwen/Qwen2.5-0.5B-Instruct")

        command, messages = build_vllm_command(profile)

        self.assertEqual(messages, [])
        self.assertEqual(
            command,
            [
                "~/bin/vllm-rocm",
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
            ],
        )

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
        from modules.vllm_profiles import host_guidance_lines

        text = "\n".join(host_guidance_lines())

        self.assertIn("127.0.0.1 = local only", text)
        self.assertIn("Tailscale IP = private remote access", text)
        self.assertIn("0.0.0.0 = advanced/exposed", text)

    def test_vllm_profile_preview_is_read_only_and_separate(self) -> None:
        launcher = load_launcher_module()

        text = launcher.vllm_profile_preview_text()

        self.assertIn("vLLM profile preview (read-only)", text)
        self.assertIn("vLLM 전용 profile", text)
        self.assertIn("llama.cpp 파라미터와 별개", text)
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
