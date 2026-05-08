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
        self.assertIn("[W] 현재 설정 저장", text)

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


if __name__ == "__main__":
    unittest.main()
