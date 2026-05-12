import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from modules.model_registry import (
    ModelRegistry,
    load_model_registry,
    registered_model_from_candidate,
    registered_model_summary_line,
    save_model_registry,
    upsert_registered_model,
)
from modules.vllm_model_scan import (
    render_readiness_human_lines,
    render_vllm_model_candidate_lines,
    render_unregistered_candidate_summary_line,
    scan_vllm_model_candidates,
)


class VllmModelScanTests(unittest.TestCase):
    def test_awq_folder_name_is_detected(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "Qwen2.5-14B-Instruct-AWQ"
            self._write_local_hf_model(model_dir)

            cache = scan_vllm_model_candidates([directory], scanned_at="2026-05-11T00:00:00+00:00")

        candidate = self._only_candidate(cache.candidates)
        self.assertEqual(candidate.candidate_backend, "vllm")
        self.assertEqual(candidate.classification_guess.quant, "awq")
        self.assertEqual(candidate.classification_guess.size_b, 14)
        self.assertIn("path_contains_awq", candidate.classification_guess.evidence)
        self.assertIn("safetensors_weights_found", candidate.classification_guess.evidence)
        self.assertEqual(candidate.readiness.state, "needs_registration")
        self.assertFalse(candidate.readiness.blocking)

    def test_missing_tokenizer_is_needs_files(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "Qwen2.5-14B-Instruct-AWQ"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2", "quantization_config": {"quant_method": "awq"}}))
            (model_dir / "model.safetensors").write_text("fake")

            cache = scan_vllm_model_candidates([directory])

        candidate = self._only_candidate(cache.candidates)
        self.assertEqual(candidate.readiness.state, "needs_files")
        self.assertIn("tokenizer", candidate.readiness.missing)
        self.assertTrue(candidate.readiness.blocking)
        self.assertEqual(
            render_readiness_human_lines(candidate.readiness),
            ["상태: 파일 보완 필요", "다음 파일이 필요합니다: tokenizer", "실행 가능 여부: 불가"],
        )

    def test_folder_name_guess_survives_missing_files_and_renders_human_status(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "alonsoko__gemma-4-31b-it-abliterated-heretic-ara-AWQ"
            model_dir.mkdir()

            cache = scan_vllm_model_candidates([directory])

        candidate = self._only_candidate(cache.candidates)
        self.assertEqual(candidate.classification_guess.quant, "awq")
        self.assertEqual(candidate.classification_guess.family, "gemma")
        self.assertEqual(candidate.classification_guess.size_b, 31)
        self.assertEqual(candidate.readiness.state, "needs_files")
        self.assertEqual(candidate.readiness.missing, ["config", "tokenizer", "weights"])
        lines = "\n".join(render_vllm_model_candidate_lines(cache))
        self.assertIn("추정: gemma / 31B / AWQ", lines)
        self.assertIn("상태: 파일 보완 필요", lines)
        self.assertIn("다음 파일이 필요합니다: config, tokenizer, weights", lines)
        self.assertIn("실행 가능 여부: 불가", lines)

    def test_suite_profile_hint_is_shown_separately_from_readiness(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "Qwen2.5-14B-Instruct-AWQ"
            self._write_local_hf_model(model_dir)
            (model_dir / "llama-suite-vllm-profile.json").write_text("{}")

            cache = scan_vllm_model_candidates([directory])

        candidate = self._only_candidate(cache.candidates)
        self.assertTrue(candidate.has_suite_profile)
        lines = "\n".join(render_vllm_model_candidate_lines(cache))
        self.assertIn("suite profile OK", lines)

    def test_gguf_routes_to_llama_cpp_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            gguf = Path(directory) / "hyperclovax-32b-Q4_K_M.gguf"
            gguf.write_text("fake")

            cache = scan_vllm_model_candidates([directory])

        candidate = self._only_candidate(cache.candidates)
        self.assertEqual(candidate.candidate_backend, "llama.cpp")
        self.assertEqual(candidate.source.kind, "gguf_file")
        self.assertEqual(candidate.classification_guess.quant, "llama")
        self.assertEqual(candidate.classification_guess.format, "gguf")

    def test_human_display_string_is_not_stored_schema(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "Qwen2.5-14B-Instruct-AWQ"
            self._write_local_hf_model(model_dir)
            candidate = self._only_candidate(scan_vllm_model_candidates([directory]).candidates)
            model = registered_model_from_candidate(candidate, alias="qwen2.5-14b-awq")

        candidate_payload = candidate.to_dict()
        model_payload = model.to_dict()
        self.assertNotIn("display_name", json.dumps(candidate_payload))
        self.assertNotIn("display_name", json.dumps(model_payload))
        self.assertNotIn("설정파일", json.dumps(model_payload, ensure_ascii=False))
        self.assertIn("AWQ", render_unregistered_candidate_summary_line(candidate))
        self.assertIn("qwen2.5-14b-awq", registered_model_summary_line(model))

    def test_registry_load_save_roundtrip(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "cyankiwi-gemma-4-26B-A4B-it-AWQ-4bit"
            self._write_local_hf_model(model_dir, model_type="gemma")
            candidate = self._only_candidate(scan_vllm_model_candidates([directory]).candidates)
            model = registered_model_from_candidate(candidate, alias="gemma4-26b-awq")
            registry_path = Path(directory) / "registry.json"

            save_result = save_model_registry(ModelRegistry(models=[model]), registry_path=registry_path)
            load_result = load_model_registry(registry_path=registry_path)

        self.assertTrue(save_result.ok, save_result.messages)
        self.assertTrue(load_result.ok, load_result.messages)
        self.assertIsNotNone(load_result.registry)
        loaded = load_result.registry.models[0]
        self.assertEqual(loaded.backend, "vllm")
        self.assertEqual(loaded.source["path"], str(model_dir))
        self.assertEqual(loaded.classification["quant"], "awq")
        self.assertEqual(loaded.readiness["state"], "ready")
        self.assertEqual(loaded.human["alias"], "gemma4-26b-awq")

    def test_registry_upsert_replaces_duplicate_path(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "Qwen2.5-14B-Instruct-AWQ"
            self._write_local_hf_model(model_dir)
            candidate = self._only_candidate(scan_vllm_model_candidates([directory]).candidates)
            first = registered_model_from_candidate(candidate, alias="qwen-awq")
            second = registered_model_from_candidate(candidate, alias="qwen-awq-renamed")

        registry = upsert_registered_model(ModelRegistry(models=[first]), second)

        self.assertEqual(len(registry.models), 1)
        self.assertEqual(registry.models[0].human["alias"], "qwen-awq-renamed")

    def _write_local_hf_model(self, model_dir: Path, *, model_type: str = "qwen2") -> None:
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": model_type, "quantization_config": {"quant_method": "awq"}}))
        (model_dir / "tokenizer.json").write_text("{}")
        (model_dir / "model.safetensors").write_text("fake")

    def _only_candidate(self, candidates):
        self.assertEqual(len(candidates), 1)
        return candidates[0]


if __name__ == "__main__":
    unittest.main()
