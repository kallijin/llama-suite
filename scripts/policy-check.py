#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config_store import (
    ensure_kv_cache_safety_args,
    has_control_char,
    normalize_extra_args,
)
from modules.script_builder import detect_model_size_billion, resolve_ctx_size


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def check_model_size_ctx_policy() -> None:
    cfg = {"ctx_size": 95000}
    cases = [
        (
            "supergemma4-26b-uncensored-fast-v2-Q4_K_M",
            "/models/supergemma4-26b-uncensored-fast-v2-Q4_K_M/model.gguf",
            26.0,
            92000,
        ),
        (
            "HyperCLOVAX-SEED-Think-32B-heretic2.Q4_K_M",
            "/models/HyperCLOVAX-SEED-Think-32B-heretic2.Q4_K_M/model.gguf",
            32.0,
            80000,
        ),
        (
            "Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.Q4_K_M",
            "/models/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated.Q4_K_M/model.gguf",
            35.0,
            80000,
        ),
        (
            "small-model-14B-Q4_K_M",
            "/models/small-model-14B-Q4_K_M/model.gguf",
            14.0,
            95000,
        ),
        (
            "unknown-size-model",
            "/models/unknown-size-model/model.gguf",
            None,
            95000,
        ),
    ]
    for name, path, expected_size, expected_ctx in cases:
        assert_equal(detect_model_size_billion(name, path), expected_size, f"size {name}")
        assert_equal(resolve_ctx_size(name, path, cfg), expected_ctx, f"ctx {name}")


def check_extra_args_policy() -> None:
    assert_equal(
        ensure_kv_cache_safety_args([]),
        ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on"],
        "empty safety args",
    )
    assert_equal(
        ensure_kv_cache_safety_args(["--temp", "0"]),
        [
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--flash-attn",
            "on",
            "--temp",
            "0",
        ],
        "prepend safety args",
    )
    assert_equal(
        ensure_kv_cache_safety_args(["--flash-attn", "--temp", "0"]),
        [
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--flash-attn",
            "on",
            "--temp",
            "0",
        ],
        "repair bare flash-attn without swallowing next option",
    )
    assert_equal(
        normalize_extra_args(
            [
                "--cache-type-k",
                "q8_0",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--flash-attn",
                "on",
                "--flash-attn",
                "on",
            ]
        ),
        ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on"],
        "dedupe safety args",
    )


def check_path_safety_policy() -> None:
    assert_equal(has_control_char("/tmp/llama-server"), False, "plain path")
    assert_equal(has_control_char("/tmp/llama\x1b[A-server"), True, "escape path")


def main() -> None:
    check_model_size_ctx_policy()
    check_extra_args_policy()
    check_path_safety_policy()
    print("POLICY CHECK OK")


if __name__ == "__main__":
    main()
