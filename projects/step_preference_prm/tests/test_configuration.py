from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.input_packing import CausalRecentPrefixPacker  # noqa: E402
from prm_pref.training.configuration import (  # noqa: E402
    training_signature,
    validate_training_config,
    with_experiment_overrides,
    with_pilot_profile,
    with_smoke_profile,
)
from prm_pref.utils.config import load_config  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        del pair
        return 1

    def prepare_for_model(self, ids: list[int], **kwargs) -> dict:
        del kwargs
        return {"input_ids": [1] + ids}


class ConfigurationTests(unittest.TestCase):
    def test_checked_in_configs_are_valid_and_signatures_are_stable(self) -> None:
        config_paths = [
            PROJECT_ROOT / "configs/train_pointwise.yaml",
            PROJECT_ROOT / "configs/train_pairwise.yaml",
            PROJECT_ROOT / "configs/train_hybrid.yaml",
            PROJECT_ROOT
            / "experiments/qwen_lora_main/train_pointwise.yaml",
            PROJECT_ROOT
            / "experiments/qwen_lora_main/train_pairwise.yaml",
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml",
        ]
        for path in config_paths:
            config = load_config(path)
            validate_training_config(config)
            if config.get("model", {}).get("backend") == "causal_lora":
                self.assertEqual(
                    config["model"]["lora"]["bias"],
                    "none",
                )
            self.assertEqual(
                training_signature(config, lambda_pair=0.3),
                training_signature(config, lambda_pair=0.3),
            )

    def test_pilot_keeps_production_sequence_length(self) -> None:
        config = load_config(
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml"
        )
        pilot = with_pilot_profile(config)
        self.assertEqual(pilot["model"]["max_length"], 2048)
        self.assertEqual(pilot["training"]["epochs"], 1)
        self.assertEqual(pilot["training"]["max_steps_per_epoch"], 200)
        self.assertTrue(pilot["output_dir"].endswith("_pilot"))

    def test_experiment_overrides_are_explicit_and_non_mutating(self) -> None:
        config = load_config(
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml"
        )
        overridden = with_experiment_overrides(
            config,
            seed=7,
            max_train_nodes=4785,
        )
        self.assertEqual(overridden["training"]["seed"], 7)
        self.assertEqual(overridden["data"]["max_train_nodes"], 4785)
        self.assertEqual(config["training"]["seed"], 42)

    def test_smoke_profile_drops_to_fp32_only_off_cuda(self) -> None:
        config = load_config(
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml"
        )
        cpu = with_smoke_profile(
            with_experiment_overrides(config, device="cpu")
        )
        self.assertEqual(cpu["model"]["load_dtype"], "fp32")
        self.assertEqual(cpu["training"]["mixed_precision"], "fp32")
        self.assertFalse(cpu["model"]["gradient_checkpointing"])

        cuda = with_smoke_profile(
            with_experiment_overrides(config, device="cuda")
        )
        self.assertEqual(cuda["model"]["load_dtype"], "bf16")
        self.assertEqual(cuda["training"]["mixed_precision"], "bf16")

        # The production config must stay untouched by either profile.
        self.assertEqual(config["model"]["load_dtype"], "bf16")
        self.assertTrue(config["model"]["gradient_checkpointing"])

    def test_backbone_override_clears_the_pinned_revision(self) -> None:
        config = load_config(
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml"
        )
        config.setdefault("model", {})["revision"] = "deadbeef"
        overridden = with_experiment_overrides(
            config,
            model_name="Qwen/Qwen2.5-0.5B",
        )
        self.assertEqual(
            overridden["model"]["name_or_path"],
            "Qwen/Qwen2.5-0.5B",
        )
        self.assertNotIn("revision", overridden["model"])
        # A substituted backbone must change the resume signature.
        self.assertNotEqual(
            training_signature(config, lambda_pair=0.3),
            training_signature(overridden, lambda_pair=0.3),
        )

    def test_invalid_device_override_is_rejected(self) -> None:
        config = load_config(
            PROJECT_ROOT / "experiments/qwen_lora_main/train_hybrid.yaml"
        )
        with self.assertRaises(ValueError):
            with_experiment_overrides(config, device="gpu0")

    def test_recent_prefix_packer_truncates_prefix_before_candidate(self) -> None:
        packer = CausalRecentPrefixPacker(
            FakeTokenizer(),
            max_length=120,
            max_problem_tokens=20,
            max_candidate_tokens=20,
        )
        measured = packer.measure(
            {
                "problem": "p" * 100,
                "prefix": ["old" * 20, "recent" * 20],
                "candidate": "candidate",
            }
        )
        self.assertEqual(measured["candidate_truncated"], 0)
        self.assertEqual(measured["problem_truncated"], 1)
        self.assertEqual(measured["prefix_truncated"], 1)
        self.assertLessEqual(measured["tokens"], 120)


if __name__ == "__main__":
    unittest.main()
