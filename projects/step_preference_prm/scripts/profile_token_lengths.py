#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from prm_pref.data.datasets import IndexedJsonlDataset  # noqa: E402
from prm_pref.data.input_packing import CausalRecentPrefixPacker  # noqa: E402
from prm_pref.utils.config import load_config  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: list[int], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[int]) -> dict:
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile Qwen candidate lengths and recent-prefix truncation."
    )
    parser.add_argument(
        "--config",
        default="experiments/qwen_lora_main/train_hybrid.yaml",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/profiles/qwen_lora_token_lengths.json",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    model_cfg = dict(config.get("model", {}))
    if model_cfg.get("backend") != "causal_lora":
        raise ValueError("Token-length profiling expects a causal_lora config")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        use_fast=True,
        revision=model_cfg.get("revision"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    packing_cfg = dict(model_cfg.get("packing", {}))
    packer = CausalRecentPrefixPacker(
        tokenizer,
        max_length=int(model_cfg["max_length"]),
        max_problem_tokens=int(packing_cfg.get("max_problem_tokens", 384)),
        max_candidate_tokens=int(
            packing_cfg.get("max_candidate_tokens", 512)
        ),
    )
    staged_data_root = os.environ.get("PRM_DATA_ROOT")
    nodes_dir = (
        Path(staged_data_root) / "nodes_v0"
        if staged_data_root
        else resolve_path(
            config.get("data", {}).get(
                "nodes_dir",
                "data/processed/nodes_v0",
            )
        )
    )
    nodes = IndexedJsonlDataset(
        nodes_dir / f"{args.split}.jsonl",
        max_examples=args.max_nodes,
        seed=int(config.get("training", {}).get("seed", 42)),
    )

    measurements: dict[str, list[int]] = {
        "raw_problem_tokens": [],
        "raw_prefix_tokens": [],
        "raw_candidate_tokens": [],
        "raw_total_tokens": [],
        "packed_tokens": [],
    }
    truncation = {
        "problem": 0,
        "prefix": 0,
        "candidate": 0,
    }
    for node in nodes:
        for candidate in node["candidates"]:
            measured = packer.measure(
                {
                    "problem": node["problem"],
                    "prefix": node.get("prefix", []),
                    "candidate": candidate["text"],
                }
            )
            for key in measurements:
                source_key = "tokens" if key == "packed_tokens" else key
                measurements[key].append(int(measured[source_key]))
            truncation["problem"] += measured["problem_truncated"]
            truncation["prefix"] += measured["prefix_truncated"]
            truncation["candidate"] += measured["candidate_truncated"]

    num_candidates = len(measurements["packed_tokens"])
    report = {
        "model": model_cfg["name_or_path"],
        "split": args.split,
        "max_length": int(model_cfg["max_length"]),
        "num_nodes": len(nodes),
        "num_candidates": num_candidates,
        "lengths": {
            key: summarize(values) for key, values in measurements.items()
        },
        "truncation": {
            key: {
                "count": count,
                "rate": count / num_candidates if num_candidates else 0.0,
            }
            for key, count in truncation.items()
        },
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
