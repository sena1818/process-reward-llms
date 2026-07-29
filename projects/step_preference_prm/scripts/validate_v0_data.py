#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.split_by_problem import SPLIT_NAMES, load_problem_split_map  # noqa: E402
from prm_pref.utils.config import load_config  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate materialized V0 data invariants.")
    parser.add_argument("--config", default="configs/data_v0.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)

    def path(key: str, default: str) -> Path:
        value = Path(config.get(key, default))
        return value if value.is_absolute() else PROJECT_ROOT / value

    split_path = path("problem_split_path", "data/processed/splits/problem_splits.json")
    split_map = load_problem_split_map(split_path)
    point_dir = path("pointwise_output_dir", "data/processed/pointwise")
    pair_dir = path("pairs_output_dir", "data/processed/pairs_v0")
    trajectory_dir = path("trajectory_output_dir", "data/processed/trajectories")
    nodes_dir = path("nodes_output_dir", "data/processed/nodes_v0")
    audit = read_json(PROJECT_ROOT / "outputs/audit/data_audit.json")
    pair_audit = read_json(PROJECT_ROOT / "outputs/audit/pair_stats.json")
    node_stats = read_json(nodes_dir / "stats.json")

    totals: Counter[str] = Counter()
    for split in SPLIT_NAMES:
        for line_no, item in iter_jsonl(point_dir / f"{split}.jsonl"):
            if item["label"] not in {0, 1} or item["rating"] not in {-1, 1}:
                raise ValueError(f"Non-V0 point label at {split}:{line_no}")
            if item["label"] != (1 if item["rating"] == 1 else 0):
                raise ValueError(f"Point label/rating mismatch at {split}:{line_no}")
            if split_map[item["problem_id"]] != split:
                raise ValueError(f"Point split leakage at {split}:{line_no}")
            totals["pointwise"] += 1
            totals[f"rating_{item['rating']:+d}"] += 1

        for line_no, item in iter_jsonl(pair_dir / f"{split}.jsonl"):
            if item["pair_type"] != "+1>-1" or float(item["weight"]) != 1.0:
                raise ValueError(f"Non-V0 pair at {split}:{line_no}")
            if split_map[item["problem_id"]] != split:
                raise ValueError(f"Pair split leakage at {split}:{line_no}")
            totals["pairs"] += 1

        for line_no, item in iter_jsonl(trajectory_dir / f"{split}.jsonl"):
            if split_map[item["problem_id"]] != split:
                raise ValueError(f"Trajectory split leakage at {split}:{line_no}")
            first_error = item.get("first_error_index")
            if first_error is not None and not 1 <= int(first_error) <= len(item["steps"]):
                raise ValueError(f"Invalid first-error index at {split}:{line_no}")
            totals["trajectories"] += 1
            totals["first_error_trajectories"] += first_error is not None

        for line_no, item in iter_jsonl(nodes_dir / f"{split}.jsonl"):
            candidates = item.get("candidates") or []
            labels = [int(candidate["label"]) for candidate in candidates]
            ratings = [int(candidate["rating"]) for candidate in candidates]
            if not candidates or set(labels) != {0, 1}:
                raise ValueError(
                    f"Strict node lacks both labels at {split}:{line_no}"
                )
            if any(
                label != (1 if rating == 1 else 0)
                for label, rating in zip(labels, ratings)
            ):
                raise ValueError(
                    f"Node label/rating mismatch at {split}:{line_no}"
                )
            ratings_by_text: dict[str, set[int]] = {}
            for candidate in candidates:
                ratings_by_text.setdefault(
                    str(candidate["text"]),
                    set(),
                ).add(int(candidate["rating"]))
            if any(
                {-1, 1}.issubset(text_ratings)
                for text_ratings in ratings_by_text.values()
            ):
                raise ValueError(
                    f"Conflicting binary text retained at {split}:{line_no}"
                )
            positives = sum(labels)
            negatives = len(labels) - positives
            derived_pairs = positives * negatives
            if (
                int(item["num_candidates"]) != len(candidates)
                or int(item["num_positive"]) != positives
                or int(item["num_negative"]) != negatives
                or int(item["num_pairs"]) != derived_pairs
            ):
                raise ValueError(
                    f"Node count metadata mismatch at {split}:{line_no}"
                )
            if split_map[item["problem_id"]] != split:
                raise ValueError(f"Node split leakage at {split}:{line_no}")
            totals["nodes"] += 1
            totals["node_candidates"] += len(candidates)
            totals["node_pairs"] += derived_pairs

    expected_positive = int(audit["label_distribution"]["+1"])
    expected_negative = int(audit["label_distribution"]["-1"])
    expected_pairs = int(pair_audit["pair_counts"]["+1>-1"])
    expected_first_errors = int(audit["first_error"]["solution_samples_with_first_error"])
    expected = {
        "pointwise": expected_positive + expected_negative,
        "rating_+1": expected_positive,
        "rating_-1": expected_negative,
        "pairs": expected_pairs,
        "first_error_trajectories": expected_first_errors,
        "nodes": sum(
            int(node_stats["counts"].get(f"nodes_{split}", 0))
            for split in SPLIT_NAMES
        ),
        "node_candidates": sum(
            int(node_stats["counts"].get(f"candidates_{split}", 0))
            for split in SPLIT_NAMES
        ),
        "node_pairs": sum(
            int(node_stats["counts"].get(f"derived_pairs_{split}", 0))
            for split in SPLIT_NAMES
        ),
    }
    for key, value in expected.items():
        if totals[key] != value:
            raise ValueError(f"Count mismatch for {key}: {totals[key]} != {value}")

    report = {"status": "ok", "verified_counts": dict(sorted(totals.items()))}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
