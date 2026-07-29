from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.load_prm800k import (  # noqa: E402
    candidate_entries,
    problem_id_from_text,
    selected_step_text_and_rating,
)
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    IndexedJsonlDataset = None
else:
    from prm_pref.data.datasets import IndexedJsonlDataset  # noqa: E402
from prm_pref.data.materialize import (  # noqa: E402
    materialize_nodes_v0,
    materialize_pairs_v0,
    materialize_pointwise_v0,
)
from prm_pref.data.split_by_problem import (  # noqa: E402
    build_problem_splits,
    validate_problem_splits,
    write_problem_splits,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def record(problem: str) -> dict:
    return {
        "question": {
            "problem": problem,
            "pre_generated_steps": ["correct first step", "actual bad step"],
        },
        "label": {
            "finish_reason": "found_error",
            "steps": [
                {
                    "completions": [
                        {"text": "correct first step", "rating": 1, "flagged": False},
                        {"text": "wrong alternative", "rating": -1, "flagged": False},
                        {"text": "neutral", "rating": 0, "flagged": False},
                    ],
                    "human_completion": None,
                    "chosen_completion": 0,
                },
                {
                    "completions": [
                        {"text": "actual bad step", "rating": -1, "flagged": False},
                    ],
                    "human_completion": {
                        "text": "human correction",
                        "rating": None,
                        "source": "human",
                        "flagged": False,
                    },
                    "chosen_completion": None,
                },
            ],
        },
    }


class LoaderTests(unittest.TestCase):
    def test_human_completion_uses_text_field(self) -> None:
        step = record("p")["label"]["steps"][1]
        text, rating = selected_step_text_and_rating(
            step, include_human_completion_as_positive=True
        )
        self.assertEqual(text, "human correction")
        self.assertEqual(rating, 1)
        entries = candidate_entries(
            step,
            include_human_completion_as_positive=True,
            skip_flagged_completions=True,
        )
        self.assertEqual(entries[-1]["text"], "human correction")

    @unittest.skipIf(
        IndexedJsonlDataset is None,
        "PyTorch is not installed in the local audit environment",
    )
    def test_budget_subsets_are_nested_for_the_same_seed(self) -> None:
        assert IndexedJsonlDataset is not None
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            write_jsonl(
                path,
                [{"index": index} for index in range(30)],
            )
            small = IndexedJsonlDataset(
                path,
                max_examples=5,
                seed=7,
            )
            large = IndexedJsonlDataset(
                path,
                max_examples=15,
                seed=7,
            )
            self.assertTrue(set(small.offsets).issubset(large.offsets))


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            records = [record(f"problem {index}") for index in range(10)]
            write_jsonl(raw / "tiny.jsonl", records)
            first = build_problem_splits(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                seed=7,
                train_fraction=0.6,
                val_fraction=0.2,
            )
            second = build_problem_splits(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                seed=7,
                train_fraction=0.6,
                val_fraction=0.2,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["counts"], {"train": 6, "val": 2, "test": 2})
            validate_problem_splits(first)


class MaterializationTests(unittest.TestCase):
    def test_v0_filters_zero_and_builds_exact_prefix_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            write_jsonl(raw / "tiny.jsonl", [record("one problem")])
            split_payload = {
                "version": 1,
                "total_problems": 1,
                "problem_ids": {
                    "train": [problem_id_from_text("one problem")],
                    "val": [],
                    "test": [],
                },
            }
            split_path = root / "splits.json"
            write_problem_splits(split_payload, split_path)

            point_dir = root / "pointwise"
            trajectory_dir = root / "trajectories"
            pair_dir = root / "pairs"
            node_dir = root / "nodes"
            materialize_pointwise_v0(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                problem_split_path=split_path,
                output_dir=point_dir,
                trajectory_output_dir=trajectory_dir,
            )
            materialize_pairs_v0(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                problem_split_path=split_path,
                output_dir=pair_dir,
            )
            materialize_nodes_v0(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                problem_split_path=split_path,
                output_dir=node_dir,
            )

            points = read_jsonl(point_dir / "train.jsonl")
            self.assertEqual([item["rating"] for item in points], [1, -1, -1, 1])
            self.assertNotIn(0, [item["rating"] for item in points])

            pairs = read_jsonl(pair_dir / "train.jsonl")
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(item["pair_type"] == "+1>-1" for item in pairs))
            self.assertEqual(pairs[1]["prefix"], ["correct first step"])
            self.assertEqual(pairs[1]["positive"], "human correction")

            trajectories = read_jsonl(trajectory_dir / "train.jsonl")
            self.assertEqual(trajectories[0]["first_error_index"], 2)
            self.assertEqual(trajectories[0]["steps"][1]["text"], "actual bad step")

            nodes = read_jsonl(node_dir / "train.jsonl")
            self.assertEqual(len(nodes), 2)
            self.assertEqual(nodes[0]["num_candidates"], 2)
            self.assertEqual(nodes[0]["num_pairs"], 1)
            self.assertEqual(
                {candidate["label"] for candidate in nodes[1]["candidates"]},
                {0, 1},
            )

    def test_strict_nodes_drop_identical_text_with_conflicting_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            conflict_record = record("conflict problem")
            conflict_record["label"]["steps"][0]["completions"].extend(
                [
                    {"text": "ambiguous", "rating": 1, "flagged": False},
                    {"text": "ambiguous", "rating": -1, "flagged": False},
                ]
            )
            write_jsonl(raw / "tiny.jsonl", [conflict_record])
            split_payload = {
                "version": 1,
                "total_problems": 1,
                "problem_ids": {
                    "train": [problem_id_from_text("conflict problem")],
                    "val": [],
                    "test": [],
                },
            }
            split_path = root / "splits.json"
            write_problem_splits(split_payload, split_path)
            node_dir = root / "nodes"
            summary = materialize_nodes_v0(
                raw_data_dir=raw,
                raw_splits=["tiny"],
                problem_split_path=split_path,
                output_dir=node_dir,
            )
            nodes = read_jsonl(node_dir / "train.jsonl")
            self.assertNotIn(
                "ambiguous",
                [
                    candidate["text"]
                    for node in nodes
                    for candidate in node["candidates"]
                ],
            )
            self.assertEqual(
                summary["counts"]["conflicting_binary_texts_train"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
