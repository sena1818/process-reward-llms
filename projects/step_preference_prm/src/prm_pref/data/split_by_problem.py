from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from prm_pref.data.load_prm800k import get_problem_text, iter_records, problem_id_from_text


SPLIT_NAMES = ("train", "val", "test")


def build_problem_splits(
    *,
    raw_data_dir: Path,
    raw_splits: Iterable[str],
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    problem_by_id: dict[str, str] = {}
    for _, _, record in iter_records(raw_data_dir, raw_splits):
        problem = get_problem_text(record)
        if not problem:
            continue
        problem_id = problem_id_from_text(problem)
        existing = problem_by_id.setdefault(problem_id, problem)
        if existing != problem:
            raise RuntimeError(f"Problem hash collision for {problem_id}")

    problem_ids = sorted(problem_by_id)
    random.Random(seed).shuffle(problem_ids)

    total = len(problem_ids)
    train_end = int(total * train_fraction)
    val_end = train_end + int(total * val_fraction)
    assignments = {
        "train": sorted(problem_ids[:train_end]),
        "val": sorted(problem_ids[train_end:val_end]),
        "test": sorted(problem_ids[val_end:]),
    }

    return {
        "version": 1,
        "strategy": "exact_problem_text_sha256_then_seeded_shuffle",
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "counts": {name: len(ids) for name, ids in assignments.items()},
        "total_problems": total,
        "problem_ids": assignments,
    }


def write_problem_splits(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_problem_split_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problem_ids = payload.get("problem_ids") or {}
    result: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for problem_id in problem_ids.get(split, []):
            if problem_id in result:
                raise ValueError(f"Problem appears in multiple splits: {problem_id}")
            result[str(problem_id)] = split
    if len(result) != int(payload.get("total_problems", len(result))):
        raise ValueError("Split file total does not match its problem assignments")
    return result


def validate_problem_splits(payload: dict) -> None:
    groups = [set(payload["problem_ids"][name]) for name in SPLIT_NAMES]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Problem-level split leakage detected")
    if sum(map(len, groups)) != payload["total_problems"]:
        raise ValueError("Problem-level split count mismatch")
