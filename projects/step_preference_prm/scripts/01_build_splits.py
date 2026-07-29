#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.split_by_problem import (  # noqa: E402
    build_problem_splits,
    validate_problem_splits,
    write_problem_splits,
)
from prm_pref.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-free PRM800K problem splits.")
    parser.add_argument("--config", default="configs/data_v0.yaml")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    config = load_config(project_path(parse_args().config))
    split_cfg = config.get("problem_split", {})
    payload = build_problem_splits(
        raw_data_dir=project_path(config.get("raw_data_dir", "data/raw/prm800k")),
        raw_splits=config.get("raw_splits", []),
        seed=int(split_cfg.get("seed", 42)),
        train_fraction=float(split_cfg.get("train_fraction", 0.8)),
        val_fraction=float(split_cfg.get("val_fraction", 0.1)),
    )
    validate_problem_splits(payload)
    output_path = project_path(
        config.get("problem_split_path", "data/processed/splits/problem_splits.json")
    )
    write_problem_splits(payload, output_path)
    print(f"Wrote {output_path}")
    print(f"Problem counts: {payload['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
