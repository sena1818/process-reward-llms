#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.materialize import (  # noqa: E402
    materialize_pointwise_v0,
    write_materialization_summary,
)
from prm_pref.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize V0 pointwise data and trajectories.")
    parser.add_argument("--config", default="configs/data_v0.yaml")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trajectory-output-dir", default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config = load_config(project_path(args.config))
    materialize_cfg = config.get("materialize", {})
    output_dir = project_path(
        args.output_dir or config.get("pointwise_output_dir", "data/processed/pointwise")
    )
    trajectory_output_dir = project_path(
        args.trajectory_output_dir
        or config.get("trajectory_output_dir", "data/processed/trajectories")
    )
    summary = materialize_pointwise_v0(
        raw_data_dir=project_path(config.get("raw_data_dir", "data/raw/prm800k")),
        raw_splits=config.get("raw_splits", []),
        problem_split_path=project_path(
            config.get("problem_split_path", "data/processed/splits/problem_splits.json")
        ),
        output_dir=output_dir,
        trajectory_output_dir=trajectory_output_dir,
        include_human_completion_as_positive=bool(
            materialize_cfg.get("include_human_completion_as_positive", True)
        ),
        skip_flagged_completions=bool(
            materialize_cfg.get("skip_flagged_completions", True)
        ),
        dedupe_candidates_by_text_and_rating=bool(
            materialize_cfg.get("dedupe_candidates_by_text_and_rating", True)
        ),
        max_records=args.max_records,
    )
    write_materialization_summary(summary, output_dir / "stats.json")
    print(f"Wrote V0 pointwise data to {output_dir}")
    print(f"Wrote trajectories to {trajectory_output_dir}")
    print(f"Counts: {summary['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
