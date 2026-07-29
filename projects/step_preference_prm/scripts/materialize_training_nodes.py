#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.data.materialize import (  # noqa: E402
    materialize_nodes_v0,
    write_materialization_summary,
)
from prm_pref.utils.config import load_config  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the strict exact-prefix node cohort used by all V0 objectives."
    )
    parser.add_argument("--config", default="configs/data_v0.yaml")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    materialize_cfg = config.get("materialize", {})
    output_dir = resolve_path(
        config.get("nodes_output_dir", "data/processed/nodes_v0")
    )
    summary = materialize_nodes_v0(
        raw_data_dir=resolve_path(config.get("raw_data_dir", "data/raw/prm800k")),
        raw_splits=config.get("raw_splits", []),
        problem_split_path=resolve_path(
            config.get(
                "problem_split_path",
                "data/processed/splits/problem_splits.json",
            )
        ),
        output_dir=output_dir,
        include_human_completion_as_positive=bool(
            materialize_cfg.get("include_human_completion_as_positive", True)
        ),
        skip_flagged_completions=bool(
            materialize_cfg.get("skip_flagged_completions", True)
        ),
        dedupe_candidates_by_text_and_rating=bool(
            materialize_cfg.get("dedupe_candidates_by_text_and_rating", True)
        ),
        drop_conflicting_binary_texts=bool(
            materialize_cfg.get("drop_conflicting_binary_texts", True)
        ),
        max_records=args.max_records,
    )
    write_materialization_summary(summary, output_dir / "stats.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
