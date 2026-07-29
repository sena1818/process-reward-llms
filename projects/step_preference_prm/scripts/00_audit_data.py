#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from prm_pref.data.audit import audit_prm800k, write_audit_outputs  # noqa: E402
from prm_pref.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PRM800K labels and preference pairs.")
    parser.add_argument("--config", default="configs/audit.yaml")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))

    audit_cfg = config.get("audit", {})
    data_audit, pair_stats = audit_prm800k(
        raw_data_dir=resolve_project_path(config.get("raw_data_dir", "data/raw/prm800k")),
        splits=config.get("splits", []),
        include_human_completion_as_positive=bool(
            audit_cfg.get("include_human_completion_as_positive", True)
        ),
        skip_flagged_completions=bool(audit_cfg.get("skip_flagged_completions", True)),
        dedupe_candidates_by_text_and_rating=bool(
            audit_cfg.get("dedupe_candidates_by_text_and_rating", True)
        ),
    )

    output_dir = resolve_project_path(config.get("output_dir", "outputs/audit"))
    processed_audit_dir = resolve_project_path(
        config.get("processed_audit_dir", "data/processed/audit")
    )
    write_audit_outputs(
        data_audit=data_audit,
        pair_stats=pair_stats,
        output_dir=output_dir,
        processed_audit_dir=processed_audit_dir,
    )

    print(f"Wrote {output_dir / 'data_audit.json'}")
    print(f"Wrote {output_dir / 'pair_stats.json'}")
    print("")
    print("Key stats:")
    print(f"  solution_samples: {data_audit['solution_samples']}")
    print(f"  unique_problem_texts: {data_audit['unique_problem_texts']}")
    print(f"  annotated_step_nodes: {data_audit['annotated_step_nodes']}")
    print(f"  label_distribution: {data_audit['label_distribution']}")
    print(
        "  exact_prefix_nodes_with_preference_pairs: "
        f"{pair_stats['exact_prefix_nodes_with_preference_pairs']}"
    )
    print(f"  pair_counts: {pair_stats['pair_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
