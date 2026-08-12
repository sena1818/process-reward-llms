#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.eval.evaluator import build_evaluation_summary, evaluate_run  # noqa: E402
from prm_pref.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all V0 checkpoints.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run directory or run name. Repeat to evaluate multiple runs.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Evaluate requested runs without rewriting the global summary.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Build the summary from existing per-run metrics.json files.",
    )
    args = parser.parse_args()
    if args.metrics_only and args.summarize_only:
        parser.error("--metrics-only and --summarize-only are mutually exclusive")
    if args.summarize_only and args.run:
        parser.error("--summarize-only does not accept --run")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    runs_dir = Path(config.get("runs_dir", "outputs/runs"))
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir

    if args.summarize_only:
        metrics_filename = str(
            config.get("evaluation", {}).get("metrics_filename", "metrics.json")
        )
        if Path(metrics_filename).name != metrics_filename:
            raise ValueError("evaluation.metrics_filename must be a plain filename")
        metric_paths = sorted(runs_dir.glob(f"*/{metrics_filename}"))
        if not metric_paths:
            raise FileNotFoundError(
                f"No {metrics_filename} files found under {runs_dir}"
            )
        results = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in metric_paths
        ]
        summary = build_evaluation_summary(
            results,
            selection_seed=int(
                config.get("evaluation", {}).get("selection_seed", 42)
            ),
        )
        summary_filename = str(
            config.get("evaluation", {}).get("summary_filename", "v0_summary.json")
        )
        if Path(summary_filename).name != summary_filename:
            raise ValueError("evaluation.summary_filename must be a plain filename")
        output_path = runs_dir / summary_filename
        output_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        print(json.dumps(summary, indent=2))
        return 0

    checkpoint_choice = str(
        config.get("evaluation", {}).get("checkpoint", "last")
    )
    if args.run:
        run_dirs = []
        for value in args.run:
            path = Path(value)
            run_dirs.append(path if path.is_absolute() else runs_dir / path)
    else:
        run_dirs = sorted(
            path.parent
            for path in runs_dir.glob(f"*/{checkpoint_choice}.pt")
        )
    if not run_dirs:
        raise FileNotFoundError(
            f"No {checkpoint_choice}.pt checkpoints found under {runs_dir}"
        )

    results = []
    for run_dir in run_dirs:
        print(f"Evaluating {run_dir.name}")
        results.append(evaluate_run(project_root=PROJECT_ROOT, run_dir=run_dir, config=config))

    if not args.metrics_only:
        summary = build_evaluation_summary(
            results,
            selection_seed=int(
                config.get("evaluation", {}).get("selection_seed", 42)
            ),
        )
        output_path = runs_dir / "v0_summary.json"
        output_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
