#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prm_pref.training.runner import run_training  # noqa: E402
from prm_pref.training.configuration import (  # noqa: E402
    with_experiment_overrides,
    with_pilot_profile,
    with_smoke_profile,
)
from prm_pref.utils.config import load_config  # noqa: E402


def lambda_name(value: float) -> str:
    return f"{value:.1f}" if value in {0.1, 0.3, 0.5, 1.0} else f"{value:g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the hybrid_v0 lambda sweep.")
    parser.add_argument("--config", default="configs/train_hybrid.yaml")
    parser.add_argument(
        "--lambda-pair",
        type=float,
        action="append",
        default=None,
        help="Train only selected lambda(s); repeat this option as needed.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-nodes", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.pilot:
        parser.error("--smoke and --pilot are mutually exclusive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    config = with_experiment_overrides(
        config,
        seed=args.seed,
        max_train_nodes=args.max_train_nodes,
    )
    if args.smoke:
        config = with_smoke_profile(config)
    elif args.pilot:
        config = with_pilot_profile(config)
    lambdas = args.lambda_pair or (
        [0.3]
        if args.smoke or args.pilot
        else config.get("lambda_sweep", [0.1, 0.3, 0.5, 1.0])
    )
    base_name = str(config.get("run", {}).get("name", "hybrid_v0"))
    suffix = ""
    if args.seed is not None:
        suffix += f"_seed{args.seed}"
    if args.max_train_nodes is not None:
        suffix += f"_n{args.max_train_nodes}"
    for value in map(float, lambdas):
        run_dir = run_training(
            project_root=PROJECT_ROOT,
            config=config,
            run_name=(
                f"{base_name}_lam{lambda_name(value)}{suffix}_smoke"
                if args.smoke
                else f"{base_name}_lam{lambda_name(value)}{suffix}_pilot"
                if args.pilot
                else f"{base_name}_lam{lambda_name(value)}{suffix}"
            ),
            lambda_pair_override=value,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        print(f"Completed lambda={value}: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
