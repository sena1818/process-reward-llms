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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train pointwise_v0 scalar PRM.")
    parser.add_argument("--config", default="configs/train_pointwise.yaml")
    parser.add_argument("--run-name", default=None)
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
    base_name = str(config.get("run", {}).get("name", "pointwise_v0"))
    suffix = ""
    if args.seed is not None:
        suffix += f"_seed{args.seed}"
    if args.max_train_nodes is not None:
        suffix += f"_n{args.max_train_nodes}"
    run_dir = run_training(
        project_root=PROJECT_ROOT,
        config=config,
        run_name=args.run_name
        or (
            f"{base_name}_smoke"
            if args.smoke
            else f"{base_name}_pilot"
            if args.pilot
            else f"{base_name}{suffix}"
            if suffix
            else None
        ),
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"Completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
