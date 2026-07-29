from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from prm_pref.data.datasets import (
    CandidateCollator,
    ExactPrefixNodeCollator,
    IndexedJsonlDataset,
)
from prm_pref.data.input_packing import build_input_packer
from prm_pref.eval.inference import (
    score_node_loader,
    score_pointwise_loader,
    score_trajectories,
)
from prm_pref.models.encoder_reward_model import load_reward_checkpoint
from prm_pref.utils.metrics import (
    binary_classification_metrics,
    first_error_metrics,
    node_macro_pairwise_metrics,
    problem_cluster_bootstrap,
    select_binary_threshold,
    select_first_error_threshold,
)


def _path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _flatten_nodes(nodes: list[dict]) -> tuple[list[float], list[int]]:
    return (
        [score for node in nodes for score in node["scores"]],
        [label for node in nodes for label in node["labels"]],
    )


def evaluate_run(
    *,
    project_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> dict:
    eval_cfg = dict(config.get("evaluation", {}))
    data_cfg = dict(config.get("data", {}))
    device = _device(str(eval_cfg.get("device", "auto")))
    precision = str(eval_cfg.get("mixed_precision", "bf16"))
    model, checkpoint = load_reward_checkpoint(
        run_dir / "best.pt",
        device=device,
    )
    model_config = dict(checkpoint["model_config"])
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "tokenizer",
        use_fast=True,
    )
    packer = build_input_packer(tokenizer, model_config)
    batch_size = int(eval_cfg.get("batch_size", 16))
    num_workers = int(eval_cfg.get("num_workers", 0))
    seed = int(eval_cfg.get("seed", 42))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    staged_data_root = os.environ.get("PRM_DATA_ROOT")
    if staged_data_root:
        nodes_dir = Path(staged_data_root) / "nodes_v0"
        trajectories_dir = Path(staged_data_root) / "trajectories"
        pointwise_dir = Path(staged_data_root) / "pointwise"
    else:
        nodes_dir = _path(
            project_root,
            data_cfg.get("nodes_dir", "data/processed/nodes_v0"),
        )
        trajectories_dir = _path(
            project_root,
            data_cfg.get(
                "trajectories_dir",
                "data/processed/trajectories",
            ),
        )
        pointwise_dir = _path(
            project_root,
            data_cfg.get("pointwise_dir", "data/processed/pointwise"),
        )

    split_results: dict[str, dict] = {}
    scored_nodes: dict[str, list[dict]] = {}
    scored_trajectories: dict[str, list[dict]] = {}
    full_pointwise: dict[str, tuple[list[float], list[int]]] = {}
    for split_index, split in enumerate(("val", "test")):
        node_dataset = IndexedJsonlDataset(
            nodes_dir / f"{split}.jsonl",
            max_examples=_optional_int(eval_cfg.get("max_nodes")),
            seed=seed + split_index,
        )
        node_loader = DataLoader(
            node_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ExactPrefixNodeCollator(packer),
            **loader_kwargs,
        )
        nodes = score_node_loader(
            model,
            node_loader,
            device,
            precision=precision,
        )
        scored_nodes[split] = nodes

        trajectories = score_trajectories(
            model=model,
            packer=packer,
            path=trajectories_dir / f"{split}.jsonl",
            batch_size=batch_size,
            device=device,
            precision=precision,
            max_trajectories=_optional_int(
                eval_cfg.get("max_trajectories")
            ),
        )
        scored_trajectories[split] = trajectories
        split_results[split] = {
            "pairwise": node_macro_pairwise_metrics(nodes),
            "num_trajectories": len(trajectories),
        }

        if bool(eval_cfg.get("report_full_pointwise", True)):
            point_dataset = IndexedJsonlDataset(
                pointwise_dir / f"{split}.jsonl",
                max_examples=_optional_int(
                    eval_cfg.get("max_pointwise_examples")
                ),
                seed=seed + split_index,
            )
            point_loader = DataLoader(
                point_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=CandidateCollator(packer),
                **loader_kwargs,
            )
            full_pointwise[split] = score_pointwise_loader(
                model,
                point_loader,
                device,
                precision=precision,
            )

    validation_scores, validation_labels = _flatten_nodes(
        scored_nodes["val"]
    )
    test_scores, test_labels = _flatten_nodes(scored_nodes["test"])
    step_calibration = select_binary_threshold(
        validation_scores,
        validation_labels,
        max_candidates=int(eval_cfg.get("threshold_candidates", 201)),
    )
    fixed_step_threshold = float(step_calibration["threshold"])
    split_results["val"]["step"] = step_calibration["validation_step"]
    split_results["test"]["step"] = binary_classification_metrics(
        test_scores,
        test_labels,
        threshold=fixed_step_threshold,
    )
    if full_pointwise:
        split_results["val"]["full_pointwise_step"] = (
            binary_classification_metrics(
                *full_pointwise["val"],
                threshold=fixed_step_threshold,
            )
        )
        split_results["test"]["full_pointwise_step"] = (
            binary_classification_metrics(
                *full_pointwise["test"],
                threshold=fixed_step_threshold,
            )
        )

    # This is deliberately the only first-error threshold-selection call.
    calibration = select_first_error_threshold(
        scored_trajectories["val"],
        max_candidates=int(eval_cfg.get("threshold_candidates", 201)),
    )
    fixed_threshold = float(calibration["threshold"])
    split_results["val"]["first_error"] = calibration[
        "validation_first_error"
    ]
    split_results["test"]["first_error"] = first_error_metrics(
        scored_trajectories["test"],
        fixed_threshold,
    )
    bootstrap_samples = int(eval_cfg.get("bootstrap_samples", 1000))

    def node_metrics(records: list[dict]) -> dict[str, float]:
        scores, labels = _flatten_nodes(records)
        step = binary_classification_metrics(
            scores,
            labels,
            threshold=fixed_step_threshold,
        )
        pairwise = node_macro_pairwise_metrics(records)
        return {
            "step_macro_f1": float(step["macro_f1"]),
            "pairwise_accuracy": float(pairwise["accuracy"]),
        }

    known_test_trajectories = [
        trajectory
        for trajectory in scored_trajectories["test"]
        if trajectory.get("first_error_index") is not None
    ]

    def trajectory_metrics(records: list[dict]) -> dict[str, float]:
        metrics = first_error_metrics(records, fixed_threshold)
        return {
            "first_error_exact": float(metrics["exact_match"]),
            "first_error_within_1": float(metrics["within_1"]),
            "first_error_mae": float(metrics["mean_absolute_error"]),
        }

    confidence_intervals = {
        "strict_nodes": problem_cluster_bootstrap(
            scored_nodes["test"],
            node_metrics,
            num_samples=bootstrap_samples,
            seed=seed,
        ),
        "known_first_error_trajectories": problem_cluster_bootstrap(
            known_test_trajectories,
            trajectory_metrics,
            num_samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }

    training_cfg = checkpoint.get("training_config", {})
    result = {
        "run_name": training_cfg.get("run_name", run_dir.name),
        "mode": training_cfg.get("mode"),
        "backend": model_config.get("backend", "encoder"),
        "model_name": model_config.get("name_or_path"),
        "max_length": model_config.get("max_length"),
        "lambda_pair": training_cfg.get("lambda_pair"),
        "seed": training_cfg.get("seed"),
        "train_nodes": training_cfg.get("train_nodes"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "step_threshold_calibration": step_calibration,
        "threshold_calibration": calibration,
        "validation": split_results["val"],
        "test": split_results["test"],
        "test_confidence_intervals": confidence_intervals,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_evaluation_summary(
    results: list[dict],
    *,
    selection_seed: int = 42,
) -> dict:
    rows = []
    for result in results:
        rows.append(
            {
                "run_name": result["run_name"],
                "mode": result["mode"],
                "backend": result.get("backend"),
                "model_name": result.get("model_name"),
                "max_length": result.get("max_length"),
                "lambda_pair": result["lambda_pair"],
                "seed": result.get("seed"),
                "train_nodes": result.get("train_nodes"),
                "val_step_macro_f1": result["validation"]["step"][
                    "macro_f1"
                ],
                "val_pairwise_accuracy": result["validation"]["pairwise"][
                    "accuracy"
                ],
                "val_first_error_within_1": result["validation"][
                    "first_error"
                ]["within_1"],
                "test_step_macro_f1": result["test"]["step"]["macro_f1"],
                "test_step_auroc": result["test"]["step"]["auroc"],
                "test_step_brier": result["test"]["step"]["brier"],
                "test_step_ece_10": result["test"]["step"]["ece_10"],
                "test_pairwise_accuracy": result["test"]["pairwise"][
                    "accuracy"
                ],
                "test_first_error_exact": result["test"]["first_error"][
                    "exact_match"
                ],
                "test_first_error_within_1": result["test"]["first_error"][
                    "within_1"
                ],
            }
        )
    aggregate_keys = [
        "val_step_macro_f1",
        "val_pairwise_accuracy",
        "val_first_error_within_1",
        "test_step_macro_f1",
        "test_step_auroc",
        "test_step_brier",
        "test_step_ece_10",
        "test_pairwise_accuracy",
        "test_first_error_exact",
        "test_first_error_within_1",
    ]
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["backend"],
            row["model_name"],
            row["max_length"],
            row["mode"],
            row["lambda_pair"],
            row["train_nodes"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (
        backend,
        model_name,
        max_length,
        mode,
        lambda_pair,
        train_nodes,
    ), group in grouped.items():
        metrics = {}
        for key in aggregate_keys:
            values = [
                float(row[key])
                for row in group
                if row[key] is not None
            ]
            metrics[key] = {
                "mean": statistics.fmean(values) if values else None,
                "sample_std": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
            }
        aggregates.append(
            {
                "backend": backend,
                "model_name": model_name,
                "max_length": max_length,
                "mode": mode,
                "lambda_pair": lambda_pair,
                "train_nodes": train_nodes,
                "n_runs": len(group),
                "seeds": sorted(
                    row["seed"]
                    for row in group
                    if row["seed"] is not None
                ),
                "run_names": [row["run_name"] for row in group],
                "metrics": metrics,
            }
        )
    aggregates.sort(
        key=lambda item: (
            str(item["backend"]),
            str(item["model_name"]),
            int(item["max_length"] or 0),
            str(item["mode"]),
            float(
                item["lambda_pair"]
                if item["lambda_pair"] is not None
                else -1.0
            ),
            int(item["train_nodes"] or 0),
        )
    )
    available_budgets = [
        item["train_nodes"]
        for item in aggregates
        if item["train_nodes"] is not None
    ]
    full_budget = max(available_budgets) if available_budgets else None
    selection_rows = [
        row
        for row in rows
        if row["mode"] == "hybrid"
        and (full_budget is None or row["train_nodes"] == full_budget)
        and row["seed"] == selection_seed
    ]
    if not selection_rows:
        selection_rows = [
            row
            for row in rows
            if row["mode"] == "hybrid"
            and (
                full_budget is None
                or row["train_nodes"] == full_budget
            )
        ]
    selected_row = (
        max(
            selection_rows,
            key=lambda row: (
                row["val_first_error_within_1"],
                row["val_pairwise_accuracy"],
            ),
        )
        if selection_rows
        else None
    )
    selected_aggregate = next(
        (
            item
            for item in aggregates
            if selected_row is not None
            and item["mode"] == "hybrid"
            and item["backend"] == selected_row["backend"]
            and item["model_name"] == selected_row["model_name"]
            and item["max_length"] == selected_row["max_length"]
            and item["lambda_pair"] == selected_row["lambda_pair"]
            and (
                full_budget is None
                or item["train_nodes"] == full_budget
            )
        ),
        None,
    )
    return {
        "selection_rule": (
            f"hybrid lambda selected once at full budget on seed "
            f"{selection_seed} by validation first-error within +/-1; "
            "validation node-macro pairwise accuracy breaks ties; "
            "replicate seeds do not retune lambda"
        ),
        "selected_hybrid_lambda": (
            selected_row["lambda_pair"] if selected_row else None
        ),
        "selected_hybrid_runs": (
            selected_aggregate["run_names"]
            if selected_aggregate is not None
            else []
        ),
        "replicate_aggregates": aggregates,
        "rows": rows,
    }
