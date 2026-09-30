#!/usr/bin/env python3
"""Reproduce validation-fitted Platt calibration from saved V1 scores.

Uses only Python's standard library and the project's standard-library metrics.
No model loading, training, or test-label fitting is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from prm_pref.utils.metrics import expected_calibration_error  # noqa: E402


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def read_scores(lines) -> tuple[list[float], list[int]]:
    scores, labels = [], []
    for line in lines:
        row = json.loads(line)
        if len(row["scores"]) != len(row["labels"]):
            raise ValueError("Each node must have one label per score")
        for score, label in zip(row["scores"], row["labels"]):
            score = float(score)
            if not math.isfinite(score) or label not in (0, 1):
                raise ValueError("Scores must be finite and labels binary")
            scores.append(score)
            labels.append(int(label))
    if not scores:
        raise ValueError("Score file is empty")
    return scores, labels


def cross_entropy(scores, labels, slope, bias) -> float:
    total = 0.0
    for score, label in zip(scores, labels):
        logit = slope * score + bias
        total += max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - label * logit
    return total / len(labels)


def fit_platt(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Minimise unregularised candidate-level validation binary cross-entropy."""
    if not scores or len(scores) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("Fit requires equal-length scores and both binary labels")
    slope, bias = 1.0, 0.0
    for _ in range(50):
        g_slope = g_bias = h_ss = h_sb = h_bb = 0.0
        for score, label in zip(scores, labels):
            probability = sigmoid(slope * score + bias)
            residual = probability - label
            variance = probability * (1.0 - probability)
            g_slope += residual * score
            g_bias += residual
            h_ss += variance * score * score
            h_sb += variance * score
            h_bb += variance
        if max(abs(g_slope), abs(g_bias)) / len(labels) < 1e-10:
            return slope, bias
        determinant = h_ss * h_bb - h_sb * h_sb
        if determinant <= 0:
            raise ValueError("Calibration fit is singular; scores may be constant or separable")
        step_slope = (h_bb * g_slope - h_sb * g_bias) / determinant
        step_bias = (h_ss * g_bias - h_sb * g_slope) / determinant
        previous_loss = cross_entropy(scores, labels, slope, bias)
        scale = 1.0
        for _ in range(32):
            new_slope = slope - scale * step_slope
            new_bias = bias - scale * step_bias
            if cross_entropy(scores, labels, new_slope, new_bias) <= previous_loss + 1e-14:
                break
            scale *= 0.5
        else:
            raise RuntimeError("Calibration line search did not converge")
        slope, bias = new_slope, new_bias
    raise RuntimeError("Calibration fit did not converge in 50 iterations")


def calibration_metrics(scores, labels, slope=1.0, bias=0.0) -> dict:
    probabilities = [sigmoid(slope * score + bias) for score in scores]
    return {
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels),
        "ece_10": expected_calibration_error(probabilities, labels, num_bins=10),
    }


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--runs-dir", help="Directory containing V1 run directories and v1_summary.json")
    sources.add_argument("--scores-zip", help="Score ZIP; defaults to the bundled submission archive")
    parser.add_argument("--summary", help="Override the v1_summary.json used to identify runs")
    parser.add_argument("--output", required=True, help="Output JSON path (relative to project or absolute)")
    args = parser.parse_args()
    runs_dir = project_path(args.runs_dir) if args.runs_dir else None
    summary_path = project_path(args.summary) if args.summary else (
        runs_dir / "v1_summary.json" if runs_dir else PROJECT_ROOT / "submission/artifacts/v1_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    run_specs = summary["rows"]
    if len({row["run_name"] for row in run_specs}) != len(run_specs):
        raise ValueError("Summary has duplicate runs")
    rows = []
    archive = zipfile.ZipFile(project_path(args.scores_zip or "submission/artifacts/v1_scores.zip")) if runs_dir is None else None
    try:
        for spec in run_specs:
            run_name = spec["run_name"]
            data = {}
            for split in ("val", "test"):
                name = f"{run_name}/strict_node_scores_{split}.jsonl"
                handle = archive.open(name) if archive else (runs_dir / name).open("rb")
                with handle:
                    data[split] = read_scores(handle)
            slope, bias = fit_platt(*data["val"])
            if slope <= 0:
                raise ValueError(f"{run_name}: non-positive slope would not preserve ranking")
            test_scores, test_labels = data["test"]
            rows.append({
                "run_name": run_name, "mode": spec["mode"], "lambda_pair": spec["lambda_pair"],
                "seed": spec["seed"], "slope": slope, "bias": bias,
                "validation_candidates": len(data["val"][0]), "test_candidates": len(test_scores),
                "test_raw": calibration_metrics(test_scores, test_labels),
                "test_platt": calibration_metrics(test_scores, test_labels, slope, bias),
            })
    finally:
        if archive:
            archive.close()
    groups = {}
    for row in rows:
        groups.setdefault((row["mode"], row["lambda_pair"]), []).append(row)
    aggregates = []
    for (mode, weight), group in groups.items():
        metrics = {}
        for field in ("brier", "ece_10"):
            for variant in ("raw", "platt"):
                values = [row[f"test_{variant}"][field] for row in group]
                metrics[f"test_{variant}_{field}"] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                }
        aggregates.append({"mode": mode, "lambda_pair": weight, "n_runs": len(group), "metrics": metrics})
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "method": "sigmoid(a*r+b); unregularised candidate-level BCE fitted on validation only",
        "metric_scale": "fractions; multiply by 100 for report tables",
        "all_slopes_positive": True, "rows": rows, "replicate_aggregates": aggregates,
    }, indent=2) + "\n")
    print(f"Wrote {output} ({len(rows)} runs; all slopes positive)")


if __name__ == "__main__":
    main()
