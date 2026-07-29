from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2.0 * precision * recall, precision + recall)


def binary_auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def binary_classification_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    threshold: float = 0.0,
) -> dict:
    score_list = list(scores)
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have equal length")
    predictions = [1 if score >= threshold else 0 for score in score_list]
    probabilities = [_sigmoid(score) for score in score_list]
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, label_list))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, label_list))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, label_list))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, label_list))
    f1_positive = _f1(tp, fp, fn)
    f1_negative = _f1(tn, fn, fp)
    return {
        "n": len(label_list),
        "threshold": threshold,
        "accuracy": _safe_div(tp + tn, len(label_list)),
        "macro_f1": (f1_positive + f1_negative) / 2.0,
        "f1_positive": f1_positive,
        "f1_negative": f1_negative,
        "auroc": binary_auroc(score_list, label_list),
        "brier": _safe_div(
            sum(
                (probability - label) ** 2
                for probability, label in zip(probabilities, label_list)
            ),
            len(label_list),
        ),
        "ece_10": expected_calibration_error(
            probabilities,
            label_list,
            num_bins=10,
        ),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def expected_calibration_error(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    num_bins: int = 10,
) -> float:
    probability_list = list(probabilities)
    label_list = [int(label) for label in labels]
    if len(probability_list) != len(label_list):
        raise ValueError("probabilities and labels must have equal length")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    total = len(label_list)
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        members = [
            index
            for index, probability in enumerate(probability_list)
            if lower <= probability < upper
            or (bin_index == num_bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probability_list[index] for index in members) / len(
            members
        )
        accuracy = sum(label_list[index] for index in members) / len(members)
        ece += len(members) / total * abs(accuracy - confidence)
    return ece


def pairwise_metrics(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> dict:
    positives = list(positive_scores)
    negatives = list(negative_scores)
    if len(positives) != len(negatives):
        raise ValueError("positive and negative score arrays must have equal length")
    wins = sum(pos > neg for pos, neg in zip(positives, negatives))
    ties = sum(pos == neg for pos, neg in zip(positives, negatives))
    return {
        "n": len(positives),
        "accuracy": _safe_div(wins, len(positives)),
        "ties": ties,
        "mean_margin": _safe_div(
            sum(pos - neg for pos, neg in zip(positives, negatives)), len(positives)
        ),
    }


def node_macro_pairwise_metrics(scored_nodes: Iterable[dict]) -> dict:
    """Compute pair accuracy inside each node, then macro-average nodes."""

    node_accuracies: list[float] = []
    node_margins: list[float] = []
    flat_wins = 0
    flat_ties = 0
    flat_pairs = 0
    for node in scored_nodes:
        scores = list(node["scores"])
        labels = [int(label) for label in node["labels"]]
        positives = [
            score for score, label in zip(scores, labels) if label == 1
        ]
        negatives = [
            score for score, label in zip(scores, labels) if label == 0
        ]
        margins = [
            positive - negative
            for positive in positives
            for negative in negatives
        ]
        if not margins:
            raise ValueError("Strict evaluation node lacks a positive/negative pair")
        wins = sum(margin > 0 for margin in margins)
        ties = sum(margin == 0 for margin in margins)
        node_accuracies.append(wins / len(margins))
        node_margins.append(sum(margins) / len(margins))
        flat_wins += wins
        flat_ties += ties
        flat_pairs += len(margins)
    return {
        "n_nodes": len(node_accuracies),
        "n_pairs": flat_pairs,
        "accuracy": _safe_div(sum(node_accuracies), len(node_accuracies)),
        "mean_margin": _safe_div(sum(node_margins), len(node_margins)),
        "flat_accuracy": _safe_div(flat_wins, flat_pairs),
        "flat_ties": flat_ties,
    }


def problem_cluster_bootstrap(
    records: Iterable[dict],
    metric_fn: Callable[[list[dict]], dict[str, float]],
    *,
    num_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap scalar metrics by resampling whole problem clusters."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    clusters: dict[str, list[dict]] = defaultdict(list)
    materialized = list(records)
    for record in materialized:
        problem_id = str(record.get("problem_id", ""))
        if not problem_id:
            raise ValueError("Every bootstrap record needs a problem_id")
        clusters[problem_id].append(record)
    cluster_values = list(clusters.values())
    if not cluster_values:
        raise ValueError("Cannot bootstrap an empty record collection")

    observed = metric_fn(materialized)
    distributions = {key: [] for key in observed}
    generator = random.Random(seed)
    for _ in range(num_samples):
        sample: list[dict] = []
        for _ in range(len(cluster_values)):
            sample.extend(
                cluster_values[generator.randrange(len(cluster_values))]
            )
        metrics = metric_fn(sample)
        if metrics.keys() != observed.keys():
            raise ValueError("metric_fn returned inconsistent metric keys")
        for key, value in metrics.items():
            distributions[key].append(float(value))

    return {
        "resampling_unit": "problem_id",
        "n_clusters": len(cluster_values),
        "n_bootstrap_samples": num_samples,
        "confidence_level": 0.95,
        "metrics": {
            key: {
                "estimate": float(observed[key]),
                "lower": _percentile(distribution, 0.025),
                "upper": _percentile(distribution, 0.975),
            }
            for key, distribution in distributions.items()
        },
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def select_binary_threshold(
    validation_scores: Iterable[float],
    validation_labels: Iterable[int],
    *,
    max_candidates: int = 201,
) -> dict:
    scores = list(validation_scores)
    labels = [int(label) for label in validation_labels]
    candidates = _candidate_thresholds(scores, max_candidates)
    best_threshold = None
    best_metrics = None
    best_key = None
    for threshold in candidates:
        metrics = binary_classification_metrics(scores, labels, threshold=threshold)
        key = (metrics["macro_f1"], metrics["accuracy"], -abs(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Could not select a validation classification threshold")
    return {
        "threshold": best_threshold,
        "selection_split": "validation",
        "selection_metric": "step_macro_f1",
        "num_candidate_thresholds": len(candidates),
        "validation_step": best_metrics,
    }


def first_error_metrics(scored_trajectories: Iterable[dict], threshold: float) -> dict:
    known = [item for item in scored_trajectories if item.get("first_error_index") is not None]
    errors: list[int] = []
    exact = 0
    within_one = 0
    detected = 0
    for trajectory in known:
        scores = trajectory["scores"]
        true_index = int(trajectory["first_error_index"])
        prediction = next(
            (index for index, score in enumerate(scores, start=1) if score < threshold),
            len(scores) + 1,
        )
        if prediction <= len(scores):
            detected += 1
        error = abs(prediction - true_index)
        errors.append(error)
        exact += error == 0
        within_one += error <= 1
    return {
        "n_known_first_error": len(known),
        "threshold": threshold,
        "exact_match": _safe_div(exact, len(known)),
        "within_1": _safe_div(within_one, len(known)),
        "mean_absolute_error": _safe_div(sum(errors), len(errors)),
        "detection_rate": _safe_div(detected, len(known)),
    }


def _candidate_thresholds(scores: list[float], max_candidates: int) -> list[float]:
    unique = sorted(set(scores))
    if not unique:
        raise ValueError("No validation scores available for threshold calibration")
    if len(unique) > max_candidates - 2:
        last = len(unique) - 1
        indices = {
            round(index * last / (max_candidates - 3)) for index in range(max_candidates - 2)
        }
        unique = [unique[index] for index in sorted(indices)]
    span = max(abs(unique[0]), abs(unique[-1]), 1.0)
    epsilon = span * 1e-6
    thresholds = [unique[0] - epsilon]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + epsilon)
    return thresholds


def select_first_error_threshold(
    validation_trajectories: list[dict],
    *,
    max_candidates: int = 201,
) -> dict:
    # V0 calibration explicitly ignores neutral (0) step labels.
    calibration_scores = [
        score
        for trajectory in validation_trajectories
        for score, rating in zip(trajectory["scores"], trajectory["ratings"])
        if rating in {-1, 1}
    ]
    candidates = _candidate_thresholds(calibration_scores, max_candidates)
    best_threshold = None
    best_metrics = None
    best_key = None
    for threshold in candidates:
        metrics = first_error_metrics(validation_trajectories, threshold)
        key = (
            metrics["within_1"],
            metrics["exact_match"],
            -metrics["mean_absolute_error"],
            -abs(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Could not select a validation threshold")
    if best_metrics["n_known_first_error"] == 0:
        raise ValueError("Validation trajectories contain no known first error")
    return {
        "threshold": best_threshold,
        "selection_split": "validation",
        "selection_metric": "first_error_within_1",
        "neutral_labels_used_for_candidates": False,
        "num_candidate_thresholds": len(candidates),
        "validation_first_error": best_metrics,
    }
