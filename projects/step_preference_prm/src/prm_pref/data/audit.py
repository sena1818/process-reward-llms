from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from prm_pref.data.load_prm800k import (
    candidate_entries,
    dedupe_candidates,
    get_finish_reason,
    get_problem_text,
    get_steps,
    iter_records,
    iter_split_paths,
    selected_step_text_and_rating,
)


PAIR_KEYS = (
    "+1>-1",
    "+1>0",
    "0>-1",
)


def _rating_key(rating: int) -> str:
    return "+1" if rating == 1 else str(rating)


def _pair_counts_for_ratings(rating_counts: Counter[int]) -> dict[str, int]:
    counts = {
        "+1>-1": rating_counts[1] * rating_counts[-1],
        "+1>0": rating_counts[1] * rating_counts[0],
        "0>-1": rating_counts[0] * rating_counts[-1],
    }
    counts["total"] = sum(counts.values())
    return counts


def audit_prm800k(
    *,
    raw_data_dir: Path,
    splits: Iterable[str],
    include_human_completion_as_positive: bool = True,
    skip_flagged_completions: bool = True,
    dedupe_candidates_by_text_and_rating: bool = True,
) -> tuple[dict, dict]:
    raw_data_dir = raw_data_dir.resolve()
    split_list = list(splits)

    source_files = [str(path) for path in iter_split_paths(raw_data_dir, split_list)]
    per_split: dict[str, dict] = {}
    problems_seen: set[str] = set()
    problems_with_first_error: set[str] = set()

    label_distribution: Counter[int] = Counter()
    finish_reason_distribution: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()

    totals = Counter(
        {
            "solution_samples": 0,
            "annotated_step_nodes": 0,
            "candidate_labels": 0,
            "human_completion_as_positive_labels": 0,
            "exact_prefix_nodes_total": 0,
            "exact_prefix_nodes_with_multiple_candidates": 0,
            "exact_prefix_nodes_with_multiple_rating_levels": 0,
            "exact_prefix_nodes_with_preference_pairs": 0,
            "solution_samples_with_first_error": 0,
            "solution_samples_with_chosen_negative": 0,
            "chosen_steps_with_rating": 0,
        }
    )

    for split, line_no, record in iter_records(raw_data_dir, split_list):
        split_stats = per_split.setdefault(
            split,
            {
                "solution_samples": 0,
                "unique_problem_texts": 0,
                "annotated_step_nodes": 0,
                "candidate_labels": 0,
                "label_distribution": {"-1": 0, "0": 0, "+1": 0},
                "first_error_samples": 0,
            },
        )

        del line_no
        totals["solution_samples"] += 1
        split_stats["solution_samples"] += 1

        problem = get_problem_text(record)
        if problem:
            problems_seen.add(problem)

        finish_reason = get_finish_reason(record)
        finish_reason_distribution[finish_reason] += 1
        steps = get_steps(record)
        totals["annotated_step_nodes"] += len(steps)
        split_stats["annotated_step_nodes"] += len(steps)

        chosen_negative_found = False
        final_step_has_negative_candidate = False

        for step_index, step in enumerate(steps):
            candidates = candidate_entries(
                step,
                include_human_completion_as_positive=include_human_completion_as_positive,
                skip_flagged_completions=skip_flagged_completions,
            )
            if dedupe_candidates_by_text_and_rating:
                candidates = dedupe_candidates(candidates)

            if candidates:
                totals["exact_prefix_nodes_total"] += 1

            if len(candidates) > 1:
                totals["exact_prefix_nodes_with_multiple_candidates"] += 1

            rating_counts: Counter[int] = Counter()
            for candidate in candidates:
                rating = candidate["rating"]
                rating_counts[rating] += 1
                label_distribution[rating] += 1
                candidate_source_counts[candidate["source"]] += 1
                totals["candidate_labels"] += 1
                split_stats["candidate_labels"] += 1
                if candidate["source"] == "human_completion":
                    totals["human_completion_as_positive_labels"] += 1
                split_stats["label_distribution"][_rating_key(rating)] += 1

            if len(rating_counts) > 1:
                totals["exact_prefix_nodes_with_multiple_rating_levels"] += 1

            node_pair_counts = _pair_counts_for_ratings(rating_counts)
            for key in PAIR_KEYS:
                pair_counts[key] += node_pair_counts[key]
            if node_pair_counts["total"] > 0:
                totals["exact_prefix_nodes_with_preference_pairs"] += 1

            selected_text, selected_rating = selected_step_text_and_rating(
                step,
                include_human_completion_as_positive=include_human_completion_as_positive,
            )
            if selected_text is not None and selected_rating is not None:
                totals["chosen_steps_with_rating"] += 1
                if selected_rating == -1:
                    chosen_negative_found = True

            if step_index == len(steps) - 1:
                final_step_has_negative_candidate = any(
                    candidate["rating"] == -1 for candidate in candidates
                )

        first_error_found = chosen_negative_found or (
            finish_reason == "found_error" and final_step_has_negative_candidate
        )

        if chosen_negative_found:
            totals["solution_samples_with_chosen_negative"] += 1

        if first_error_found:
            totals["solution_samples_with_first_error"] += 1
            split_stats["first_error_samples"] += 1
            if problem:
                problems_with_first_error.add(problem)

    # Fill split-level unique problem counts in a second pass to avoid storing all
    # problem texts per split in the main stats object.
    per_split_problem_sets: dict[str, set[str]] = {split: set() for split in split_list}
    for split, _, record in iter_records(raw_data_dir, split_list):
        problem = get_problem_text(record)
        if problem:
            per_split_problem_sets.setdefault(split, set()).add(problem)
    for split, problem_set in per_split_problem_sets.items():
        if split in per_split:
            per_split[split]["unique_problem_texts"] = len(problem_set)

    total_pairs = sum(pair_counts[key] for key in PAIR_KEYS)
    pair_counts["total"] = total_pairs

    data_audit = {
        "source_files": source_files,
        "solution_samples": totals["solution_samples"],
        "unique_problem_texts": len(problems_seen),
        "annotated_step_nodes": totals["annotated_step_nodes"],
        "candidate_labels": totals["candidate_labels"],
        "human_completion_as_positive_labels": totals[
            "human_completion_as_positive_labels"
        ],
        "chosen_steps_with_rating": totals["chosen_steps_with_rating"],
        "label_distribution": {
            "-1": label_distribution[-1],
            "0": label_distribution[0],
            "+1": label_distribution[1],
        },
        "candidate_source_distribution": dict(sorted(candidate_source_counts.items())),
        "finish_reason_distribution": dict(sorted(finish_reason_distribution.items())),
        "first_error": {
            "solution_samples_with_first_error": totals[
                "solution_samples_with_first_error"
            ],
            "solution_samples_with_chosen_negative": totals[
                "solution_samples_with_chosen_negative"
            ],
            "finish_reason_found_error_samples": finish_reason_distribution["found_error"],
            "problems_with_at_least_one_first_error": len(problems_with_first_error),
        },
        "per_split": per_split,
    }

    pair_stats = {
        "exact_prefix_nodes_total": totals["exact_prefix_nodes_total"],
        "exact_prefix_nodes_with_multiple_candidates": totals[
            "exact_prefix_nodes_with_multiple_candidates"
        ],
        "exact_prefix_nodes_with_multiple_rating_levels": totals[
            "exact_prefix_nodes_with_multiple_rating_levels"
        ],
        "exact_prefix_nodes_with_preference_pairs": totals[
            "exact_prefix_nodes_with_preference_pairs"
        ],
        "pair_counts": {
            "+1>-1": pair_counts["+1>-1"],
            "+1>0": pair_counts["+1>0"],
            "0>-1": pair_counts["0>-1"],
            "total": pair_counts["total"],
        },
        "default_pair_weights": {
            "+1>-1": 1.0,
            "+1>0": 0.5,
            "0>-1": 0.5,
        },
        "candidate_inclusion": {
            "include_human_completion_as_positive": include_human_completion_as_positive,
            "skip_flagged_completions": skip_flagged_completions,
            "dedupe_candidates_by_text_and_rating": dedupe_candidates_by_text_and_rating,
        },
    }
    return data_audit, pair_stats


def write_audit_outputs(
    *,
    data_audit: dict,
    pair_stats: dict,
    output_dir: Path,
    processed_audit_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "data_audit.json", data_audit)
    _write_json(output_dir / "pair_stats.json", pair_stats)

    if processed_audit_dir is not None:
        processed_audit_dir.mkdir(parents=True, exist_ok=True)
        _write_json(processed_audit_dir / "data_audit.json", data_audit)
        _write_json(processed_audit_dir / "pair_stats.json", pair_stats)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
