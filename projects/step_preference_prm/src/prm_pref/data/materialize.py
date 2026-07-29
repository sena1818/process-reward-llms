from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Iterator

from prm_pref.data.load_prm800k import (
    candidate_entries,
    dedupe_candidates,
    drop_conflicting_binary_candidates,
    get_finish_reason,
    get_problem_text,
    get_steps,
    iter_records,
    problem_id_from_text,
    sample_id,
    selected_step_text_and_rating,
    trajectory_step_text_and_rating,
)
from prm_pref.data.split_by_problem import SPLIT_NAMES, load_problem_split_map


def _jsonl_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _iter_assigned_records(
    *,
    raw_data_dir: Path,
    raw_splits: Iterable[str],
    problem_split_path: Path,
    max_records: int | None,
) -> Iterator[tuple[str, str, int, dict, str, str]]:
    split_map = load_problem_split_map(problem_split_path)
    seen = 0
    for raw_split, line_no, record in iter_records(raw_data_dir, raw_splits):
        problem = get_problem_text(record)
        if not problem:
            continue
        problem_id = problem_id_from_text(problem)
        split = split_map.get(problem_id)
        if split is None:
            raise KeyError(f"Problem missing from split file: {problem_id}")
        yield split, raw_split, line_no, record, problem_id, problem
        seen += 1
        if max_records is not None and seen >= max_records:
            break


def _candidate_list(
    step: dict,
    *,
    include_human_completion_as_positive: bool,
    skip_flagged_completions: bool,
    dedupe_candidates_by_text_and_rating: bool,
) -> list[dict]:
    candidates = candidate_entries(
        step,
        include_human_completion_as_positive=include_human_completion_as_positive,
        skip_flagged_completions=skip_flagged_completions,
    )
    return dedupe_candidates(candidates) if dedupe_candidates_by_text_and_rating else candidates


def materialize_pointwise_v0(
    *,
    raw_data_dir: Path,
    raw_splits: Iterable[str],
    problem_split_path: Path,
    output_dir: Path,
    trajectory_output_dir: Path | None = None,
    include_human_completion_as_positive: bool = True,
    skip_flagged_completions: bool = True,
    dedupe_candidates_by_text_and_rating: bool = True,
    max_records: int | None = None,
) -> dict:
    """Write V0 pointwise examples and selected model trajectories.

    Pointwise records contain only ratings +1 and -1.  Trajectories retain
    rating 0 because a neutral step can precede the known first error; zero is
    never used as a V0 training target or threshold-calibration label.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if trajectory_output_dir is not None:
        trajectory_output_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        point_handles = {
            split: stack.enter_context((output_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
            for split in SPLIT_NAMES
        }
        trajectory_handles = (
            {
                split: stack.enter_context(
                    (trajectory_output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
                )
                for split in SPLIT_NAMES
            }
            if trajectory_output_dir is not None
            else None
        )

        for split, raw_split, line_no, record, problem_id, problem in _iter_assigned_records(
            raw_data_dir=raw_data_dir,
            raw_splits=raw_splits,
            problem_split_path=problem_split_path,
            max_records=max_records,
        ):
            record_sample_id = sample_id(raw_split, line_no)
            prefix: list[str] = []
            trajectory_steps: list[dict] = []
            chosen_negative_indices: list[int] = []
            final_step_has_negative = False

            record_steps = get_steps(record)
            for zero_based_index, step in enumerate(record_steps):
                step_index = zero_based_index + 1
                candidates = _candidate_list(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                    skip_flagged_completions=skip_flagged_completions,
                    dedupe_candidates_by_text_and_rating=dedupe_candidates_by_text_and_rating,
                )
                if zero_based_index == len(record_steps) - 1:
                    final_step_has_negative = any(item["rating"] == -1 for item in candidates)
                for candidate_index, candidate in enumerate(candidates):
                    rating = int(candidate["rating"])
                    if rating not in {-1, 1}:
                        continue
                    point_handles[split].write(
                        _jsonl_line(
                            {
                                "problem_id": problem_id,
                                "sample_id": record_sample_id,
                                "raw_split": raw_split,
                                "step_index": step_index,
                                "candidate_index": candidate_index,
                                "problem": problem,
                                "prefix": list(prefix),
                                "candidate": candidate["text"],
                                "label": 1 if rating == 1 else 0,
                                "rating": rating,
                            }
                        )
                    )
                    counts[f"pointwise_{split}"] += 1
                    counts[f"pointwise_{split}_label_{rating:+d}"] += 1

                trajectory_text, trajectory_rating = trajectory_step_text_and_rating(
                    record,
                    zero_based_index,
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                )
                if trajectory_text is not None and trajectory_rating is not None:
                    trajectory_steps.append(
                        {
                            "step_index": step_index,
                            "text": trajectory_text,
                            "rating": trajectory_rating,
                        }
                    )

                selected_text, selected_rating = selected_step_text_and_rating(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                )
                if selected_rating == -1:
                    chosen_negative_indices.append(step_index)
                if selected_text is not None:
                    prefix.append(selected_text)
                elif trajectory_text is not None:
                    # This is normally the terminal found-error step.  Appending
                    # it is harmless and makes the logic robust to longer trees.
                    prefix.append(trajectory_text)

            if trajectory_handles is not None and trajectory_steps:
                finish_reason = get_finish_reason(record)
                if chosen_negative_indices:
                    first_error_index = chosen_negative_indices[0]
                elif finish_reason == "found_error" and final_step_has_negative:
                    first_error_index = len(record_steps)
                else:
                    first_error_index = None
                trajectory_handles[split].write(
                    _jsonl_line(
                        {
                            "problem_id": problem_id,
                            "sample_id": record_sample_id,
                            "raw_split": raw_split,
                            "problem": problem,
                            "finish_reason": finish_reason,
                            "steps": trajectory_steps,
                            "first_error_index": first_error_index,
                        }
                    )
                )
                counts[f"trajectories_{split}"] += 1
                if first_error_index is not None:
                    counts[f"first_error_trajectories_{split}"] += 1

    return _materialization_summary("pointwise_v0", counts, max_records)


def materialize_pairs_v0(
    *,
    raw_data_dir: Path,
    raw_splits: Iterable[str],
    problem_split_path: Path,
    output_dir: Path,
    include_human_completion_as_positive: bool = True,
    skip_flagged_completions: bool = True,
    dedupe_candidates_by_text_and_rating: bool = True,
    max_records: int | None = None,
) -> dict:
    """Write only exact-prefix +1 > -1 Cartesian pairs for V0."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        handles = {
            split: stack.enter_context((output_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
            for split in SPLIT_NAMES
        }
        for split, raw_split, line_no, record, problem_id, problem in _iter_assigned_records(
            raw_data_dir=raw_data_dir,
            raw_splits=raw_splits,
            problem_split_path=problem_split_path,
            max_records=max_records,
        ):
            record_sample_id = sample_id(raw_split, line_no)
            prefix: list[str] = []
            for zero_based_index, step in enumerate(get_steps(record)):
                step_index = zero_based_index + 1
                candidates = _candidate_list(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                    skip_flagged_completions=skip_flagged_completions,
                    dedupe_candidates_by_text_and_rating=dedupe_candidates_by_text_and_rating,
                )
                positives = [item for item in candidates if item["rating"] == 1]
                negatives = [item for item in candidates if item["rating"] == -1]
                context_id = f"{record_sample_id}:step:{step_index}"
                for positive_index, positive in enumerate(positives):
                    for negative_index, negative in enumerate(negatives):
                        handles[split].write(
                            _jsonl_line(
                                {
                                    "problem_id": problem_id,
                                    "sample_id": record_sample_id,
                                    "context_id": context_id,
                                    "raw_split": raw_split,
                                    "step_index": step_index,
                                    "pair_index": f"{positive_index}:{negative_index}",
                                    "problem": problem,
                                    "prefix": list(prefix),
                                    "positive": positive["text"],
                                    "negative": negative["text"],
                                    "pair_type": "+1>-1",
                                    "weight": 1.0,
                                }
                            )
                        )
                        counts[f"pairs_{split}"] += 1

                selected_text, _ = selected_step_text_and_rating(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                )
                if selected_text is not None:
                    prefix.append(selected_text)
                else:
                    trajectory_text, _ = trajectory_step_text_and_rating(
                        record,
                        zero_based_index,
                        step,
                        include_human_completion_as_positive=include_human_completion_as_positive,
                    )
                    if trajectory_text is not None:
                        prefix.append(trajectory_text)

    return _materialization_summary("pairs_v0", counts, max_records)


def materialize_nodes_v0(
    *,
    raw_data_dir: Path,
    raw_splits: Iterable[str],
    problem_split_path: Path,
    output_dir: Path,
    include_human_completion_as_positive: bool = True,
    skip_flagged_completions: bool = True,
    dedupe_candidates_by_text_and_rating: bool = True,
    drop_conflicting_binary_texts: bool = True,
    max_records: int | None = None,
) -> dict:
    """Materialize the strict, node-balanced V0 training cohort.

    One JSONL record is one annotated ``problem + exact prefix`` node.  A node
    is retained only when its cleaned binary candidates contain at least one
    +1 and one -1.  Pointwise, pairwise, and hybrid training all consume this
    same artifact, so candidate forwards and annotation budgets are directly
    comparable across objectives.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        handles = {
            split: stack.enter_context(
                (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in SPLIT_NAMES
        }
        for split, raw_split, line_no, record, problem_id, problem in _iter_assigned_records(
            raw_data_dir=raw_data_dir,
            raw_splits=raw_splits,
            problem_split_path=problem_split_path,
            max_records=max_records,
        ):
            record_sample_id = sample_id(raw_split, line_no)
            prefix: list[str] = []
            for zero_based_index, step in enumerate(get_steps(record)):
                step_index = zero_based_index + 1
                candidates = _candidate_list(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                    skip_flagged_completions=skip_flagged_completions,
                    dedupe_candidates_by_text_and_rating=dedupe_candidates_by_text_and_rating,
                )
                binary_candidates = [
                    candidate for candidate in candidates if int(candidate["rating"]) in {-1, 1}
                ]
                conflict_count = 0
                if drop_conflicting_binary_texts:
                    binary_candidates, conflict_count = drop_conflicting_binary_candidates(
                        binary_candidates
                    )
                if conflict_count:
                    counts[f"conflicting_binary_texts_{split}"] += conflict_count
                    counts[f"nodes_with_conflicting_binary_texts_{split}"] += 1

                positives = [
                    candidate for candidate in binary_candidates if int(candidate["rating"]) == 1
                ]
                negatives = [
                    candidate for candidate in binary_candidates if int(candidate["rating"]) == -1
                ]
                if positives and negatives:
                    context_id = f"{record_sample_id}:step:{step_index}"
                    node_candidates = [
                        {
                            "text": candidate["text"],
                            "label": 1 if int(candidate["rating"]) == 1 else 0,
                            "rating": int(candidate["rating"]),
                            "source": candidate["source"],
                        }
                        for candidate in binary_candidates
                    ]
                    num_pairs = len(positives) * len(negatives)
                    handles[split].write(
                        _jsonl_line(
                            {
                                "problem_id": problem_id,
                                "sample_id": record_sample_id,
                                "context_id": context_id,
                                "raw_split": raw_split,
                                "step_index": step_index,
                                "problem": problem,
                                "prefix": list(prefix),
                                "candidates": node_candidates,
                                "num_candidates": len(node_candidates),
                                "num_positive": len(positives),
                                "num_negative": len(negatives),
                                "num_pairs": num_pairs,
                            }
                        )
                    )
                    counts[f"nodes_{split}"] += 1
                    counts[f"candidates_{split}"] += len(node_candidates)
                    counts[f"positive_candidates_{split}"] += len(positives)
                    counts[f"negative_candidates_{split}"] += len(negatives)
                    counts[f"derived_pairs_{split}"] += num_pairs
                    counts["max_candidates_per_node"] = max(
                        counts["max_candidates_per_node"], len(node_candidates)
                    )
                    counts["max_pairs_per_node"] = max(
                        counts["max_pairs_per_node"], num_pairs
                    )

                selected_text, _ = selected_step_text_and_rating(
                    step,
                    include_human_completion_as_positive=include_human_completion_as_positive,
                )
                if selected_text is not None:
                    prefix.append(selected_text)
                else:
                    trajectory_text, _ = trajectory_step_text_and_rating(
                        record,
                        zero_based_index,
                        step,
                        include_human_completion_as_positive=include_human_completion_as_positive,
                    )
                    if trajectory_text is not None:
                        prefix.append(trajectory_text)

    summary = _materialization_summary("nodes_v0", counts, max_records)
    summary["cohort"] = {
        "requires_positive_and_negative": True,
        "drop_conflicting_binary_texts": drop_conflicting_binary_texts,
        "outer_sampling_unit": "annotated_exact_prefix_node",
    }
    return summary


def _materialization_summary(kind: str, counts: Counter[str], max_records: int | None) -> dict:
    return {
        "version": 1,
        "kind": kind,
        "max_raw_records": max_records,
        "counts": dict(sorted(counts.items())),
    }


def write_materialization_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
