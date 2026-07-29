from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_SPLITS = (
    "phase1_train",
    "phase1_test",
    "phase2_train",
    "phase2_test",
)


def split_to_filename(split: str) -> str:
    return split if split.endswith(".jsonl") else f"{split}.jsonl"


def iter_split_paths(raw_data_dir: Path, splits: Iterable[str]) -> Iterator[Path]:
    for split in splits:
        yield raw_data_dir / split_to_filename(split)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def iter_records(raw_data_dir: Path, splits: Iterable[str]) -> Iterator[tuple[str, int, dict]]:
    for path in iter_split_paths(raw_data_dir, splits):
        split = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Missing PRM800K split: {path}")
        for line_no, record in read_jsonl(path):
            yield split, line_no, record


def get_problem_text(record: dict) -> str:
    question = record.get("question") or {}
    return str(question.get("problem") or "")


def get_steps(record: dict) -> list[dict]:
    label = record.get("label") or {}
    steps = label.get("steps") or []
    return steps if isinstance(steps, list) else []


def get_finish_reason(record: dict) -> str:
    label = record.get("label") or {}
    return str(label.get("finish_reason") or "unknown")


def problem_id_from_text(problem: str) -> str:
    """Return a stable identifier for an exact PRM800K problem string."""

    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()
    return f"problem_{digest[:20]}"


def sample_id(split: str, line_no: int) -> str:
    return f"{split}:{line_no}"


def normalize_rating(value: object) -> int | None:
    try:
        rating = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return rating if rating in {-1, 0, 1} else None


def human_completion_text(step: dict) -> str | None:
    """Extract text from both observed PRM800K human-completion formats."""

    human_completion = step.get("human_completion")
    if isinstance(human_completion, dict):
        text = human_completion.get("text")
    else:
        text = human_completion
    if text is None:
        return None
    text = str(text)
    return text if text.strip() else None


def selected_step_text_and_rating(
    step: dict,
    *,
    include_human_completion_as_positive: bool,
) -> tuple[str | None, int | None]:
    chosen = step.get("chosen_completion")
    completions = step.get("completions") or []

    if chosen is None:
        human_completion = human_completion_text(step)
        if include_human_completion_as_positive and human_completion:
            return human_completion, 1
        return None, None

    try:
        completion = completions[int(chosen)]
    except (IndexError, TypeError, ValueError):
        return None, None

    text = completion.get("text")
    rating = normalize_rating(completion.get("rating"))
    return (str(text), rating) if text else (None, rating)


def candidate_entries(
    step: dict,
    *,
    include_human_completion_as_positive: bool,
    skip_flagged_completions: bool,
) -> list[dict]:
    entries: list[dict] = []
    for index, completion in enumerate(step.get("completions") or []):
        if skip_flagged_completions and bool(completion.get("flagged")):
            continue
        rating = normalize_rating(completion.get("rating"))
        text = completion.get("text")
        if rating is None or not text:
            continue
        entries.append(
            {
                "text": str(text),
                "rating": rating,
                "source": "model_completion",
                "completion_index": index,
            }
        )

    human_completion = human_completion_text(step)
    if include_human_completion_as_positive and human_completion:
        entries.append(
            {
                "text": human_completion,
                "rating": 1,
                "source": "human_completion",
                "completion_index": None,
            }
        )

    return entries


def dedupe_candidates(candidates: Iterable[dict]) -> list[dict]:
    """Deduplicate candidates without collapsing conflicting human ratings."""

    seen: set[tuple[int, str]] = set()
    deduped: list[dict] = []
    for candidate in candidates:
        key = (int(candidate["rating"]), str(candidate["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def drop_conflicting_binary_candidates(candidates: Iterable[dict]) -> tuple[list[dict], int]:
    """Remove texts that have both +1 and -1 labels.

    A scalar scorer receives identical model input for identical text under the
    same prefix.  Asking it to rank that text above itself creates an
    irreducible pair, so the strict V0 node cohort excludes both observations.
    Neutral ratings do not participate in this conflict check.
    """

    candidate_list = list(candidates)
    ratings_by_text: dict[str, set[int]] = {}
    for candidate in candidate_list:
        rating = int(candidate["rating"])
        if rating in {-1, 1}:
            ratings_by_text.setdefault(str(candidate["text"]), set()).add(rating)
    conflicting_texts = {
        text for text, ratings in ratings_by_text.items() if {-1, 1}.issubset(ratings)
    }
    return (
        [
            candidate
            for candidate in candidate_list
            if str(candidate["text"]) not in conflicting_texts
        ],
        len(conflicting_texts),
    )


def trajectory_step_text_and_rating(
    record: dict,
    step_index: int,
    step: dict,
    *,
    include_human_completion_as_positive: bool,
) -> tuple[str | None, int | None]:
    """Return the step on the annotated model trajectory and its rating.

    Phase-2 records store the original model solution in
    ``question.pre_generated_steps``.  At a found-error node,
    ``chosen_completion`` is intentionally null, so consulting only the
    chosen branch would drop the actual erroneous step.  We therefore match
    the pre-generated step first and fall back to the chosen/human branch for
    phase-1 records.
    """

    question = record.get("question") or {}
    generated_steps = question.get("pre_generated_steps") or []
    if isinstance(generated_steps, list) and step_index < len(generated_steps):
        generated_text = str(generated_steps[step_index])
        for completion in step.get("completions") or []:
            if str(completion.get("text") or "") == generated_text:
                return generated_text, normalize_rating(completion.get("rating"))

    return selected_step_text_and_rating(
        step,
        include_human_completion_as_positive=include_human_completion_as_positive,
    )
