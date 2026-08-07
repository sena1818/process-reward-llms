from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch

from prm_pref.data.datasets import IndexedJsonlDataset
from prm_pref.data.input_packing import InputPacker


def _to_device(tokens: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision not in {"fp16", "bf16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def score_pointwise_loader(
    model: Any,
    loader: Any,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    model.eval()
    for batch in loader:
        with _autocast_context(device, precision):
            logits = model(**_to_device(batch["tokens"], device))
        scores.extend(logits.float().cpu().tolist())
        labels.extend(batch["labels"].int().tolist())
    return scores, labels


@torch.no_grad()
def score_pairwise_loader(
    model: Any,
    loader: Any,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> tuple[list[float], list[float]]:
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    model.eval()
    for batch in loader:
        with _autocast_context(device, precision):
            positive = model(**_to_device(batch["positive_tokens"], device))
            negative = model(**_to_device(batch["negative_tokens"], device))
        positive_scores.extend(positive.float().cpu().tolist())
        negative_scores.extend(negative.float().cpu().tolist())
    return positive_scores, negative_scores


@torch.no_grad()
def score_node_loader(
    model: Any,
    loader: Any,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> list[dict]:
    results: list[dict] = []
    model.eval()
    for batch in loader:
        with _autocast_context(device, precision):
            logits = model(**_to_device(batch["tokens"], device))
        scores = logits.float().cpu().tolist()
        labels = batch["labels"].int().tolist()
        offsets = batch["node_offsets"].tolist()
        for node_index, (start, end) in enumerate(
            zip(offsets[:-1], offsets[1:])
        ):
            results.append(
                {
                    "context_id": batch["node_ids"][node_index],
                    "problem_id": batch["problem_ids"][node_index],
                    "scores": scores[start:end],
                    "labels": labels[start:end],
                }
            )
    return results


def _read_jsonl(
    path: Path,
    max_records: int | None,
    seed: int,
) -> Iterator[dict]:
    """Stream a JSONL file, or a seeded random subset when capped.

    Trajectory files are strongly ordered -- their leading records contain
    almost no annotated first errors -- so taking a prefix would produce a
    subset whose error rate is several times lower than the split's.  Capping
    therefore samples the same way ``IndexedJsonlDataset`` does, which also
    keeps ``max_trajectories`` consistent with the other evaluation caps.
    """

    if max_records is None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    subset = IndexedJsonlDataset(path, max_examples=max_records, seed=seed)
    for index in range(len(subset)):
        yield subset[index]


@torch.no_grad()
def score_trajectories(
    *,
    model: Any,
    packer: InputPacker,
    path: Path,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
    max_trajectories: int | None = None,
    seed: int = 42,
) -> list[dict]:
    results: list[dict] = []
    pending_contexts: list[dict] = []
    pending_candidates: list[str] = []
    pending_locations: list[tuple[int, int]] = []

    def flush() -> None:
        if not pending_contexts:
            return
        examples = [
            {
                "problem": context["problem"],
                "prefix": context["prefix"],
                "candidate": candidate,
            }
            for context, candidate in zip(
                pending_contexts,
                pending_candidates,
            )
        ]
        tokens = packer.pack(examples).tokens
        with _autocast_context(device, precision):
            logits = model(**_to_device(tokens, device)).float().cpu().tolist()
        for (trajectory_index, step_index), score in zip(pending_locations, logits):
            results[trajectory_index]["scores"][step_index] = score
        pending_contexts.clear()
        pending_candidates.clear()
        pending_locations.clear()

    model.eval()
    for trajectory in _read_jsonl(path, max_trajectories, seed):
        result_index = len(results)
        steps = trajectory["steps"]
        results.append(
            {
                "sample_id": trajectory["sample_id"],
                "problem_id": trajectory["problem_id"],
                "first_error_index": trajectory.get("first_error_index"),
                "ratings": [int(step["rating"]) for step in steps],
                "scores": [0.0] * len(steps),
            }
        )
        prefix: list[str] = []
        for step_index, step in enumerate(steps):
            pending_contexts.append(
                {"problem": trajectory["problem"], "prefix": list(prefix)}
            )
            pending_candidates.append(step["text"])
            pending_locations.append((result_index, step_index))
            prefix.append(step["text"])
            if len(pending_contexts) >= batch_size:
                flush()
    flush()
    return results
