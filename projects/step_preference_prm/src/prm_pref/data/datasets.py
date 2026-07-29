from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from prm_pref.data.input_packing import InputPacker
from prm_pref.utils.text_format import format_context


class IndexedJsonlDataset(Dataset):
    """Memory-light random access over a JSONL file using byte offsets."""

    def __init__(
        self,
        path: Path,
        *,
        max_examples: int | None = None,
        seed: int = 42,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self._handle = None
        offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        if max_examples is not None and max_examples < len(offsets):
            # A single seeded permutation makes smaller label budgets strict
            # subsets of larger budgets for the same seed.
            random.Random(seed).shuffle(offsets)
            offsets = offsets[:max_examples]
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def _tokenize_candidates(
    tokenizer: Any,
    records: list[dict],
    candidate_key: str,
    max_length: int,
) -> dict:
    contexts = [format_context(item["problem"], item.get("prefix", [])) for item in records]
    candidates = [item[candidate_key] for item in records]
    # Tokenize as a text pair and truncate only the context.  The candidate is
    # the object being judged and must not disappear at max_seq_len=512.
    return tokenizer(
        contexts,
        candidates,
        padding=True,
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt",
    )


class PointwiseCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: list[dict]) -> dict:
        import torch

        return {
            "tokens": _tokenize_candidates(
                self.tokenizer, records, "candidate", self.max_length
            ),
            "labels": torch.tensor([item["label"] for item in records], dtype=torch.float32),
        }


class PairwiseCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: list[dict]) -> dict:
        import torch

        return {
            "positive_tokens": _tokenize_candidates(
                self.tokenizer, records, "positive", self.max_length
            ),
            "negative_tokens": _tokenize_candidates(
                self.tokenizer, records, "negative", self.max_length
            ),
            "weights": torch.tensor(
                [item.get("weight", 1.0) for item in records], dtype=torch.float32
            ),
        }


class CandidateCollator:
    """Collate flat candidate records through the configured input packer."""

    def __init__(
        self,
        packer: InputPacker,
        *,
        candidate_key: str = "candidate",
        include_labels: bool = True,
    ) -> None:
        self.packer = packer
        self.candidate_key = candidate_key
        self.include_labels = include_labels

    def __call__(self, records: list[dict]) -> dict:
        import torch

        examples = [
            {
                "problem": item["problem"],
                "prefix": item.get("prefix", []),
                "candidate": item[self.candidate_key],
            }
            for item in records
        ]
        packed = self.packer.pack(examples)
        result = {
            "tokens": packed.tokens,
            "telemetry": packed.telemetry,
        }
        if self.include_labels:
            result["labels"] = torch.tensor(
                [item["label"] for item in records],
                dtype=torch.float32,
            )
        return result


class ExactPrefixNodeCollator:
    """Flatten candidates while preserving their exact-prefix node index."""

    def __init__(self, packer: InputPacker) -> None:
        self.packer = packer

    def __call__(self, records: list[dict]) -> dict:
        import torch

        examples: list[dict] = []
        labels: list[int] = []
        node_indices: list[int] = []
        node_offsets = [0]
        for node_index, record in enumerate(records):
            for candidate in record["candidates"]:
                examples.append(
                    {
                        "problem": record["problem"],
                        "prefix": record.get("prefix", []),
                        "candidate": candidate["text"],
                    }
                )
                labels.append(int(candidate["label"]))
                node_indices.append(node_index)
            node_offsets.append(len(examples))

        packed = self.packer.pack(examples)
        return {
            "tokens": packed.tokens,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "node_indices": torch.tensor(node_indices, dtype=torch.long),
            "node_offsets": torch.tensor(node_offsets, dtype=torch.long),
            "node_ids": [str(record["context_id"]) for record in records],
            "problem_ids": [str(record["problem_id"]) for record in records],
            "num_nodes": len(records),
            "num_candidates": len(examples),
            "num_pairs": sum(int(record["num_pairs"]) for record in records),
            "telemetry": packed.telemetry,
        }
