from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from prm_pref.utils.text_format import format_context


@dataclass(frozen=True)
class PackedBatch:
    tokens: dict[str, Any]
    telemetry: dict[str, int]


class InputPacker(Protocol):
    """The input-format seam shared by training and evaluation."""

    def pack(self, examples: list[dict]) -> PackedBatch:
        """Pack ``problem/prefix/candidate`` examples into model tokens."""


class EncoderPairPacker:
    """Pack encoder inputs as ``context, candidate`` text pairs."""

    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def pack(self, examples: list[dict]) -> PackedBatch:
        contexts = [
            format_context(item["problem"], item.get("prefix", []))
            for item in examples
        ]
        candidates = [item["candidate"] for item in examples]
        tokens = self.tokenizer(
            contexts,
            candidates,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        token_count = int(tokens["attention_mask"].sum().item())
        return PackedBatch(
            tokens=tokens,
            telemetry={
                "examples": len(examples),
                "tokens": token_count,
                "problem_truncated": 0,
                "prefix_truncated": 0,
                "candidate_truncated": 0,
            },
        )


class CausalRecentPrefixPacker:
    """Pack causal-PRM inputs while retaining the most recent reasoning.

    The fixed order is ``problem -> recent prefix -> candidate``.  Long
    problems and candidates are capped independently, then the remaining token
    budget is assigned to the suffix of the prefix.  This prevents ordinary
    right truncation from deleting the candidate or the newest reasoning step.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        max_problem_tokens: int = 384,
        max_candidate_tokens: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if max_problem_tokens <= 0 or max_candidate_tokens <= 0:
            raise ValueError("problem and candidate token budgets must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_problem_tokens = max_problem_tokens
        self.max_candidate_tokens = max_candidate_tokens
        self._problem_header = self._encode("Problem:\n")
        self._prefix_header = self._encode("\n\nPrevious steps:\n")
        self._candidate_header = self._encode("\n\nCandidate step:\n")
        self._none_prefix = self._encode("(none)")
        self._special_tokens = int(
            tokenizer.num_special_tokens_to_add(pair=False)
        )
        fixed_headers = (
            len(self._problem_header)
            + len(self._prefix_header)
            + len(self._candidate_header)
            + self._special_tokens
        )
        if fixed_headers + 2 > max_length:
            raise ValueError("max_length is too small for the input section headers")

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _prefix_ids(self, prefix: list[str]) -> list[int]:
        if not prefix:
            return list(self._none_prefix)
        pieces = [
            f"{index}. {step}\n"
            for index, step in enumerate(prefix, start=1)
        ]
        return self._encode("".join(pieces))

    def measure(self, example: dict) -> dict[str, int]:
        problem_ids = self._encode(str(example["problem"]))
        prefix_ids = self._prefix_ids(list(example.get("prefix", [])))
        candidate_ids = self._encode(str(example["candidate"]))
        packed, telemetry = self._pack_ids(problem_ids, prefix_ids, candidate_ids)
        del packed
        return {
            **telemetry,
            "raw_problem_tokens": len(problem_ids),
            "raw_prefix_tokens": len(prefix_ids),
            "raw_candidate_tokens": len(candidate_ids),
            "raw_total_tokens": (
                len(problem_ids)
                + len(prefix_ids)
                + len(candidate_ids)
                + len(self._problem_header)
                + len(self._prefix_header)
                + len(self._candidate_header)
                + self._special_tokens
            ),
        }

    def _pack_ids(
        self,
        problem_ids: list[int],
        prefix_ids: list[int],
        candidate_ids: list[int],
    ) -> tuple[list[int], dict[str, int]]:
        content_budget = self.max_length - self._special_tokens
        fixed_headers = (
            len(self._problem_header)
            + len(self._prefix_header)
            + len(self._candidate_header)
        )
        available = content_budget - fixed_headers

        kept_problem = problem_ids[: min(len(problem_ids), self.max_problem_tokens)]
        kept_candidate = candidate_ids[
            : min(len(candidate_ids), self.max_candidate_tokens)
        ]
        if len(kept_problem) + len(kept_candidate) > available:
            # Candidate is the object being judged, so shrink the problem first.
            problem_budget = max(1, available - len(kept_candidate))
            kept_problem = kept_problem[:problem_budget]
        if len(kept_problem) + len(kept_candidate) > available:
            candidate_budget = max(1, available - len(kept_problem))
            kept_candidate = kept_candidate[:candidate_budget]

        prefix_budget = max(
            available - len(kept_problem) - len(kept_candidate),
            0,
        )
        kept_prefix = (
            prefix_ids[-prefix_budget:] if prefix_budget else []
        )
        content_ids = (
            self._problem_header
            + kept_problem
            + self._prefix_header
            + kept_prefix
            + self._candidate_header
            + kept_candidate
        )
        encoded = self.tokenizer.prepare_for_model(
            content_ids,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = list(encoded["input_ids"])
        if len(input_ids) > self.max_length:
            raise RuntimeError("Recent-prefix packer exceeded max_length")
        return input_ids, {
            "tokens": len(input_ids),
            "problem_truncated": int(len(kept_problem) < len(problem_ids)),
            "prefix_truncated": int(len(kept_prefix) < len(prefix_ids)),
            "candidate_truncated": int(len(kept_candidate) < len(candidate_ids)),
        }

    def pack(self, examples: list[dict]) -> PackedBatch:
        features: list[dict[str, list[int]]] = []
        totals = {
            "examples": len(examples),
            "tokens": 0,
            "problem_truncated": 0,
            "prefix_truncated": 0,
            "candidate_truncated": 0,
        }
        for example in examples:
            input_ids, telemetry = self._pack_ids(
                self._encode(str(example["problem"])),
                self._prefix_ids(list(example.get("prefix", []))),
                self._encode(str(example["candidate"])),
            )
            features.append({"input_ids": input_ids})
            for key, value in telemetry.items():
                totals[key] += value
        tokens = self.tokenizer.pad(
            features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return PackedBatch(tokens=tokens, telemetry=totals)


def build_input_packer(
    tokenizer: Any,
    model_config: dict[str, Any],
) -> InputPacker:
    backend = str(model_config.get("backend", "encoder"))
    max_length = int(model_config.get("max_length", 512))
    if backend == "encoder":
        return EncoderPairPacker(tokenizer, max_length=max_length)
    if backend == "causal_lora":
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Causal tokenizer needs a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        packing = dict(model_config.get("packing", {}))
        return CausalRecentPrefixPacker(
            tokenizer,
            max_length=max_length,
            max_problem_tokens=int(packing.get("max_problem_tokens", 384)),
            max_candidate_tokens=int(packing.get("max_candidate_tokens", 512)),
        )
    raise ValueError(f"Unsupported model backend: {backend}")
