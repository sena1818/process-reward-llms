from __future__ import annotations

from typing import Iterable


def format_context(problem: str, prefix: Iterable[str]) -> str:
    steps = list(prefix)
    previous = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    if not previous:
        previous = "(none)"
    return f"Problem:\n{problem}\n\nPrevious steps:\n{previous}"


def format_reward_input(problem: str, prefix: Iterable[str], candidate: str) -> str:
    return f"{format_context(problem, prefix)}\n\nCandidate step:\n{candidate}"
