#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from prm_pref.data.load_prm800k import (  # noqa: E402
    candidate_entries,
    get_finish_reason,
    get_problem_text,
    get_steps,
    iter_records,
)
from prm_pref.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview raw PRM800K JSONL records.")
    parser.add_argument("--config", default="configs/audit.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def clip(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def rating_label(rating: int) -> str:
    return "+1" if rating == 1 else str(rating)


def record_preview(
    *,
    split: str,
    line_no: int,
    record: dict,
    max_steps: int,
    max_candidate_chars: int,
    include_human_completion_as_positive: bool,
    skip_flagged_completions: bool,
) -> tuple[dict, list[str]]:
    question = record.get("question") or {}
    steps = get_steps(record)

    structured = {
        "split": split,
        "line_no": line_no,
        "problem": get_problem_text(record),
        "ground_truth_answer": question.get("ground_truth_answer"),
        "finish_reason": get_finish_reason(record),
        "num_steps": len(steps),
        "steps": [],
    }

    markdown = [
        f"## {split}:{line_no}",
        "",
        f"Finish reason: `{structured['finish_reason']}`",
        "",
        "**Problem**",
        "",
        clip(structured["problem"], 900),
        "",
        f"Ground truth answer: `{structured['ground_truth_answer']}`",
        "",
        "**Step preview**",
        "",
    ]

    for step_index, step in enumerate(steps[:max_steps], start=1):
        candidates = candidate_entries(
            step,
            include_human_completion_as_positive=include_human_completion_as_positive,
            skip_flagged_completions=skip_flagged_completions,
        )
        chosen = step.get("chosen_completion")
        structured_step = {
            "step_index": step_index,
            "chosen_completion": chosen,
            "human_completion": step.get("human_completion"),
            "candidates": candidates,
        }
        structured["steps"].append(structured_step)

        markdown.append(f"### Step {step_index}")
        markdown.append("")
        markdown.append(f"Chosen completion: `{chosen}`")
        markdown.append("")
        for candidate in candidates:
            rating = candidate["rating"]
            source = candidate["source"]
            text = clip(candidate["text"], max_candidate_chars)
            markdown.append(f"- rating `{rating_label(rating)}` / {source}: {text}")
        markdown.append("")

    return structured, markdown


def main() -> int:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    raw_data_dir = resolve_project_path(config.get("raw_data_dir", "data/raw/prm800k"))
    output_dir = resolve_project_path(config.get("output_dir", "outputs/audit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split else config.get("splits", [])
    preview_cfg = config.get("sample_preview", {})
    audit_cfg = config.get("audit", {})
    limit = args.limit or int(preview_cfg.get("max_records", 3))
    max_steps = int(preview_cfg.get("max_steps_per_record", 6))
    max_candidate_chars = int(preview_cfg.get("max_candidate_chars", 280))

    include_human = bool(audit_cfg.get("include_human_completion_as_positive", True))
    skip_flagged = bool(audit_cfg.get("skip_flagged_completions", True))

    records = []
    markdown = ["# PRM800K Sample Preview", ""]

    for index, (split, line_no, record) in enumerate(iter_records(raw_data_dir, splits), start=1):
        structured, md = record_preview(
            split=split,
            line_no=line_no,
            record=record,
            max_steps=max_steps,
            max_candidate_chars=max_candidate_chars,
            include_human_completion_as_positive=include_human,
            skip_flagged_completions=skip_flagged,
        )
        records.append(structured)
        markdown.extend(md)
        if index >= limit:
            break

    if not records:
        raise RuntimeError(f"No records found in {raw_data_dir} for splits: {splits}")

    sample_json_path = output_dir / "sample_records.json"
    sample_md_path = output_dir / "sample_preview.md"
    sample_json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sample_md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(f"Wrote {sample_md_path}")
    print(f"Wrote {sample_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
