#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data"
FILES = {
    "phase1_train": {
        "filename": "phase1_train.jsonl",
        "bytes": 7_900_236,
        "sha256": "e9da6a73f827ffb9a8c0dc644c541d34ed76b3d4d1e4896ff5f7b37ddf5ae34d",
    },
    "phase1_test": {
        "filename": "phase1_test.jsonl",
        "bytes": 829_105,
        "sha256": "f4b3bc5b095e45c816453dc4d748b755c680d61d55f9895d929a335b487c727d",
    },
    "phase2_train": {
        "filename": "phase2_train.jsonl",
        "bytes": 456_135_365,
        "sha256": "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089",
    },
    "phase2_test": {
        "filename": "phase2_test.jsonl",
        "bytes": 12_240_719,
        "sha256": "6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAI PRM800K JSONL splits.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Download all PRM800K splits.")
    group.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(FILES),
        help="Specific splits to download.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/prm800k",
        help="Destination directory, relative to the project root unless absolute.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print source URLs and expected sizes without downloading.",
    )
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def print_manifest(splits: list[str]) -> None:
    for split in splits:
        meta = FILES[split]
        url = f"{BASE_URL}/{meta['filename']}"
        print(
            f"{split:12s} {human_size(meta['bytes']):>10s}  "
            f"sha256={meta['sha256']}  {url}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_split(split: str, raw_dir: Path, *, force: bool) -> None:
    meta = FILES[split]
    filename = meta["filename"]
    expected_bytes = meta["bytes"]
    expected_sha256 = meta["sha256"]
    url = f"{BASE_URL}/{filename}"
    destination = raw_dir / filename
    partial = raw_dir / f"{filename}.part"

    if destination.exists() and not force:
        size = destination.stat().st_size
        if size == expected_bytes:
            actual_sha256 = sha256(destination)
            if actual_sha256 == expected_sha256:
                print(
                    f"[skip] {filename} already exists and checksum matches "
                    f"({human_size(size)})."
                )
                return
            print(
                f"[warn] {filename} has checksum {actual_sha256}, "
                f"expected {expected_sha256}. Re-downloading."
            )
        else:
            print(
                f"[warn] {filename} exists but size is {human_size(size)}, "
                f"expected {human_size(expected_bytes)}. Re-downloading."
            )

    raw_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "step-preference-prm-data-downloader"},
    )

    print(f"[download] {filename} ({human_size(expected_bytes)})")
    start = time.time()
    downloaded = 0
    next_report = 8 * 1024 * 1024

    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report or downloaded == expected_bytes:
                elapsed = max(time.time() - start, 1e-6)
                rate = downloaded / elapsed
                print(
                    f"  {human_size(downloaded):>9s} / {human_size(expected_bytes):<9s} "
                    f"at {human_size(int(rate))}/s"
                )
                next_report += 32 * 1024 * 1024

    actual_bytes = partial.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Downloaded {filename} has size {actual_bytes}, expected {expected_bytes}."
        )
    actual_sha256 = sha256(partial)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Downloaded {filename} has sha256 {actual_sha256}, "
            f"expected {expected_sha256}."
        )
    partial.replace(destination)
    print(f"[ok] {destination}")


def main() -> int:
    args = parse_args()
    splits = sorted(FILES) if args.all or not args.splits else args.splits
    raw_dir = resolve_project_path(args.raw_dir)

    print_manifest(splits)
    if args.manifest_only:
        return 0

    for split in splits:
        download_split(split, raw_dir, force=args.force)
    print(f"\nDownloaded PRM800K files into: {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
