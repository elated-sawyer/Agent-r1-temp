#!/usr/bin/env python3
"""Merge sft_records from validation chunk pickles into one pickle file.

Example:
  conda run -n Retro_R1 python scripts/merge_chunk_sft.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch  # noqa: F401  # Required for unpickling tensors stored in chunks.


DEFAULT_INPUT_DIRS = (
    Path(
        "/mnt/shared-storage-user/wangzifu/Agent-r1-temp/checkpoints/val/"
        "test_train_h4_10_api-A1-preview_val-train_h4_10_turns-100_noloop-True/"
        "global_step_0/val_shards"
    ),
    Path(
        "/mnt/shared-storage-user/wangzifu/Agent-r1-temp/checkpoints/val/"
        "test_train_h4_10_api-A1-preview_val-train_h4_10_turns-100_noloop-False/"
        "global_step_0/val_shards"
    ),
)
DEFAULT_OUTPUT = Path("/mnt/shared-storage-user/wangzifu/Agent-r1-temp/data/chunk_sft.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sft_records from multiple val_shards chunk directories."
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dirs",
        action="append",
        type=Path,
        help=(
            "Input val_shards directory. Can be passed multiple times. "
            "Defaults to the requested noloop=True/False directories."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output pickle path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def iter_chunk_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    chunk_files = sorted(input_dir.glob("chunk_*.pkl"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pkl files found in: {input_dir}")
    return chunk_files


def load_sft_records(chunk_path: Path) -> list[object]:
    with chunk_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"{chunk_path} does not contain a dict payload")
    if "sft_records" not in payload:
        raise KeyError(f"{chunk_path} does not contain 'sft_records'")

    records = payload["sft_records"]
    if not isinstance(records, list):
        raise TypeError(f"{chunk_path} has non-list sft_records: {type(records).__name__}")
    return records


def main() -> None:
    args = parse_args()
    input_dirs = tuple(path.expanduser().resolve() for path in (args.input_dirs or DEFAULT_INPUT_DIRS))
    output_path = args.output.expanduser().resolve()

    merged_records: list[object] = []

    for input_dir in input_dirs:
        chunk_files = iter_chunk_files(input_dir)
        dir_total = 0
        print(f"[dir] {input_dir}")
        print(f"  chunk files: {len(chunk_files)}")

        for chunk_path in chunk_files:
            records = load_sft_records(chunk_path)
            merged_records.extend(records)
            dir_total += len(records)
            print(f"  - {chunk_path.name}: {len(records)} records")

        print(f"  total records from dir: {dir_total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"sft_records": merged_records}, f)

    print(f"[done] wrote {len(merged_records)} merged records to {output_path}")


if __name__ == "__main__":
    main()
