#!/usr/bin/env python3
"""Merge chunk sft_records with source tagging and deduplication.

Default behavior:
1. Read all chunk_*.pkl files from the requested val_shards directories.
2. Attach merge provenance to each record.
3. Deduplicate by full record content, which is safer than deduplicating by `id`
   for this dataset because the same `id` can appear with different content.

Example:
  conda run -n Retro_R1 python scripts/merge_chunk_sft_tagged_dedup.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch  # noqa: F401  # Required for unpickling tensors stored in chunks.


PROJECT_ROOT = Path("/mnt/shared-storage-user/wangzifu/Agent-r1-temp")
DEFAULT_INPUT_DIRS = (
    PROJECT_ROOT
    / "checkpoints/val/test_train_h4_10_api-A1-preview_val-train_h4_10_turns-100_noloop-True/global_step_0/val_shards",
    PROJECT_ROOT
    / "checkpoints/val/test_train_h4_10_api-A1-preview_val-train_h4_10_turns-100_noloop-False/global_step_0/val_shards",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/chunk_sft_tagged_dedup.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sft_records with source tags and deduplication."
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
    parser.add_argument(
        "--dedupe-by",
        choices=("content", "id", "none"),
        default="content",
        help=(
            "Deduplication strategy. `content` keeps only exact duplicate records "
            "collapsed together, `id` collapses records sharing the same id, "
            "and `none` disables deduplication."
        ),
    )
    return parser.parse_args()


def iter_chunk_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    chunk_files = sorted(input_dir.glob("chunk_*.pkl"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pkl files found in: {input_dir}")
    return chunk_files


def load_sft_records(chunk_path: Path) -> list[dict[str, Any]]:
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


def build_source_info(input_dir: Path, chunk_path: Path) -> dict[str, str]:
    relative_dir = input_dir.relative_to(PROJECT_ROOT)
    source_dir = str(relative_dir)
    source_dir_name = input_dir.parent.parent.name

    if "noloop-True" in source_dir:
        source_tag = "noloop-True"
    elif "noloop-False" in source_dir:
        source_tag = "noloop-False"
    else:
        source_tag = source_dir_name

    return {
        "source_dir": source_dir,
        "source_dir_name": source_dir_name,
        "source_tag": source_tag,
        "chunk_file": chunk_path.name,
    }


def make_output_record(record: dict[str, Any], source_info: dict[str, str]) -> dict[str, Any]:
    output_record = copy.deepcopy(record)
    output_record["_merge_sources"] = [source_info]
    output_record["_merge_source_dirs"] = [source_info["source_dir"]]
    output_record["_merge_source_tags"] = [source_info["source_tag"]]
    return output_record


def fingerprint_by_content(record: dict[str, Any]) -> str:
    serialized = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def fingerprint_record(record: dict[str, Any], dedupe_by: str, running_index: int) -> str:
    if dedupe_by == "content":
        return f"content:{fingerprint_by_content(record)}"
    if dedupe_by == "id":
        record_id = record.get("id")
        if record_id is None:
            raise KeyError("Encountered a record without 'id' while deduplicating by id")
        return f"id:{record_id}"
    return f"row:{running_index}"


def append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def merge_source_info(existing: dict[str, Any], source_info: dict[str, str]) -> None:
    sources = existing.setdefault("_merge_sources", [])
    if source_info not in sources:
        sources.append(source_info)

    source_dirs = existing.setdefault("_merge_source_dirs", [])
    append_unique(source_dirs, source_info["source_dir"])

    source_tags = existing.setdefault("_merge_source_tags", [])
    append_unique(source_tags, source_info["source_tag"])


def main() -> None:
    args = parse_args()
    input_dirs = tuple(path.expanduser().resolve() for path in (args.input_dirs or DEFAULT_INPUT_DIRS))
    output_path = args.output.expanduser().resolve()

    merged_records: list[dict[str, Any]] = []
    seen_records: dict[str, dict[str, Any]] = {}
    total_input_records = 0
    duplicate_records = 0
    conflicting_id_records = 0
    ids_seen_with_content: dict[str, str] = {}

    for input_dir in input_dirs:
        chunk_files = iter_chunk_files(input_dir)
        dir_total = 0
        dir_written = 0
        print(f"[dir] {input_dir}")
        print(f"  chunk files: {len(chunk_files)}")

        for chunk_path in chunk_files:
            records = load_sft_records(chunk_path)
            dir_total += len(records)
            total_input_records += len(records)
            print(f"  - {chunk_path.name}: {len(records)} records")

            for record in records:
                if not isinstance(record, dict):
                    raise TypeError(
                        f"{chunk_path} contains a non-dict sft_record: {type(record).__name__}"
                    )

                source_info = build_source_info(input_dir, chunk_path)
                content_fingerprint = fingerprint_by_content(record)
                record_id = record.get("id")
                if record_id is not None:
                    previous_fingerprint = ids_seen_with_content.get(record_id)
                    if previous_fingerprint is None:
                        ids_seen_with_content[record_id] = content_fingerprint
                    elif previous_fingerprint != content_fingerprint:
                        conflicting_id_records += 1

                dedupe_key = fingerprint_record(record, args.dedupe_by, total_input_records)
                existing = seen_records.get(dedupe_key)
                if existing is None:
                    output_record = make_output_record(record, source_info)
                    seen_records[dedupe_key] = output_record
                    merged_records.append(output_record)
                    dir_written += 1
                else:
                    duplicate_records += 1
                    merge_source_info(existing, source_info)

        print(f"  total records from dir: {dir_total}")
        print(f"  records kept from dir after dedupe: {dir_written}")

    output_payload = {
        "sft_records": merged_records,
        "merge_info": {
            "input_dirs": [str(path) for path in input_dirs],
            "dedupe_by": args.dedupe_by,
            "total_input_records": total_input_records,
            "total_output_records": len(merged_records),
            "duplicate_records_removed": duplicate_records,
            "conflicting_duplicate_ids": conflicting_id_records,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(output_payload, f)

    print(f"[done] wrote {len(merged_records)} merged records to {output_path}")
    print(
        "[summary] "
        f"input={total_input_records}, output={len(merged_records)}, "
        f"duplicates_removed={duplicate_records}, "
        f"conflicting_duplicate_ids={conflicting_id_records}"
    )


if __name__ == "__main__":
    main()
