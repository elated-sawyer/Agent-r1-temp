#!/usr/bin/env python3
"""Load a val_shards chunk_*.pkl and print sft_records (spot-check).

Chunk pickles are built with the same stack as training (often torch in nested
objects). Run with the project conda env, e.g.:
  conda activate Retro_R1
  python scripts/inspect_chunk_sft.py path/to/chunk_00000.pkl
"""

from __future__ import annotations

import argparse
import pickle
import pprint
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect sft_records inside a validation chunk pickle."
    )
    parser.add_argument(
        "pkl_path",
        type=Path,
        help="Path to chunk_NNNNN.pkl under .../val_shards/",
    )
    args = parser.parse_args()
    path = args.pkl_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    recs = payload.get("sft_records", [])
    pp = pprint.PrettyPrinter(indent=2, width=120)

    print(f"len(sft_records) = {len(recs)}")

    if len(recs) > 1:
        print("\n--- sft_records[1] ---")
        pp.pprint(recs[1])
    elif len(recs) == 1:
        print("\n--- sft_records[1] ---")
        print("(skip: only one record)")

    if len(recs) > 0:
        print("\n--- sft_records[0] ---")
        pp.pprint(recs[0])
    else:
        print("\n--- sft_records[0] ---")
        print("(empty)")


if __name__ == "__main__":
    main()
