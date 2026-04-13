"""Print the first 5 rows of each parquet dataset under reaction_pathway_search."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "reaction_pathway_search"


def print_full_prompt_one_row(path: Path | None = None, row_index: int = 0) -> None:
    """Print the complete `prompt` field for a single row (no truncation)."""
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {DATA_DIR}")
        return
    path = path or parquet_files[0]
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    prompt = df.iloc[row_index]["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    print("=" * 80)
    print(f"Full `prompt` column — file: {path.name}, row: {row_index}")
    print("-" * 80)
    if isinstance(prompt, (list, dict)):
        print(json.dumps(prompt, ensure_ascii=False, indent=2, default=str))
    else:
        print(prompt)
    print()


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {DATA_DIR}")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)

    for path in parquet_files:
        df = pd.read_parquet(path)
        print("=" * 80)
        print(f"File: {path.name}  shape={df.shape}")
        print(df.head(5))
        print()

    print_full_prompt_one_row()


if __name__ == "__main__":
    main()
