#!/usr/bin/env python3
"""Format prompts from raw parquet data.

Run this whenever prompt format changes - no need to touch the raw data.

Usage:
    python scripts/format_prompts.py
    python scripts/format_prompts.py --raw-dir data/raw --out-dir data/processed
"""
import argparse
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recruiter.prompts import format_prompt


def format_dataset(input_path: Path, output_path: Path) -> int:
    """Format a single parquet file with prompts.

    Returns number of records processed.
    """
    df = pd.read_parquet(input_path)

    # Apply prompt formatting
    df['prompt'] = df['role_json'].apply(format_prompt)

    # Write formatted data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Format prompts from raw data")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw parquet files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for formatted data")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}")
        print("Run 'python scripts/prepare_dataset.py' first to create raw data")
        sys.exit(1)

    total = 0
    for parquet_file in raw_dir.glob("*.parquet"):
        output_path = out_dir / parquet_file.name
        count = format_dataset(parquet_file, output_path)
        print(f"Formatted {count} examples: {parquet_file.name} -> {output_path}")
        total += count

    if total == 0:
        print(f"No parquet files found in {raw_dir}")
        sys.exit(1)

    print(f"\nTotal: {total} examples formatted to {out_dir}")
    print("Ready for training!")


if __name__ == "__main__":
    main()
