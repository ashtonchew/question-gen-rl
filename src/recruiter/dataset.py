"""Custom dataset that formats prompts at load time.

This allows changing prompt format without regenerating parquet files.
"""
import json
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

from .prompts import format_prompt


def load_and_format_dataset(parquet_paths: List[str]) -> Dataset:
    """Load parquet files and format prompts at runtime.

    Args:
        parquet_paths: List of paths to parquet files containing role_json

    Returns:
        HuggingFace Dataset with formatted prompts
    """
    dfs = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Format prompts from raw role data
    def format_row(row):
        role_json = row.get('role_json')
        if role_json:
            row['prompt'] = format_prompt(role_json)
        return row

    combined_df = combined_df.apply(format_row, axis=1)

    return Dataset.from_pandas(combined_df)


def create_formatted_parquet(input_paths: List[str], output_path: str) -> None:
    """Create a formatted parquet file from raw data.

    Use this if you need to create formatted data for SkyRL consumption.
    Run once when prompt format changes.

    Args:
        input_paths: Paths to raw parquet files with role_json
        output_path: Where to write formatted parquet
    """
    dataset = load_and_format_dataset(input_paths)
    dataset.to_parquet(output_path)
    print(f"Created formatted dataset at {output_path} with {len(dataset)} examples")
