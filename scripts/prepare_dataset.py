"""Convert backend_roles.json to raw parquet format.

Stores raw role data only - prompt formatting is separate (format_prompts.py).

Workflow:
    1. python scripts/prepare_dataset.py   # Creates data/raw/*.parquet (run once)
    2. python scripts/format_prompts.py    # Creates data/processed/*.parquet (run when prompts change)
    3. python -m src.recruiter.main        # Train
"""
import json
import pandas as pd
from pathlib import Path
import argparse


def main(input_path: str, output_dir: str, train_split: float = 0.8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load roles
    with open(input_path) as f:
        roles = json.load(f)

    # Store raw role data only - no prompt formatting here
    records = []
    for role in roles:
        focus = role.get('focus', role.get('domain', 'backend'))
        records.append({
            "role_json": json.dumps(role),
            "role_id": role["id"],
            "role_title": role["title"],
            "role_level": role["level"],
            "role_focus": focus,
        })

    df = pd.DataFrame(records)

    # Split train/test
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Save as parquet
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Created {len(train_df)} training examples")
    print(f"Created {len(test_df)} test examples")
    print(f"Saved to {output_dir}")
    print(f"\nNext: run 'python scripts/format_prompts.py' to format prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/backend_roles.json")
    parser.add_argument("--output_dir", default="data/raw")  # Changed to data/raw
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    main(args.input, args.output_dir, args.train_split)
