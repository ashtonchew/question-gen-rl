"""Convert backend_roles.json to SkyRL parquet format."""
import json
import pandas as pd
from pathlib import Path
import argparse


def format_prompt(role: dict) -> str:
    """Format role into training prompt."""
    # Handle both old schema (domain) and new schema (focus, stack)
    focus = role.get('focus', role.get('domain', 'backend'))
    stack = role.get('stack', [])
    stack_str = ', '.join(stack) if stack else 'Not specified'

    return f"""You are a technical recruiter creating screening questions.

## Role: {role['title']}
**ID:** {role['id']}
**Level:** {role['level']}
**Focus Area:** {focus}
**Tech Stack:** {stack_str}

**Description:** {role['description']}

**Key Skills:** {', '.join(role['key_skills'])}

---

Generate ONE technical screening question for this role. The question should:
- Be answerable in 2-5 minutes
- Test practical knowledge, not trivia
- Be appropriate for the seniority level

Question:"""


def main(input_path: str, output_dir: str, train_split: float = 0.8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load roles
    with open(input_path) as f:
        roles = json.load(f)

    # Convert to SkyRL format
    # SkyRL expects: prompt (str) and optionally other metadata
    records = []
    for role in roles:
        # Handle both old and new schema
        focus = role.get('focus', role.get('domain', 'backend'))
        records.append({
            "prompt": format_prompt(role),
            "role_id": role["id"],
            "role_title": role["title"],
            "role_level": role["level"],
            "role_focus": focus,
            # Store full role as JSON string for env access
            "role_json": json.dumps(role)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/backend_roles.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    main(args.input, args.output_dir, args.train_split)
