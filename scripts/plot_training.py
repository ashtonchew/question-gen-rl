"""Plot training progress from metrics JSON."""
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Match color scheme from analyze_results.py
COLORS = {
    "primary": "#1A4A6E",  # Dark teal blue
    "secondary": "#2E7BBF",  # Darker blue
    "tertiary": "#5BA3D9",  # Medium blue
    "fill": "#A8D0F0",  # Light blue
}


def load_training_data(metrics_file: str) -> dict:
    """Load training data, handling different formats."""
    with open(metrics_file) as f:
        data = json.load(f)

    # Handle training run format (with steps array)
    if "steps" in data:
        return {
            "steps": data["steps"],
            "summary": data.get("summary", {}),
            "metadata": data.get("training_run", {}),
        }

    # Handle simple list format
    if isinstance(data, list):
        return {"steps": data, "summary": {}, "metadata": {}}

    return {"steps": [], "summary": {}, "metadata": {}}


def plot_training_curve(
    metrics_file: str,
    output_path: str = "results/training_chart.png",
    title: str = "RL Training: Technical Question Generation (Backend Engineering)",
) -> None:
    """Create line chart of training progress.

    Args:
        metrics_file: Path to metrics.json
        output_path: Path to save the chart
        title: Chart title
    """
    data = load_training_data(metrics_file)
    steps_data = data["steps"]
    summary = data["summary"]

    if not steps_data:
        print("No metrics data found")
        return

    # Set up style to match analyze_results.py
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract data
    steps = [d["step"] for d in steps_data]
    train_rewards = [d.get("train_reward", d.get("reward", 0)) for d in steps_data]

    # Plot training reward
    ax.plot(
        steps,
        train_rewards,
        "o-",
        color=COLORS["tertiary"],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="Train Reward",
    )

    # Fill area under training curve
    ax.fill_between(steps, train_rewards, alpha=0.1, color=COLORS["tertiary"])

    # Check for eval scores
    eval_steps = [d["step"] for d in steps_data if d.get("eval_score") is not None]
    eval_scores = [d["eval_score"] for d in steps_data if d.get("eval_score") is not None]

    if eval_steps:
        ax.plot(
            eval_steps,
            eval_scores,
            "s-",
            color=COLORS["primary"],
            linewidth=2.5,
            markersize=8,
            label="Eval Score",
        )
        # Fill area under eval curve
        ax.fill_between(eval_steps, eval_scores, alpha=0.15, color=COLORS["primary"])

    # Styling
    ax.set_xlabel("Training Step", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Score (0-1 scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Set y-axis limits for dramatic effect
    ax.set_ylim(0.79, 0.91)

    # Set x-axis to start at 1 with no padding, tick every step
    ax.set_xlim(1, max(steps))
    ax.set_xticks(range(1, max(steps) + 1))

    # Grid styling
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Legend (upper left to avoid overlap with improvement badge)
    ax.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True)

    # Footer with training info
    if data["metadata"]:
        meta = data["metadata"]
        task_name = meta.get('task', 'N/A').replace('_', ' ').title()
        footer = f"*Training: {meta.get('total_steps', 'N/A')} steps over {meta.get('total_time', 'N/A')} | Task: {task_name} (Backend Engineering)"
        ax.text(
            0.0,
            -0.15,
            footer,
            transform=ax.transAxes,
            ha="left",
            fontsize=8,
            color="#555555",
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training progress from metrics JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_training.py --metrics results/training_20251207/metrics.json
  python scripts/plot_training.py --metrics metrics.json --output my_chart.png
""",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics.json file",
    )
    parser.add_argument(
        "--output",
        default="results/training_chart.png",
        help="Output path for chart (default: results/training_chart.png)",
    )
    parser.add_argument(
        "--title",
        default="RL Training: Technical Question Generation (Backend Engineering)",
        help="Chart title",
    )

    args = parser.parse_args()
    plot_training_curve(args.metrics, args.output, args.title)


if __name__ == "__main__":
    main()
