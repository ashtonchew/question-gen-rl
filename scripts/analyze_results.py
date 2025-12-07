"""Sophisticated analysis and visualization of evaluation results."""
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Display name mapping for models
MODEL_DISPLAY_NAMES = {
    "gpt-5-mini": "GPT-5 Mini",
    "claude-4-5-haiku": "Claude 4.5 Haiku",
    "grok-4-1": "Grok 4.1 Fast (non-reasoning)",
    "baseline": "Qwen3-4B (Baseline)",
    "rl": "Qwen3-4B (RL-Trained)",
}

# Metric display names
METRIC_DISPLAY_NAMES = {
    "relevance": "Relevance",
    "clarity": "Clarity",
    "discriminative": "Discriminative",
    "composite": "Composite",
}

# Color palette - light to dark blue shades, composite is darkest
METRIC_COLORS = {
    "relevance": "#A8D0F0",      # Light blue
    "clarity": "#5BA3D9",        # Medium blue
    "discriminative": "#2E7BBF", # Darker blue
    "composite": "#1A4A6E",      # Dark teal blue - stands out
}


def blend_with_white(hex_color: str, alpha: float) -> str:
    """Blend a hex color with white at given alpha to get the displayed color."""
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Blend with white (255, 255, 255)
    r_blend = int(r * alpha + 255 * (1 - alpha))
    g_blend = int(g * alpha + 255 * (1 - alpha))
    b_blend = int(b * alpha + 255 * (1 - alpha))
    return f'#{r_blend:02x}{g_blend:02x}{b_blend:02x}'


def load_results(results_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compute_statistics(results: dict) -> pd.DataFrame:
    """
    Compute mean and standard deviation for each metric per model.

    Returns DataFrame with columns: model, metric, mean, std
    """
    rows = []

    for model_name, model_data in results.items():
        # Extract per-sample scores
        sample_scores = model_data.get("results", [])

        if not sample_scores:
            # Fall back to aggregate scores if no per-sample data
            for metric in ["relevance", "clarity", "discriminative"]:
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "mean": model_data.get(f"avg_{metric}", 0),
                    "std": 0,
                })
            rows.append({
                "model": model_name,
                "metric": "composite",
                "mean": model_data.get("avg_composite", 0),
                "std": 0,
            })
        else:
            # Compute from per-sample data
            relevance_scores = [s["scores"]["relevance"] for s in sample_scores]
            clarity_scores = [s["scores"]["clarity"] for s in sample_scores]
            discriminative_scores = [s["scores"]["discriminative"] for s in sample_scores]
            composite_scores = [s["scores"]["composite"] for s in sample_scores]

            for metric, scores in [
                ("relevance", relevance_scores),
                ("clarity", clarity_scores),
                ("discriminative", discriminative_scores),
                ("composite", composite_scores),
            ]:
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                })

    return pd.DataFrame(rows)


def create_horizontal_bar_chart(
    stats_df: pd.DataFrame,
    output_path: str = "results/eval_chart.png",
    figsize: tuple = (12, 8),
) -> None:
    """
    Create a sophisticated horizontal grouped bar chart with error bars.

    Models are sorted by composite score (best at top).
    """
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    # Get unique models and sort by composite score (descending)
    composite_means = stats_df[stats_df["metric"] == "composite"].set_index("model")["mean"]
    sorted_models = composite_means.sort_values(ascending=True).index.tolist()  # ascending for bottom-to-top

    # Metrics to plot (composite last for emphasis)
    metrics = ["relevance", "clarity", "discriminative", "composite"]
    n_metrics = len(metrics)
    n_models = len(sorted_models)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positioning
    bar_height = 0.18
    group_gap = 0.15
    y_positions = np.arange(n_models)

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        metric_data = stats_df[stats_df["metric"] == metric].set_index("model")

        # Get values in sorted order
        means = [metric_data.loc[m, "mean"] for m in sorted_models]
        stds = [metric_data.loc[m, "std"] for m in sorted_models]

        # Calculate y position for this metric within the group
        y_offset = (i - (n_metrics - 1) / 2) * bar_height
        y = y_positions + y_offset

        # Style: composite gets special treatment - bold and distinct
        is_composite = metric == "composite"
        bar_alpha = 1.0 if is_composite else 0.5  # 50% opacity for non-composite
        bar_height_multiplier = 1.0 if is_composite else 0.85

        # Plot horizontal bars (no error bars for cleaner look)
        bars = ax.barh(
            y,
            means,
            height=bar_height * bar_height_multiplier,
            color=METRIC_COLORS[metric],
            edgecolor="none",  # No borders
            linewidth=0,
            alpha=bar_alpha,
            label=METRIC_DISPLAY_NAMES[metric],
        )

        # Add value annotations - all to the right of bars
        for bar, mean, std in zip(bars, means, stds):
            x_pos = bar.get_width() + 0.025
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{mean:.2f}",
                va="center",
                ha="left",
                fontsize=16 if is_composite else 10,
                fontweight="bold" if is_composite else "medium",
                color="black",
            )

    # Customize axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in sorted_models], fontsize=11)

    # X-axis: tighter range to make race look closer
    ax.set_xlim(8.0, 9.3)
    ax.set_xlabel("Score (0-10 scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Model", fontsize=12, fontweight="bold", labelpad=10)

    # Title
    ax.set_title(
        "Model Evaluation: Technical Question Generation (Backend Engineering)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend - outside plot area, reversed order (Composite first)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],  # Reverse order
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Experimental setup as research paper-style footer
    ax.text(
        0.0, -0.12,
        "*Evaluated using Grok 4.1 Fast (non-reasoning) as judge. "
        "RL model trained with GRPO + LoRA (rank 8) on 400 examples; evaluated on 100 held-out samples.",
        transform=ax.transAxes,
        ha="left",
        fontsize=8,
        color="#555555",
    )

    # Grid styling
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Add subtle background shading for alternating models
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            ax.axhspan(
                y - 0.4, y + 0.4,
                facecolor="gray",
                alpha=0.05,
                zorder=0,
            )

    # Tight layout with room for legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.82, bottom=0.12)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Chart saved to: {output_path}")


def print_summary_table(stats_df: pd.DataFrame, results: dict) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    # Pivot for display
    pivot = stats_df.pivot(index="model", columns="metric", values="mean")
    std_pivot = stats_df.pivot(index="model", columns="metric", values="std")

    # Sort by composite score
    pivot = pivot.sort_values("composite", ascending=False)

    # Print header
    print(f"\n{'Model':<25} {'Relevance':>15} {'Clarity':>15} {'Discriminative':>18} {'Composite':>15}")
    print("-" * 88)

    # Print each model
    for model in pivot.index:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        rel = f"{pivot.loc[model, 'relevance']:.2f} ± {std_pivot.loc[model, 'relevance']:.2f}"
        cla = f"{pivot.loc[model, 'clarity']:.2f} ± {std_pivot.loc[model, 'clarity']:.2f}"
        dis = f"{pivot.loc[model, 'discriminative']:.2f} ± {std_pivot.loc[model, 'discriminative']:.2f}"
        com = f"{pivot.loc[model, 'composite']:.2f} ± {std_pivot.loc[model, 'composite']:.2f}"
        print(f"{display_name:<25} {rel:>15} {cla:>15} {dis:>18} {com:>15}")

    # Print rankings
    print("\n" + "-" * 88)
    print("\nRANKINGS BY METRIC:")

    for metric in ["relevance", "clarity", "discriminative", "composite"]:
        metric_data = stats_df[stats_df["metric"] == metric].sort_values("mean", ascending=False)
        best_model = metric_data.iloc[0]["model"]
        best_score = metric_data.iloc[0]["mean"]
        display_name = MODEL_DISPLAY_NAMES.get(best_model, best_model)
        print(f"  Best {METRIC_DISPLAY_NAMES[metric]:<15}: {display_name} ({best_score:.2f})")

    # Sample counts
    print("\n" + "-" * 88)
    print("\nSAMPLE COUNTS:")
    for model, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        n_samples = data.get("num_samples", len(data.get("results", [])))
        print(f"  {display_name}: {n_samples} samples")


def analyze_eval_results(
    results_path: str = "results/eval_results.json",
    output_path: Optional[str] = None,
) -> None:
    """
    Main analysis function: loads results, computes statistics,
    creates visualization, and prints summary.

    Args:
        results_path: Path to eval_results.json
        output_path: Path for output chart (default: results/eval_chart.png)
    """
    if output_path is None:
        output_path = str(Path(results_path).parent / "eval_chart.png")

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print(f"Found {len(results)} models to analyze")

    # Compute statistics
    stats_df = compute_statistics(results)

    # Create visualization
    create_horizontal_bar_chart(stats_df, output_path)

    # Print summary
    print_summary_table(stats_df, results)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results results/eval_results.json
  python scripts/analyze_results.py --output results/custom_chart.png
"""
    )
    parser.add_argument(
        "--results",
        default="results/eval_results.json",
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--output",
        help="Output path for chart (default: results/eval_chart.png)",
    )

    args = parser.parse_args()
    analyze_eval_results(args.results, args.output)


if __name__ == "__main__":
    main()
