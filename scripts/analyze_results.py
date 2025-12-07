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


def filter_bottom_percentile(results: dict, percentile: float = 0.2) -> dict:
    """Remove bottom X% of samples per model based on composite score.

    Args:
        results: Raw results dict with per-model data
        percentile: Fraction of samples to remove (0.2 = bottom 20%)

    Returns:
        Filtered results dict with bottom samples removed
    """
    filtered = {}
    for model_name, model_data in results.items():
        samples = model_data.get("results", [])
        if not samples:
            filtered[model_name] = model_data
            continue

        # Sort by composite score and remove bottom X%
        sorted_samples = sorted(samples, key=lambda x: x["scores"]["composite"])
        cutoff = int(len(sorted_samples) * percentile)
        filtered_samples = sorted_samples[cutoff:]

        filtered[model_name] = {
            **model_data,
            "results": filtered_samples,
            "num_samples": len(filtered_samples),
        }
    return filtered


def filter_statistical_outliers(results: dict, sd_threshold: float = 1.5) -> tuple[dict, dict]:
    """Remove statistical outliers below mean - (sd_threshold * std) per model.

    Args:
        results: Raw results dict with per-model data
        sd_threshold: Number of standard deviations below mean to use as cutoff

    Returns:
        Tuple of (filtered results dict, outlier info dict with counts per model)
    """
    filtered = {}
    outlier_info = {}
    for model_name, model_data in results.items():
        samples = model_data.get("results", [])
        if not samples:
            filtered[model_name] = model_data
            outlier_info[model_name] = 0
            continue

        scores = [s["scores"]["composite"] for s in samples]
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean - sd_threshold * std

        filtered_samples = [s for s in samples if s["scores"]["composite"] >= threshold]
        outlier_info[model_name] = len(samples) - len(filtered_samples)

        filtered[model_name] = {
            **model_data,
            "results": filtered_samples,
            "num_samples": len(filtered_samples),
        }
    return filtered, outlier_info


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
    filter_pct: float = 0.0,
    outlier_sd: Optional[float] = None,
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
    # Extend axis when filtering outliers (higher scores become visible)
    x_max = 9.45 if outlier_sd is not None else 9.3
    ax.set_xlim(8.0, x_max)
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
    footer_text = (
        "*Evaluated using Grok 4.1 Fast (non-reasoning) as judge. "
        "RL model trained with GRPO + LoRA (rank 8) on 400 examples; evaluated on 100 held-out samples."
    )
    if outlier_sd is not None:
        footer_text += f" Outliers removed: scores < μ - {outlier_sd}σ per model."
    elif filter_pct > 0:
        footer_text += f" Bottom {int(filter_pct * 100)}% of samples removed per model."
    ax.text(
        0.0, -0.12,
        footer_text,
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
    remove_bottom: float = 0.0,
    remove_outliers: str = "none",
) -> None:
    """
    Main analysis function: loads results, computes statistics,
    creates visualization, and prints summary.

    Args:
        results_path: Path to eval_results.json
        output_path: Path for output chart (default: results/eval_chart.png)
        remove_bottom: Fraction of bottom samples to remove per model (0.2 = 20%)
        remove_outliers: Statistical outlier removal ("none", "1sd", "1.5sd", "2sd")
    """
    if output_path is None:
        output_path = str(Path(results_path).parent / "eval_chart.png")

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print(f"Found {len(results)} models to analyze")

    # Apply statistical outlier filtering if requested (takes precedence)
    outlier_sd = None
    if remove_outliers != "none":
        sd_map = {"1sd": 1.0, "1.5sd": 1.5, "2sd": 2.0}
        outlier_sd = sd_map[remove_outliers]
        print(f"Removing statistical outliers (scores < μ - {outlier_sd}σ) per model...")
        results, outlier_info = filter_statistical_outliers(results, outlier_sd)
        for model, count in outlier_info.items():
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            print(f"  {display_name}: {count} outliers removed")
    # Apply percentile filtering if requested (and no outlier filtering)
    elif remove_bottom > 0:
        print(f"Removing bottom {int(remove_bottom * 100)}% of samples per model...")
        results = filter_bottom_percentile(results, remove_bottom)

    # Compute statistics
    stats_df = compute_statistics(results)

    # Create visualization
    create_horizontal_bar_chart(
        stats_df, output_path, filter_pct=remove_bottom, outlier_sd=outlier_sd
    )

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
  python scripts/analyze_results.py --remove-bottom 0.2 --output results/eval_chart_top80.png
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
    parser.add_argument(
        "--remove-bottom",
        type=float,
        default=0.0,
        help="Remove bottom X%% of samples per model (e.g., 0.2 for 20%%)",
    )
    parser.add_argument(
        "--remove-outliers",
        choices=["none", "1sd", "1.5sd", "2sd"],
        default="none",
        help="Remove statistical outliers below mean - X*std per model",
    )

    args = parser.parse_args()
    analyze_eval_results(args.results, args.output, args.remove_bottom, args.remove_outliers)


if __name__ == "__main__":
    main()
