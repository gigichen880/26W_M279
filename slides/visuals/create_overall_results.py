"""
Create grouped bar chart for Slide 11: Overall Results comparison.
Output: slides/visuals/overall_results_comparison.png (12×5 in, 300 dpi).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_overall_results_chart():
    """Create 3-panel comparison chart for Slide 11."""

    # Data (from comprehensive_baseline_comparison.csv)
    methods = ["Roll", "Pers", "Shrink", "Mix", "Model"]
    frobenius = [0.0258, 0.0280, 0.0231, 0.0230, 0.0238]
    sharpe = [0.754, -0.940, 0.996, 1.015, 1.041]
    turnover = [0.458, 0.756, 0.327, 0.398, 0.548]

    # Colors
    colors = {
        "Roll": "#95A5A6",
        "Pers": "#E74C3C",
        "Shrink": "#3498DB",
        "Mix": "#F39C12",
        "Model": "#2ECC71",
    }
    bar_colors = [colors[m] for m in methods]

    # Best performers (for gold border)
    best_fro = np.argmin(frobenius)  # Mix or Shrink (tied) – highlight Mix
    best_sharpe = np.argmax(sharpe)  # Model
    best_turn = np.argmin(turnover)  # Shrink

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    x = np.arange(len(methods))

    # Panel 1: Frobenius (lower is better)
    ax = axes[0]
    bars = ax.bar(
        x,
        frobenius,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
    )
    # highlight Mix (index 3)
    bars[3].set_edgecolor("gold")
    bars[3].set_linewidth(3)

    for bar, val in zip(bars, frobenius):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Error", fontsize=11)
    ax.set_title(
        "Frobenius Error\n(lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(frobenius) * 1.20)

    # Panel 2: GMVP Sharpe (higher is better)
    ax = axes[1]
    bars = ax.bar(
        x,
        sharpe,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
    )
    bars[best_sharpe].set_edgecolor("gold")
    bars[best_sharpe].set_linewidth(3)

    for bar, val in zip(bars, sharpe):
        height = bar.get_height()
        if height >= 0:
            y_pos = height + 0.03
            va = "bottom"
        else:
            y_pos = height - 0.03
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title(
        "GMVP Sharpe Ratio\n(higher is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.0, color="black", linewidth=1.0)

    # Panel 3: Turnover (lower is better)
    ax = axes[2]
    bars = ax.bar(
        x,
        turnover,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
    )
    bars[best_turn].set_edgecolor("gold")
    bars[best_turn].set_linewidth(3)

    for bar, val in zip(bars, turnover):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Turnover", fontsize=11)
    ax.set_title(
        "Portfolio Turnover\n(lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(turnover) * 1.20)

    fig.suptitle(
        "Performance Comparison: All Methods (2013-2021, 369 dates)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overall_results_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {out_path}")
    return fig


if __name__ == "__main__":
    create_overall_results_chart()

