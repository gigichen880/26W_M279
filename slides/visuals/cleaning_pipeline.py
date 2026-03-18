"""
Vertical flowchart: 4-stage data cleaning pipeline.
Output: slides/visuals/cleaning_pipeline_fixed.png (6×11 in, 300 dpi).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "cleaning_pipeline_fixed.png"


def create_cleaning_pipeline():
    fig, ax = plt.subplots(figsize=(6, 11), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    # Plain-text box specs with explicit y and height, no dynamic layout.
    boxes = [
        {
            "y": 13.0,
            "title": "Raw Data",
            "content": "100 stocks x 3,655 days\nDaily returns (2007-2021)",
            "color": "#E0E0E0",
            "height": 1.2,
        },
        {
            "y": 11.0,
            "title": "Stage 1: Outlier Detection",
            "content": "Cell-wise MAD filtering\nFlag extreme returns\nCross-validate market moves",
            "color": "#E8F4F8",
            "height": 1.5,
        },
        {
            "y": 8.5,
            "title": "Stage 2: Missing Data",
            "content": "Pairwise-complete covariance\nUse all available pairs",
            "color": "#C0DEEC",
            "height": 1.2,
        },
        {
            "y": 6.0,
            "title": "Stage 3: Window Validation",
            "content": "Skip windows if >30% NA\nRequire minimum quality",
            "color": "#98C8E0",
            "height": 1.2,
        },
        {
            "y": 3.5,
            "title": "Stage 4: SPD Projection",
            "content": "Require 80% stock coverage\nEigenvalue clipping (e=1e-4)\nEnsure positive definite",
            "color": "#84BCE0",
            "height": 1.8,
        },
        {
            "y": 1.2,
            "title": "Clean Covariance Matrices",
            "content": "Ready for forecasting",
            "color": "#90EE90",
            "height": 1.0,
        },
    ]

    # Draw boxes and labels
    for box_spec in boxes:
        y = box_spec["y"]
        h = box_spec["height"]
        # Box
        box = FancyBboxPatch(
            (1.5, y - h / 2.0),
            7.0,
            h,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor=box_spec["color"],
            linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(box)

        # Title (plain text)
        ax.text(
            5.0,
            y + h / 2.0 - 0.15,
            box_spec["title"],
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            family="sans-serif",
            zorder=3,
        )

        # Content (plain text, no italics)
        ax.text(
            5.0,
            y - 0.30,
            box_spec["content"],
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            family="sans-serif",
            zorder=3,
        )

    # Simple arrows between box centers, with symmetric vertical padding.
    arrow_pairs = [
        (13.0, 11.0),
        (11.0, 8.5),
        (8.5, 6.0),
        (6.0, 3.5),
        (3.5, 1.2),
    ]
    for y_start, y_end in arrow_pairs:
        arrow = FancyArrowPatch(
            (5.0, y_start - 0.7),
            (5.0, y_end + 0.7),
            arrowstyle="->",
            mutation_scale=30,
            linewidth=3.0,
            color="black",
            zorder=1,
        )
        ax.add_patch(arrow)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {OUT}")
    return OUT


if __name__ == "__main__":
    create_cleaning_pipeline()
