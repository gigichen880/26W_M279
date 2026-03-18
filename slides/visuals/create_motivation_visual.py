"""
Publication-quality vertical comparison: Rolling Window vs Regime-Aware Similarity.
Output: motivation_comparison_vertical.png (6×11 in, 300 dpi).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ── Colorblind-friendly palette (per spec) ─────────────────────────────────
ROLL_COLORS = ["#E8F4F8", "#D4E9F2", "#C0DEEC", "#ACD3E6", "#84BCE0"]
RED_NOW = "#FF0000"
REGIME = {
    0: "#90EE90",  # Calm
    1: "#87CEEB",  # Moderate
    2: "#FFD700",  # Normal
    3: "#FF6B6B",  # Crisis
}


def _linspace(a, b, n):
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _bracket(ax, x0, x1, y, height=0.35, color="steelblue", lw=2):
    """Simple bracket under timeline spanning [x0, x1]."""
    yb = y - height
    ax.plot([x0, x0], [y - 0.05, yb], color=color, linewidth=lw)
    ax.plot([x1, x1], [y - 0.05, yb], color=color, linewidth=lw)
    ax.plot([x0, x1], [yb, yb], color=color, linewidth=lw)


def create_vertical_comparison():
    fig = plt.figure(figsize=(6, 11), facecolor="white")
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.22, top=0.96, bottom=0.04, left=0.08, right=0.95)

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 1: Rolling Window (Static)
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    ax1.text(
        5,
        9.35,
        "Rolling Window (Static)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color="#1a1a1a",
    )

    timeline_y = 6.2
    n_hist = 5
    x_start, x_end_hist = 1.15, 6.85
    xs_hist = _linspace(x_start, x_end_hist, n_hist)
    box_w, box_h = 0.95, 0.55

    for i, (xc, col, lab) in enumerate(
        zip(xs_hist, ROLL_COLORS, ["t-50", "t-40", "t-30", "t-20", "t-10"])
    ):
        bx = xc - box_w / 2
        by = timeline_y - box_h / 2
        ax1.add_patch(
            FancyBboxPatch(
                (bx, by),
                box_w,
                box_h,
                boxstyle="round,pad=0.06",
                edgecolor="steelblue",
                facecolor=col,
                linewidth=2,
            )
        )
        ax1.text(xc, timeline_y, lab, ha="center", va="center", fontsize=10, color="#1a1a1a")

    # Current: red circle + label
    x_now = 8.15
    ax1.add_patch(Circle((x_now, timeline_y), 0.28, facecolor="#FFE0E0", edgecolor=RED_NOW, linewidth=2.5))
    ax1.text(x_now, timeline_y, "t", ha="center", va="center", fontsize=11, fontweight="bold", color=RED_NOW)
    ax1.text(x_now, timeline_y - 0.85, "(now)", ha="center", fontsize=10, color=RED_NOW, fontweight="bold")

    # Bracket: fixed 50-day window over last 5 boxes
    bracket_y = timeline_y - box_h / 2 - 0.08
    x_br0 = xs_hist[0] - box_w / 2
    x_br1 = xs_hist[-1] + box_w / 2
    _bracket(ax1, x_br0, x_br1, bracket_y, height=0.4, color="steelblue", lw=2)
    ax1.text(
        (x_br0 + x_br1) / 2,
        bracket_y - 0.55,
        "Fixed 50-day window",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="steelblue",
    )

    # Arrow to output
    mid_x = 5.0
    ax1.add_patch(
        FancyArrowPatch(
            (mid_x, 4.35),
            (mid_x, 3.55),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="black",
        )
    )

    out1 = FancyBboxPatch(
        (0.85, 2.35),
        8.3,
        0.95,
        boxstyle="round,pad=0.12",
        edgecolor="black",
        facecolor="#ECECEC",
        linewidth=2,
    )
    ax1.add_patch(out1)
    ax1.text(
        5,
        2.825,
        r"$\hat{\Sigma}_{\mathrm{roll}} = \mathrm{Sample\,Cov}(\mathrm{last\;}50\mathrm{\,days})$",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1a1a1a",
    )

    ax1.text(
        5,
        1.55,
        "\u26A0  Treats all periods equally",
        ha="center",
        fontsize=10,
        style="italic",
        color="#B22222",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 2: Regime-Aware Similarity (Adaptive)
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    ax2.text(
        5,
        9.35,
        "Regime-Aware Similarity (Adaptive)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color="#1a1a1a",
    )

    regime_seq = [0, 0, 1, 3, 1, 2, 0, 3, 2, 1, 2, 2, 1, 3]
    n = len(regime_seq)
    x_hist = _linspace(0.9, 6.7, n)
    ty = 6.35
    sq = 0.38

    for xc, r in zip(x_hist, regime_seq):
        ax2.add_patch(
            FancyBboxPatch(
                (xc - sq / 2, ty - sq / 2),
                sq,
                sq,
                boxstyle="round,pad=0.04",
                edgecolor="#222222",
                facecolor=REGIME[r],
                linewidth=1.8,
            )
        )

    # Red circles on OTHER regime-3 historical boxes (similarity retrieval)
    for xc, r in zip(x_hist, regime_seq):
        if r == 3:
            ax2.add_patch(Circle((xc, ty), 0.52, fill=False, edgecolor=RED_NOW, linewidth=2.5))

    x_now2 = 8.0
    ax2.add_patch(
        FancyBboxPatch(
            (x_now2 - sq / 2, ty - sq / 2),
            sq,
            sq,
            boxstyle="round,pad=0.04",
            edgecolor=RED_NOW,
            facecolor=REGIME[3],
            linewidth=3,
        )
    )
    ax2.add_patch(Circle((x_now2, ty), 0.22, facecolor=RED_NOW, edgecolor="white", linewidth=1.5))
    ax2.text(
        x_now2,
        ty + 0.72,
        "t (now), Regime 3",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=RED_NOW,
    )

    ax2.text(
        5,
        ty - 0.95,
        "Retrieved: similar Regime 3 periods (circled)",
        ha="center",
        fontsize=9,
        style="italic",
        fontweight="bold",
        color="#B22222",
    )

    ax2.add_patch(
        FancyArrowPatch(
            (5, 4.65),
            (5, 3.85),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="black",
        )
    )

    out2 = FancyBboxPatch(
        (0.65, 2.5),
        8.7,
        1.05,
        boxstyle="round,pad=0.12",
        edgecolor="black",
        facecolor="#E8F5E9",
        linewidth=2,
    )
    ax2.add_patch(out2)
    ax2.text(
        5,
        3.25,
        r"$\hat{\Sigma}_t = \sum_k \alpha_t(k)\,\hat{\Sigma}^{(k)}$",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax2.text(
        5,
        2.78,
        "[weighted by regime]",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#333333",
    )

    ax2.text(
        5,
        1.85,
        "\u2713  Adapts to current market state",
        ha="center",
        fontsize=10,
        style="italic",
        color="#1B5E20",
    )

    # Legend
    names = ["Calm", "Moderate", "Normal", "Crisis"]
    leg_y = 0.72
    ax2.text(5, 1.2, "Regimes", ha="center", fontsize=10, fontweight="bold", color="#1a1a1a")
    total_w = 4 * 1.55 + 3 * 0.35
    start_x = 5 - total_w / 2
    for i in range(4):
        lx = start_x + i * 1.9
        ax2.add_patch(
            FancyBboxPatch(
                (lx, leg_y),
                0.32,
                0.32,
                boxstyle="round,pad=0.03",
                edgecolor="#222222",
                facecolor=REGIME[i],
                linewidth=1.5,
            )
        )
        ax2.text(lx + 0.45, leg_y + 0.16, names[i], ha="left", va="center", fontsize=9, color="#1a1a1a")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "motivation_comparison_vertical.png")
    # Exact 6"×11" at 300 dpi → 1800×3300 px (portrait)
    fig.savefig(out_path, dpi=300, facecolor="white", edgecolor="none", bbox_inches=None)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    create_vertical_comparison()
