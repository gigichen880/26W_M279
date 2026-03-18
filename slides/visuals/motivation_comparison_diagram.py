"""
Generate side-by-side comparison diagram for motivation slide.
Shows rolling window (static) vs regime-aware (adaptive) approaches.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_motivation_diagram():
    """
    Create 2-panel diagram comparing rolling window vs regime-aware.
    Vertical layout optimized for PowerPoint slides.
    """

    fig = plt.figure(figsize=(7, 11))

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1: Rolling Window (Top)
    # ════════════════════════════════════════════════════════════════════════

    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3.2)
    ax1.axis('off')

    # Title
    ax1.text(5, 3.0, 'Rolling Window: Static Estimation',
            fontsize=13, fontweight='bold', ha='center')

    # Timeline
    timeline_y = 2.3
    ax1.plot([0.8, 9.2], [timeline_y, timeline_y], 'k-', linewidth=1.5)

    # Historical periods (boxes) - more compact
    colors_hist = ['#E8F4F8', '#D4E9F2', '#C0DEEC', '#ACD3E6', '#98C8E0']
    for i, color in enumerate(colors_hist):
        x_start = 0.8 + i*1.65
        box = FancyBboxPatch((x_start, timeline_y-0.12), 1.55, 0.24,
                            boxstyle="round,pad=0.03",
                            edgecolor='steelblue', facecolor=color,
                            linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(x_start + 0.775, timeline_y, f't-{50-i*10}',
                ha='center', va='center', fontsize=8)

    # Current time marker
    ax1.plot([9.2], [timeline_y], 'ro', markersize=12)
    ax1.text(9.2, timeline_y + 0.3, 't', ha='center', fontsize=10,
            fontweight='bold', color='red')

    # Rolling window bracket - more compact
    ax1.plot([7.5, 9.2], [timeline_y-0.4, timeline_y-0.4], 'b-', linewidth=1.5)
    ax1.plot([7.5, 7.5], [timeline_y-0.4, timeline_y-0.48], 'b-', linewidth=1.5)
    ax1.plot([9.2, 9.2], [timeline_y-0.4, timeline_y-0.48], 'b-', linewidth=1.5)
    ax1.text(8.35, timeline_y-0.65, 'Fixed 50-day\nwindow',
            ha='center', fontsize=8, color='blue', fontweight='bold')

    # Arrow to forecast
    arrow1 = FancyArrowPatch((5, timeline_y-0.95), (5, timeline_y-1.25),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=1.5, color='black')
    ax1.add_patch(arrow1)

    # Forecast output - more compact
    forecast_box1 = FancyBboxPatch((1.5, timeline_y-1.7), 7, 0.32,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor='lightgray',
                                  linewidth=1.5)
    ax1.add_patch(forecast_box1)
    ax1.text(5, timeline_y-1.54, 'Σ̂ = Sample Cov(last 50 days)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Problem annotation - more compact
    ax1.text(0.5, 0.25, '⚠ Problem:', fontsize=9, color='red',
            fontweight='bold')
    ax1.text(0.5, 0.05, 'Treats all periods equally,\ncannot adapt to regimes',
            fontsize=8, color='red', style='italic')

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2: Regime-Aware Similarity (Bottom)
    # ════════════════════════════════════════════════════════════════════════

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3.2)
    ax2.axis('off')

    # Title
    ax2.text(5, 3.0, 'Regime-Aware Similarity: Adaptive',
            fontsize=13, fontweight='bold', ha='center')

    # Timeline with regime coloring
    timeline_y = 2.3

    # Color-code by regime
    regime_colors = {
        0: '#90EE90',  # Calm (green)
        1: '#87CEEB',  # Moderate (light blue)
        2: '#FFD700',  # Normal (yellow)
        3: '#FF6B6B',  # Crisis (red)
    }

    # Historical periods with regime labels - more compact
    regime_sequence = [0, 0, 1, 3, 1, 2, 0, 3, 2, 1, 2, 2, 1, 3]
    x_positions = np.linspace(0.8, 8.7, len(regime_sequence))

    for i, (x, regime) in enumerate(zip(x_positions, regime_sequence)):
        color = regime_colors[regime]
        box = FancyBboxPatch((x-0.13, timeline_y-0.13), 0.26, 0.26,
                            boxstyle="round,pad=0.015",
                            edgecolor='black', facecolor=color,
                            linewidth=1.2, alpha=0.75)
        ax2.add_patch(box)

    # Current time marker
    ax2.plot([9.2], [timeline_y], 'ro', markersize=12)
    ax2.text(9.2, timeline_y + 0.3, 't\n(R3)', ha='center',
            fontsize=9, fontweight='bold', color='red')

    # Highlight similar periods (regime 3) - more compact circles
    similar_indices = [i for i, r in enumerate(regime_sequence) if r == 3]
    for idx in similar_indices:
        x = x_positions[idx]
        circle = plt.Circle((x, timeline_y), 0.22, fill=False,
                          edgecolor='red', linewidth=2.5)
        ax2.add_patch(circle)

    ax2.text(5, timeline_y - 0.55,
            '← Similar regime 3 periods',
            ha='center', fontsize=8, color='red',
            fontweight='bold', style='italic')

    # Arrow to forecast
    arrow2 = FancyArrowPatch((5, timeline_y-0.88), (5, timeline_y-1.18),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=1.5, color='black')
    ax2.add_patch(arrow2)

    # Forecast output - more compact
    forecast_box2 = FancyBboxPatch((0.8, timeline_y-1.64), 8.4, 0.32,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor='lightgreen',
                                  linewidth=1.5)
    ax2.add_patch(forecast_box2)
    ax2.text(5, timeline_y-1.48,
            'Σ̂ = Σ_k α_t(k) Σ̂^(k)  [regime-weighted]',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Benefit annotation - more compact
    ax2.text(0.5, 0.25, '✓ Benefit:', fontsize=9, color='green',
            fontweight='bold')
    ax2.text(0.5, 0.05, 'Adapts to market state,\nretrieves relevant patterns',
            fontsize=8, color='green', style='italic')

    # Legend for regimes - more compact
    legend_x = 7.3
    legend_y = 1.05
    ax2.text(legend_x, legend_y + 0.22, 'Regimes:',
            fontsize=8, fontweight='bold')

    for i, (regime, color) in enumerate(regime_colors.items()):
        y_pos = legend_y - i*0.17
        box = FancyBboxPatch((legend_x, y_pos-0.065), 0.12, 0.12,
                            boxstyle="round,pad=0.015",
                            edgecolor='black', facecolor=color,
                            linewidth=0.8, alpha=0.75)
        ax2.add_patch(box)

        regime_names = {0: 'Calm', 1: 'Moderate', 2: 'Normal', 3: 'Crisis'}
        ax2.text(legend_x + 0.2, y_pos, regime_names[regime],
                fontsize=7, va='center')

    # plt.tight_layout()  # Skip tight_layout to avoid potential hanging
    plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, hspace=0.15)

    # Save both versions
    plt.savefig('slides/visuals/motivation_comparison_vertical.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: slides/visuals/motivation_comparison_vertical.png")

    # Also save original name for backwards compatibility
    plt.savefig('slides/visuals/motivation_comparison.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✓ Saved: slides/visuals/motivation_comparison.png")

    return None

if __name__ == "__main__":
    import os
    os.makedirs('slides/visuals', exist_ok=True)
    create_motivation_diagram()
    # plt.show()  # Commented out to avoid blocking in non-interactive mode
