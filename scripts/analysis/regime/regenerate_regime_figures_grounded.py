"""
Regenerate the three regime figures with the grounded label map.

Grounded labels (assigned from per-regime realized volatility + average
cross-asset correlation; see paper Table 7):
    0 -> Calm Bull        (low vol, low corr, positive drift)
    1 -> High Stress      (high vol, highest corr, most crisis overlap)
    2 -> Normal           (low vol, most predictable)
    3 -> High Volatility  (highest vol, lowest corr, dispersed)

Fixes issues 1.4 (GMM title -> FCM), 1.7 (date-range mismatch), and 1.5
(figure labels now match paper Table 7).

Run:
    .venv/bin/python -m scripts.analysis.regime.regenerate_regime_figures_grounded
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKTEST = REPO_ROOT / "results" / "regime_covariance" / "backtest.csv"
OUT_FIGS = REPO_ROOT / "results" / "regime_covariance" / "figs" / "regime"
PAPER_FIGS = REPO_ROOT / "paper" / "figs"

LABELS = {0: "Calm Bull", 1: "High Stress", 2: "Normal", 3: "High Volatility"}
# green=calm, red=stress, blue=normal, orange=high-vol
COLORS = {0: "#2ca02c", 1: "#d62728", 2: "#1f77b4", 3: "#ff7f0e"}

CRISES = [
    ("2008 GFC", "2008-09-01", "2009-03-31"),
    ("EU/China", "2015-08-01", "2016-02-28"),
    ("Q4 Selloff", "2018-10-01", "2019-01-31"),
    ("COVID-19", "2020-02-01", "2020-06-30"),
]


def load():
    df = pd.read_csv(BACKTEST, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fig_timeline(df, save_paths):
    """Scatter of the dominant (hard) FCM regime assignment over time.

    The saved filtered-posterior columns are degenerate (uniform 0.25) in this
    backtest export, so we plot the hard assignment, which carries the temporal
    structure.
    """
    d = df.dropna(subset=["regime_assigned"]).copy()
    d["regime_assigned"] = d["regime_assigned"].astype(int)
    fig, ax = plt.subplots(figsize=(14, 5))
    for k in range(4):
        sub = d[d["regime_assigned"] == k]
        ax.scatter(sub["date"], sub["regime_assigned"], s=14, color=COLORS[k], label=LABELS[k], alpha=0.75)
    for name, s, e in CRISES:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        ax.axvspan(s, e, color="black", alpha=0.10, zorder=0)
        ax.text(s + (e - s) / 2, 3.55, name, ha="center", va="bottom", fontsize=8, style="italic", color="dimgray")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([LABELS[k] for k in range(4)], fontsize=10)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlim(d["date"].min(), d["date"].max())
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("FCM Regime Assignments Over Time", fontsize=14, fontweight="bold", pad=22)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()


def fig_transition(df, save_paths):
    s = df["regime_assigned"].dropna().astype(int).values
    K = 4
    C = np.zeros((K, K))
    for a, b in zip(s[:-1], s[1:]):
        C[a, b] += 1
    A = C / C.sum(axis=1, keepdims=True)
    names = [LABELS[k] for k in range(K)]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        A, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, square=True,
        linewidths=0.5, linecolor="gray", cbar_kws={"label": "Transition Probability"}, ax=ax,
    )
    ax.set_xlabel("To Regime (t)", fontsize=12, fontweight="bold")
    ax.set_ylabel("From Regime (t-1)", fontsize=12, fontweight="bold")
    ax.set_title("Regime Transition Matrix (Markov Chain)", fontsize=14, fontweight="bold", pad=16)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, rotation=0, fontsize=10)
    for i in range(K):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return A


def fig_performance(df, save_paths):
    methods = [("model", "Model"), ("shrink", "Shrink"), ("roll", "Roll"), ("pers", "Pers")]
    g = df.groupby("regime_assigned")
    regimes = [0, 1, 2, 3]
    vals = {m: [g.get_group(r)[f"{m}_gmvp_sharpe"].mean() for r in regimes] for m, _ in methods}
    x = np.arange(len(regimes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    mcolors = {"model": "#d62728", "shrink": "#1f77b4", "roll": "#ff7f0e", "pers": "#7f7f7f"}
    for i, (m, lab) in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, vals[m], w, label=lab, color=mcolors[m], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in regimes], fontsize=11)
    ax.set_ylabel("Mean GMVP Sharpe", fontsize=11)
    ax.set_title("GMVP Sharpe by Regime", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()


def write_grounded_csvs(df, A):
    """Regenerate regime_characterization.csv and transition_matrix.csv with the
    grounded labels so the canonical results stay consistent with paper Table 7."""
    import pandas as pd
    rc_dir = REPO_ROOT / "results" / "regime_covariance"
    # per-regime market stats from returns
    ret = pd.read_parquet(REPO_ROOT / "data" / "processed" / "returns_universe_100.parquet")
    ret.index = pd.to_datetime(ret.index); ret = ret.sort_index()
    mkt = ret.mean(axis=1)
    d = df.dropna(subset=["regime_assigned"]).copy()
    d["regime_assigned"] = d["regime_assigned"].astype(int)
    rows = []
    for k in range(4):
        sub = d[d["regime_assigned"] == k]
        dates = pd.to_datetime(sub["date"])
        vols, corrs = [], []
        for dt in dates:
            if dt not in ret.index:
                continue
            pos = ret.index.get_loc(dt)
            w = ret.iloc[max(0, pos - 19):pos + 1]
            corrs.append(np.nanmean(w.corr().values[np.triu_indices(w.shape[1], 1)]))
        mvol_ann = float((mkt.rolling(20).std().reindex(dates).mean()) * np.sqrt(252) * 100)
        rows.append({
            "regime": k, "label": LABELS[k], "n_days": len(sub),
            "pct_time": round(100 * len(sub) / len(d), 1),
            "realized_vol_ann_pct": round(mvol_ann, 1),
            "avg_corr": round(float(np.nanmean(corrs)), 3),
            "mean_fro": round(float(sub["model_fro"].mean()), 4),
            "mean_gmvp_sharpe": round(float(sub["model_gmvp_sharpe"].mean()), 4),
        })
    pd.DataFrame(rows).to_csv(rc_dir / "regime_characterization.csv", index=False)
    names = [LABELS[k] for k in range(4)]
    pd.DataFrame(A, index=[f"From_{n}" for n in names], columns=[f"To_{n}" for n in names]).to_csv(
        rc_dir / "transition_matrix.csv")
    print("Wrote grounded regime_characterization.csv and transition_matrix.csv")


def main():
    df = load()
    fig_timeline(df, [OUT_FIGS / "regime_timeline.png", PAPER_FIGS / "regime_timeline.png"])
    A = fig_transition(df, [OUT_FIGS.parent / "transition_matrix_heatmap.png", PAPER_FIGS / "transition_matrix_heatmap.png"])
    fig_performance(df, [OUT_FIGS.parent / "performance_by_regime_gmvp_sharpe.png", PAPER_FIGS / "performance_by_regime_gmvp_sharpe.png"])
    write_grounded_csvs(df, A)
    print("Regenerated 3 figures + 2 CSVs with grounded labels.")
    print("Transition diagonal:", {LABELS[k]: round(float(A[k, k]), 3) for k in range(4)})


if __name__ == "__main__":
    main()
