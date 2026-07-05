"""
Transaction-cost-adjusted GMVP performance (report 2.2 / B1).

GMVP weights are cost-unaware, so transaction costs do not change the weights
or turnover -- they only reduce realized returns by (turnover x cost) at each
rebalance. Net performance is therefore an EXACT function of the saved
per-rebalance mean return, volatility, and turnover (re-running the backtest
with costs would reproduce identical weights). We amortize the one-time
rebalance cost over the H-day holding period.

Run:
    .venv/bin/python -m scripts.analysis.core.transaction_cost_analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKTEST = REPO_ROOT / "results" / "regime_covariance" / "backtest.csv"
H = 20                       # forecast horizon / holding period (config: model.horizon)
ANN = 252.0
METHODS = ["model", "mix", "shrink", "roll", "pers"]
BPS_LEVELS = [0.0, 5.0, 10.0]


def net_stats(df, method, bps):
    c = bps / 1e4
    sub = df[[f"{method}_gmvp_mean", f"{method}_gmvp_vol", f"{method}_gmvp_cumret",
              f"{method}_turnover_l1", f"{method}_gmvp_sharpe"]].dropna()
    mu = sub[f"{method}_gmvp_mean"].values
    vol = sub[f"{method}_gmvp_vol"].values
    cum = sub[f"{method}_gmvp_cumret"].values
    to = sub[f"{method}_turnover_l1"].values
    cost = c * to                              # one-time cost per rebalance
    # net per-rebalance Sharpe (amortize cost over H days)
    mu_net = mu - cost / H
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_net = np.where(vol > 0, (mu_net / vol) * np.sqrt(ANN), np.nan)
    # net terminal wealth (compound net period returns)
    tw_net = float(np.prod(1.0 + (cum - cost)))
    return {
        "method": method, "bps": bps, "n": len(sub),
        "net_sharpe": round(float(np.nanmean(sharpe_net)), 4),
        "net_terminal_wealth": round(tw_net, 3),
        "mean_turnover": round(float(to.mean()), 4),
    }


def main():
    df = pd.read_csv(BACKTEST)
    # validate reconstruction at 0 bps vs saved sharpe
    chk = net_stats(df, "model", 0.0)["net_sharpe"]
    saved = round(float(df["model_gmvp_sharpe"].mean()), 4)
    print(f"[validation] reconstructed model gross Sharpe={chk} vs saved mean={saved} "
          f"(match: {abs(chk-saved) < 1e-3})\n")
    rows = [net_stats(df, m, b) for b in BPS_LEVELS for m in METHODS]
    out = pd.DataFrame(rows)
    piv = out.pivot(index="method", columns="bps", values="net_sharpe").reindex(METHODS)
    piv.columns = [f"Sharpe@{int(b)}bps" for b in piv.columns]
    tw = out.pivot(index="method", columns="bps", values="net_terminal_wealth").reindex(METHODS)
    tw.columns = [f"Wealth@{int(b)}bps" for b in tw.columns]
    summary = piv.join(tw)
    summary["turnover"] = out.drop_duplicates("method").set_index("method")["mean_turnover"].reindex(METHODS)
    pd.set_option("display.width", 160)
    print(summary.to_string())
    outdir = REPO_ROOT / "results" / "regime_covariance" / "transaction_costs"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "net_sharpe_by_cost.csv")
    print(f"\nSaved -> {outdir/'net_sharpe_by_cost.csv'}")


if __name__ == "__main__":
    main()
