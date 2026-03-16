# Persistence (pers) Extreme Returns — Investigation Summary

## Objective

Investigate why the Persistence method shows high cumulative equity (e.g. >10 in some runs or near 10) and whether any dates have unrealistic returns.

## Data

- **Source:** `results/regime_covariance_backtest.parquet` (or `.csv`)
- **Structure:** One row per evaluation date; columns `pers_gmvp_cumret`, `roll_gmvp_cumret`, `pers_gmvp_mean`, `roll_gmvp_mean`, etc.

## Findings (from current backtest file)

### 1. Cumulative equity

- **pers equity** (cumprod(1 + pers_gmvp_cumret)): **~9.12** at end of sample (not >10 in this run, but high).
- **roll equity:** ~3.89.
- **Ratio pers/roll:** ~2.34 (pers builds up more wealth over time).

### 2. No single-date 3x divergence

- **Suspicious dates** defined as: same-day `(1 + pers_gmvp_cumret) / (1 + roll_gmvp_cumret)` **> 3 or < 1/3**.
- In the current backtest: **no such dates**. Pers does not beat roll by 3x on any single evaluation date; the high pers equity comes from **many dates** where pers is slightly better (or less bad) than roll.

### 3. When pers does best

- **Largest pers_gmvp_cumret** (horizon returns) occur in **2020** (COVID rebound): e.g. 2020-03-23 (28.8%), 2020-03-30 (20%), 2020-03-16 (18.6%), 2020-04-06 (18%).
- 2021: e.g. 2021-03-05 pers=8.5%, roll=9.2% (pers and roll similar).
- **2021** pers_gmvp_cumret: min ≈ -4.2%, max ≈ 8.5%, mean ≈ 2.3% — **realistic** for a 20-day horizon.

### 4. gmvp_mean (per-horizon daily mean)

- pers_gmvp_mean and roll_gmvp_mean are in a plausible range (small decimals). No obvious bad spikes from this check.

### 5. Conclusion

- **Persistence** has **higher cumulative equity** than roll in this backtest (~9.1 vs ~3.9) because it often has somewhat better (or less bad) horizon returns over time, **not** because of a few extreme days with >3x same-day ratio.
- If in another run the **pers equity exceeds 10**, it would be the same mechanism: chaining many slightly better horizon returns. To curb that you could: cap horizon returns, use a different persistence implementation, or report Sharpe/turnover instead of (or in addition to) chained equity.

## How to reproduce

Run the investigation script (requires pandas, optionally matplotlib for the plot):

```bash
python scripts/analysis/investigate_pers_extreme_returns.py
```

This will:

1. Load the backtest and report regime/cumret stats.
2. Print suspicious dates (same-day pers/roll ratio > 3 or < 1/3).
3. Print 2021 comparison and extreme |pers_gmvp_mean|.
4. Optionally save `results/figs_regime_covariance/pers_vs_roll_investigation.png` (pers vs roll gmvp_cumret and cumulative equity over time).

## Suspicious dates (summary)

- **With 3x same-day criterion:** In the current file there are **no** dates where pers (1+cumret) / roll (1+cumret) is > 3 or < 1/3.
- For a **stricter** criterion (e.g. ratio > 1.5 on same day), re-run the script and inspect its output or add that threshold in the script.
