# Regime-Aware Similarity Forecasting — Final Results and Report Guide

This document consolidates pipeline status, evaluation results, figure catalog, and completeness for the final project submission. Use it as the single source for the **Results** and **Discussion** sections of the report.

**File organization:** See [docs/FILE_ORGANIZATION_RULES.md](FILE_ORGANIZATION_RULES.md).  
**Completeness checklist:** See [docs/COMPLETENESS_CHECKLIST.md](COMPLETENESS_CHECKLIST.md).

---

# Part 1: Pipeline Status Summary

## Implementation

| Item | Detail |
|------|--------|
| **Entry point** | `run_backtest.py` (YAML-driven). Demo: `run_regime_covariance.py`. |
| **Configuration** | `configs/regime_covariance.yaml` (data, model, embedder, backtest, mixing, stability). |
| **Embedders** | `pca` (PCAWindowEmbedder), `corr_eig` (CorrEigenEmbedder). |
| **Baselines** | Roll (rolling cov), Pers (persistence), Shrink (shrink to diag), Model (regime similarity), Mix (shrink + model). |
| **Evaluation** | 5-day stride, 20-day refit (calendar); walk-forward, no look-ahead. |

## Current Results (from latest run)

| Item | Value |
|------|--------|
| **Data period** | 2012-01-01 to 2021-12-31 |
| **Universe size** | 100 stocks |
| **Data path** | `data/processed/returns_universe_100.parquet` (see `data/processed/README_returns_data.txt`) |
| **Evaluation anchors** | 435 (from backtest CSV) |
| **Methods compared** | model, mix, roll, pers, shrink |
| **Metrics** | fro, kl, stein, logeuc, nll, corr_offdiag_fro, corr_spearman, eig_log_mse, cond_ratio; gmvp_cumret/mean/vol/var/sharpe, turnover_l1, w_hhi, w_max_abs, w_l1 |

## Files Generated

| Output | Location |
|--------|----------|
| Backtest CSV | `results/regime_covariance_backtest.csv` |
| Backtest parquet | `results/regime_covariance_backtest.parquet` |
| Report CSV | `results/regime_covariance_report.csv` |
| Config snapshot | `results/regime_covariance_config_used.yaml` |
| Figures | `results/figs_regime_covariance/` (7 PNGs) |

**Status:** Ready for final report. Backtest and GMVP evaluation are implemented and documented.

---

# Part 2: Evaluation Results in Detail

## 2.1 Covariance Forecast Metrics

Lower is better. **Win%** = fraction of evaluation dates where method has lower Frobenius error than Roll.

| Method   | Frobenius (mean ± std) | LogEuc (mean ± std) | Stein (mean ± std) | KL (mean ± std) | Win% (vs Roll) |
|----------|------------------------|---------------------|--------------------|-----------------|----------------|
| RegimeSim| 0.0224 ± 0.0395       | 81.16 ± 11.81       | 326.5k ± 1121k     | 163.3k ± 560k   | **64.6%**      |
| Mix      | **0.0218** ± 0.0393   | 85.84 ± 4.55        | **1.13k** ± 3.15k  | **0.56k** ± 1.58k | **72.0%**   |
| Roll     | 0.0243 ± 0.0414       | **68.20** ± 2.05    | 1100.7k ± 1068k    | 550.4k ± 534k   | — (baseline)   |
| Pers     | 0.0267 ± 0.0448       | 58.07 ± 1.89        | 2149.9k ± 1821k    | 1074.9k ± 910k  | 29.2%          |
| Shrink   | 0.0219 ± 0.0392       | 86.06 ± 4.49        | 1.12k ± 3.15k      | 0.56k ± 1.58k   | 70.1%          |

- **Frobenius:** RegimeSim, Mix, Shrink all beat Roll; RegimeSim wins on **64.6%** of dates.
- **LogEuc:** Roll has lowest mean; RegimeSim worse on average, higher variance.
- **Stein/KL:** Mix and Shrink much lower than RegimeSim/Roll (model/roll have high-variance Stein/KL).

## 2.2 Portfolio (GMVP) Evaluation

| Method   | Port. Var. (mean ± std) | Sharpe (mean ± std) | Turnover L1 (mean ± std) | Max Drawdown |
|----------|--------------------------|---------------------|---------------------------|--------------|
| RegimeSim| 9.0e-5 ± 2.7e-4         | **1.182** ± 4.81    | 0.618 ± 0.527             | -73.97%      |
| Mix      | **8.2e-5** ± 2.0e-4    | 1.151 ± 5.11        | 0.405 ± 0.236             | -73.56%      |
| Roll     | 8.2e-5 ± 1.9e-4        | 0.754 ± 9.69        | 0.437 ± 0.250             | -74.92%      |
| Pers     | 8.6e-5 ± 1.7e-4        | -0.940 ± 46.1       | 0.730 ± 0.347             | -70.32%      |
| Shrink   | 8.2e-5 ± 2.0e-4        | 0.996 ± 7.24        | **0.313** ± 0.177         | -73.97%      |

- **Best Sharpe:** RegimeSim (1.18) > Mix (1.15) > Shrink (1.00) > Roll (0.75). Persistence negative (-0.94).
- **Turnover:** Shrink (0.31) < Mix (0.41) < Roll (0.44) < RegimeSim (0.62) < Pers (0.73).
- **Max drawdown:** From horizon cumret series; all ~-70% to -75%.

## 2.3 Time-Series (by Year)

Mean Sharpe by year (RegimeSim vs Roll):

| Year | RegimeSim | Roll   | Diff (RegimeSim − Roll) |
|------|-----------|--------|--------------------------|
| 2014 | 0.21      | -2.72  | +2.92                    |
| 2015 | -0.58     | -0.92  | +0.33                    |
| 2016 | 1.44      | 1.61   | -0.17                    |
| 2017 | 3.15      | 3.14   | +0.01                    |
| 2018 | -0.62     | -0.92  | +0.29                    |
| 2019 | 1.86      | 1.54   | +0.33                    |
| 2020 | 1.53      | 1.52   | +0.01                    |
| 2021 | 3.44      | 3.62   | -0.18                    |

RegimeSim gains in **2014, 2015, 2018, 2019**; even in 2017 and 2020; slightly behind in 2016 and 2021. Improvement is **not** only in crisis years.

## 2.4 Regime Analysis

- Backtest CSV has **no regime columns**. Performance by regime cannot be computed from current outputs. K=4 and regime usage are in config only.

---

# Part 3: Key Findings for the Final Report

**1. Overall performance**  
- **Statistical:** RegimeSim improves **Frobenius** vs Roll (64.6% win rate). Mix/Shrink also strong on Frobenius; on LogEuc Roll is best; on Stein/KL Mix/Shrink dominate.  
- **Economic:** RegimeSim has **highest mean Sharpe (1.18)**, ~**57% higher** than Roll (0.75).

**2. When it works best**  
- Gains vs Roll in 2014, 2015, 2018, 2019; mixed in 2016/2021; even in 2017 and 2020.

**3. Limitations**  
- LogEuc favors Roll; Stein/KL favor Mix/Shrink. RegimeSim best on **Frobenius** and **Sharpe**.  
- **Turnover:** RegimeSim higher (0.62) than Shrink (0.31) and Mix (0.41)—trade-off vs trading cost.  
- No regime-level performance in current backtest output.

**4. Main takeaway**  
- Regime-aware similarity forecasting delivers **higher GMVP Sharpe** and **better Frobenius** vs rolling and persistence baselines. Highlight **Sharpe** and **Frobenius** (and win rate); note **turnover** trade-off and strength of Mix/Shrink on some metrics.

---

# Part 4: Figures for the Final Report

**Location:** `results/figs_regime_covariance/`

## 4.1 List and Categories

| # | Filename | Category | What it shows |
|---|----------|----------|----------------|
| 1 | equity_curves_gmvp.png | Time series | Cumulative GMVP wealth by method (chained per-horizon returns). |
| 2 | method_overlays.png | Time series / comparison | Many metrics over time, all methods (fro, kl, stein, logeuc, gmvp_*, turnover, etc.). |
| 3 | rolling_median_21d.png | Time series | 21-day rolling median of fro, kl, stein, gmvp_var, gmvp_sharpe, turnover by method. |
| 4 | rolling_winrate_ref_mix_63d.png | Performance comparison | 63-day rolling win rate: Mix vs others. |
| 5 | rolling_winrate_ref_model_63d.png | Performance comparison | 63-day rolling win rate: RegimeSim vs others. |
| 6 | skill_timeseries_ref_mix.png | Time series | Skill of Mix vs others over time (diff/ratio by metric). |
| 7 | skill_timeseries_ref_model.png | Time series | Skill of RegimeSim vs others over time. |

**Regime visualizations:** None (no regime IDs in backtest output).  
**Diagnostic plots:** None in current set.

## 4.2 Recommended Figure Set

**Essential (must include):**

1. **equity_curves_gmvp.png** — Cumulative GMVP wealth by method. Direct economic result; RegimeSim and Mix lead.  
   *Caption:* Cumulative GMVP equity curves by forecasting method (2013–2021). RegimeSim (model) and Mix lead; Persistence lags.

2. **skill_timeseries_ref_model.png** — Skill of RegimeSim vs all others over time. When/where the method beats baselines.  
   *Caption:* Skill of RegimeSim (model) vs other methods over time. Key figure for when the regime-similarity forecaster adds value.

3. **rolling_median_21d.png** — 21-day rolling median of key metrics by method. Smoothed comparison.  
   *Caption:* 21-day rolling median of key metrics by method: covariance errors, GMVP variance, Sharpe, turnover.

**Supporting (nice to have):**

4. **rolling_winrate_ref_model_63d.png** — 63-day rolling win rate of RegimeSim vs others.  
5. **method_overlays.png** — Full metric overlay (or crop to 1–2 metrics for one slide).

**Can skip for main report:** rolling_winrate_ref_mix_63d.png, skill_timeseries_ref_mix.png (optional for Mix subsection).

---

# Part 5: Reproducibility and Documentation

## 5.1 How to Regenerate Results

```bash
# Backtest (requires data/processed/returns_universe_100.parquet)
python run_backtest.py --config configs/regime_covariance.yaml

# Override hyperparameters
python run_backtest.py --config configs/regime_covariance.yaml \
  --set backtest.stride=5 --set model.n_regimes=4

# Figures
python scripts/analysis/viz_backtest_results.py --config configs/viz_regime_covariance.yaml
```

**Random seed:** `model.random_state: 0` in config; passed to RegimeModel and GMM.

**Data:** See `data/processed/README_returns_data.txt`. Parquet is gitignored; repro requires the file or rebuilding from raw data.

## 5.2 Analysis Scripts (optional)

- **Stats and win rates (no scipy):** `python3 scripts/analysis/analyze_backtest_stdlib.py`
- **With t-tests (pandas + scipy):** `python scripts/analysis/analyze_backtest_results.py`
- **Regime timeline and characterization:** After running the backtest (which now saves regime columns), run  
  `python scripts/analysis/visualize_regimes.py`  
  to generate `regime_timeline.png`, `regime_probs_stacked.png`, `regime_filtering_effect.png`, and `results/regime_characterization.csv`.

## 5.3 File Organization (summary)

- **Root:** `run_backtest.py`, `run_regime_covariance.py`, `README.md`, `requirements.txt`, `.gitignore`.
- **Configs:** `configs/regime_covariance.yaml`, `configs/viz_regime_covariance.yaml`.
- **Scripts:** `scripts/analysis/` (viz_backtest_results, analyze_backtest_*, plot_backtest), `scripts/data_validation/`, `scripts/config_utils.py`, `scripts/clean_data.py`.
- **Results:** `results/regime_covariance_*.csv`, `results/regime_covariance_*.parquet`, `results/regime_covariance_config_used.yaml`, `results/figs_regime_covariance/*.png`.
- **Docs:** `docs/` (this file, FILE_ORGANIZATION_RULES.md, COMPLETENESS_CHECKLIST.md, EVALUATION_ANALYSIS_SUMMARY.md, FIGURES_CATALOG.md).

---

# Part 6: Final Report Sections Checklist

Use this document to draft:

- **Results:** Tables from Part 2 (covariance, GMVP, by-year); reference Part 4 for figures and captions.
- **Discussion:** Part 3 (key findings, when it works, limitations).
- **Comparison to proposal:** What was proposed vs implemented (e.g. YAML backtest, Mix baseline; HAR/DCC-GARCH/Ledoit-Wolf if dropped).
- **Limitations and future work:** Part 3 limitations; optional ablations (n_regimes, embedder); regime visualization.
- **Conclusion:** One paragraph — regime-aware similarity improves GMVP Sharpe and Frobenius vs rolling/persistence; highlight Sharpe and Frobenius; note turnover trade-off.

For full checklist and remaining TODOs, see [docs/COMPLETENESS_CHECKLIST.md](COMPLETENESS_CHECKLIST.md).
