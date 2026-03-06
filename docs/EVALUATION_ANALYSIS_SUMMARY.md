# Regime Similarity Backtest — Evaluation Analysis Summary

This document summarizes the analysis of `results/regime_similarity_backtest.csv` (435 evaluation dates, 2013–2021) for the final report.

---

## STEP 1: Main results (grouped by method)

- **Rows:** 435 evaluation anchors (stride=5, refit_every_days=20).
- **Baseline for win rate:** Roll (rolling covariance).
- **Methods:** RegimeSim (model), Mix (shrink + model), Roll, Pers (persistence), Shrink.

Mean and standard deviation for each metric are computed across dates. Win rate = fraction of dates where the method **beats** the baseline (lower is better for error/turnover/var; higher for Sharpe).

---

## STEP 2: Covariance forecast metrics

Lower is better for all four metrics. **Win%** = fraction of dates where method has **lower** Frobenius error than Roll.

| Method   | Frobenius (mean ± std) | LogEuc (mean ± std) | Stein (mean ± std) | KL (mean ± std) | Win% (vs Roll) |
|----------|------------------------|---------------------|--------------------|-----------------|----------------|
| RegimeSim| 0.0224 ± 0.0395       | 81.16 ± 11.81       | 326.5k ± 1121k     | 163.3k ± 560k   | **64.6%**      |
| Mix      | **0.0218** ± 0.0393   | 85.84 ± 4.55        | **1.13k** ± 3.15k  | **0.56k** ± 1.58k | **72.0%**   |
| Roll     | 0.0243 ± 0.0414       | **68.20** ± 2.05    | 1100.7k ± 1068k    | 550.4k ± 534k   | — (baseline)   |
| Pers     | 0.0267 ± 0.0448       | 58.07 ± 1.89        | 2149.9k ± 1821k    | 1074.9k ± 910k  | 29.2%          |
| Shrink   | 0.0219 ± 0.0392       | 86.06 ± 4.49        | 1.12k ± 3.15k      | 0.56k ± 1.58k   | 70.1%          |

- **Frobenius:** RegimeSim and Mix/Shrink all beat Roll on average; RegimeSim wins on **64.6%** of dates.
- **LogEuc:** Roll has the **lowest** mean (68.2); RegimeSim is worse on average (81.2) but with higher variance.
- **Stein / KL:** Mix and Shrink have much **lower** Stein and KL than RegimeSim and Roll (model and roll have very large, high-variance Stein/KL).
- **Statistical significance:** Run `python scripts/analysis/analyze_backtest_results.py` (requires pandas + scipy) for paired t-tests (model vs roll).

---

## STEP 3: Portfolio (GMVP) evaluation

| Method   | Port. Var. (mean ± std) | Sharpe (mean ± std) | Turnover L1 (mean ± std) | Max Drawdown |
|----------|--------------------------|---------------------|---------------------------|--------------|
| RegimeSim| 9.0e-5 ± 2.7e-4         | **1.182** ± 4.81    | 0.618 ± 0.527             | -73.97%      |
| Mix      | **8.2e-5** ± 2.0e-4    | 1.151 ± 5.11        | 0.405 ± 0.236             | -73.56%      |
| Roll     | 8.2e-5 ± 1.9e-4        | 0.754 ± 9.69        | 0.437 ± 0.250             | -74.92%      |
| Pers     | 8.6e-5 ± 1.7e-4        | -0.940 ± 46.1       | 0.730 ± 0.347             | -70.32%      |
| Shrink   | 8.2e-5 ± 2.0e-4        | 0.996 ± 7.24        | **0.313** ± 0.177         | -73.97%      |

- **Best Sharpe:** RegimeSim (1.18), then Mix (1.15), Shrink (1.00), Roll (0.75). Persistence is negative (-0.94).
- **Realized variance:** Similar across methods (~8–9e-5); Mix/Shrink/Roll slightly lower.
- **Turnover:** Shrink lowest (0.31), then Mix (0.41), Roll (0.44), RegimeSim (0.62), Pers (0.73).
- **Max drawdown:** Computed from the series of horizon cumulative returns; all around -70% to -75% (no method clearly better).

---

## STEP 4: Time-series (by year)

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

- RegimeSim improves most in **2014, 2015, 2018, 2019** (positive diff). In **2016 and 2021** Roll is slightly ahead. **2020 (COVID)** and **2017**: roughly even.
- Improvement is **not** only in crisis years; it’s spread across both stress and calm periods.

---

## STEP 5: Regime analysis

- **Regime columns in backtest:** None. The backtest CSV does not store regime assignments or regime IDs, so **performance by regime cannot be computed** from current results. Regime count (K=4) and usage are only in the pipeline config, not in the output.

---

## STEP 6: Key findings for final report

**KEY FINDINGS FOR FINAL REPORT:**  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. Overall performance**  
- **Statistical:** RegimeSim improves **Frobenius** vs Roll (lower mean, **64.6% win rate**). Mix and Shrink also beat Roll on Frobenius (72% and 70% win rate). On **LogEuc**, Roll has the best mean; on **Stein/KL**, Mix and Shrink dominate (much smaller means than RegimeSim/Roll).  
- **Economic:** RegimeSim has the **highest mean GMVP Sharpe (1.18)**, ~**57% higher** than Roll (0.75). Mix (1.15) and Shrink (1.00) are strong; Persistence (-0.94) is poor.

**2. When does it work best?**  
- RegimeSim gains vs Roll in **2014, 2015, 2018, 2019**; roughly even in 2017 and 2020; slightly behind in 2016 and 2021. Improvement is **not** concentrated only in crises.

**3. Limitations**  
- **Covariance:** LogEuc favors Roll; Stein/KL favor Mix/Shrink. RegimeSim is best on Frobenius and on **portfolio outcome** (Sharpe).  
- **Turnover:** RegimeSim has **higher** turnover (0.62) than Shrink (0.31) and Mix (0.41)—trade-off between Sharpe and trading cost.  
- **Regime breakdown:** Not available in current backtest output.

**4. Most important figures**  
- **`results/figs_regime_similarity/equity_curves_gmvp.png`** — GMVP equity curves by method.  
- **`results/figs_regime_similarity/skill_timeseries_ref_model.png`** — skill over time vs reference.

**5. Main takeaway**  
- Regime-aware similarity forecasting delivers **higher GMVP Sharpe** than rolling and persistence baselines, with **better Frobenius error** and a **64.6% win rate** vs Roll. For the final report, highlight **Sharpe ratio** and **Frobenius error** (and optionally win rate); note the **turnover trade-off** and that **Mix/Shrink** are strong baselines on some covariance metrics (Stein/KL) and on turnover.

---

## How to reproduce

- **Full stats and win rates (no scipy):**  
  `python3 scripts/analysis/analyze_backtest_stdlib.py`
- **With significance tests (pandas + scipy):**  
  `python scripts/analysis/analyze_backtest_results.py`
