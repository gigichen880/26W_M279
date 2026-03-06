# Figure catalog — results/figs_regime_similarity/

Catalog and assessment of all generated figures for the final report.  
*Source: `scripts/analysis/viz_backtest_results.py` + `configs/viz_regime_similarity.yaml`.*

---

## STEP 1: List and categorize

**All files in `results/figs_regime_similarity/`:**

| # | Filename | Size (approx) |
|---|----------|----------------|
| 1 | equity_curves_gmvp.png | 187 KB |
| 2 | method_overlays.png | 2.9 MB |
| 3 | rolling_median_21d.png | 756 KB |
| 4 | rolling_winrate_ref_mix_63d.png | 1.7 MB |
| 5 | rolling_winrate_ref_model_63d.png | 1.6 MB |
| 6 | skill_timeseries_ref_mix.png | 3.0 MB |
| 7 | skill_timeseries_ref_model.png | 3.0 MB |

**Categories:**

- **Time series plots:** equity_curves_gmvp, method_overlays, rolling_median_21d, rolling_winrate_ref_*, skill_timeseries_ref_*
- **Performance comparisons:** equity_curves_gmvp (methods over time), method_overlays (metrics over time by method), skill_timeseries (model/mix vs others), rolling_winrate (win rate over time)
- **Regime visualizations:** After re-running backtest and `scripts/analysis/visualize_regimes.py`: regime_timeline.png, regime_probs_stacked.png, regime_filtering_effect.png
- **Diagnostic plots:** none (no residuals, pred-vs-real, or calibration plots in current set; calibration is off in config)

---

## STEP 2: Per-figure description, importance, caption

---

### 1. equity_curves_gmvp.png

- **What it shows:** Cumulative GMVP wealth from chaining per-evaluation horizon returns (`gmvp_cumret`) for each method (model, mix, roll, pers, shrink) over the backtest dates. One line per method; x-axis = date.
- **Importance:** ★★★★★ (5/5)
- **Suggested caption:** *Cumulative GMVP equity curves by forecasting method (2013–2021). RegimeSim (model) and Mix lead; Persistence lags. Wealth is built by chaining per-horizon GMVP returns.*

---

### 2. method_overlays.png

- **What it shows:** One subplot per overlay metric (fro, kl, stein, logeuc, corr_spearman, corr_offdiag_fro, eig_log_mse, cond_ratio, gmvp_var, gmvp_vol, gmvp_sharpe, turnover_l1, w_hhi, w_max_abs, w_l1). Each subplot plots that metric over time for all methods (model, mix, roll, pers, shrink). Dense multi-panel time series.
- **Importance:** ★★★☆☆ (3/5)
- **Suggested caption:** *Time series of covariance and portfolio metrics by method (overlay). Top rows: Frobenius, KL, Stein, LogEuc; then correlation/eigen metrics; then GMVP variance/vol/Sharpe and turnover/weight stats.*

---

### 3. rolling_median_21d.png

- **What it shows:** 21-day rolling median of headline metrics (fro, kl, stein, gmvp_var, gmvp_sharpe, turnover_l1) by method. One subplot per metric; smoothed comparison over time.
- **Importance:** ★★★★☆ (4/5)
- **Suggested caption:** *21-day rolling median of key metrics by method: covariance errors (Frobenius, KL, Stein), GMVP variance, Sharpe ratio, and L1 turnover. Highlights medium-term performance differences.*

---

### 4. rolling_winrate_ref_mix_63d.png

- **What it shows:** 63-day rolling win rate of **Mix** vs each other method (model, roll, pers, shrink) for each overlay metric. Win rate = fraction of dates in the window where Mix beats the other method (higher is better for Sharpe/corr, lower for error/turnover). One subplot per metric; y-axis 0–1.
- **Importance:** ★★★☆☆ (3/5)
- **Suggested caption:** *63-day rolling win rate: Mix vs each other method for covariance and portfolio metrics. Values above 0.5 indicate Mix winning more often in that window.*

---

### 5. rolling_winrate_ref_model_63d.png

- **What it shows:** Same as (4) but with **Model (RegimeSim)** as the reference. Rolling win rate of Model vs mix, roll, pers, shrink for each overlay metric.
- **Importance:** ★★★★☆ (4/5)
- **Suggested caption:** *63-day rolling win rate: RegimeSim (model) vs other methods. Shows where the regime-similarity forecaster consistently beats baselines over time.*

---

### 6. skill_timeseries_ref_mix.png

- **What it shows:** Skill of **Mix** vs each other method over time. For “higher is better” metrics (e.g. gmvp_sharpe): skill = Mix − other (positive = Mix better). For “lower is better” (e.g. fro): skill = Mix / other (values < 1 = Mix better). One subplot per metric; reference line at 0 or 1.
- **Importance:** ★★★☆☆ (3/5)
- **Suggested caption:** *Skill of Mix vs other methods over time (difference or ratio by metric). Positive (or <1 for error metrics) indicates Mix outperforming.*

---

### 7. skill_timeseries_ref_model.png

- **What it shows:** Same as (6) but with **Model (RegimeSim)** as the reference. Skill of RegimeSim vs mix, roll, pers, shrink over time for each overlay metric.
- **Importance:** ★★★★★ (5/5)
- **Suggested caption:** *Skill of RegimeSim (model) vs other methods over time. Key figure for showing when and on which metrics the regime-similarity forecaster adds value.*

---

## STEP 3: Recommended figure set for report

**RECOMMENDED FIGURES FOR FINAL REPORT:**

**Essential (must include):**

1. **equity_curves_gmvp.png** — Shows cumulative GMVP wealth by method over the backtest period.  
   **Why:** Direct economic outcome; RegimeSim and Mix lead, Persistence lags. Best single “bottom line” chart.

2. **skill_timeseries_ref_model.png** — Shows skill of RegimeSim vs all others over time for covariance and portfolio metrics.  
   **Why:** Demonstrates when and where the proposed method beats baselines (and where it does not). Supports “64.6% win rate vs roll” and time variation.

3. **rolling_median_21d.png** — Shows 21-day rolling median of fro, kl, stein, gmvp_var, gmvp_sharpe, turnover by method.  
   **Why:** Smoothed comparison of key metrics without overwhelming detail; good for “medium-term performance” narrative.

**Supporting (nice to have):**

4. **rolling_winrate_ref_model_63d.png** — 63-day rolling win rate of RegimeSim vs others.  
   **Why:** Complements skill_timeseries with a simple “% of time model wins” view; useful for robustness discussion.

5. **method_overlays.png** — Raw time series of many metrics by method.  
   **Why:** Full detail for appendix or supplement; too dense for main slides. Can crop to 1–2 metrics (e.g. fro, gmvp_sharpe) for a single slide if needed.

**Can skip (or appendix only):**

- **rolling_winrate_ref_mix_63d.png** — Redundant with skill_timeseries_ref_mix and with rolling_winrate_ref_model for the main story. Include only if Mix vs others is a dedicated subsection.
- **skill_timeseries_ref_mix.png** — Supporting for “Mix is strong” narrative; not essential if space is limited. Prefer skill_timeseries_ref_model for the main method.

**Summary table**

| Include | Filename | Role |
|--------|----------|------|
| ✅ Essential | equity_curves_gmvp.png | Main economic result |
| ✅ Essential | skill_timeseries_ref_model.png | When/where RegimeSim wins |
| ✅ Essential | rolling_median_21d.png | Smoothed metric comparison |
| ✅ Supporting | rolling_winrate_ref_model_63d.png | Win rate over time |
| ✅ Supporting | method_overlays.png | Full metric overlay (or cropped) |
| ⏭ Can skip | rolling_winrate_ref_mix_63d.png | Redundant with model-centric plots |
| ⏭ Can skip | skill_timeseries_ref_mix.png | Optional for Mix discussion |

**Note:** Regime figures (regime_timeline.png, regime_probs_stacked.png, regime_filtering_effect.png) are created by `scripts/analysis/visualize_regimes.py` after the backtest is run with regime columns saved (see run_backtest.py and pipeline predict_at_raw_anchor return_regime=True).
