# Volatility MSE / R² metric audit (Bree B1)

> **SUPERSEDED NUMBERS (note added 2026-07-31).** The MSE/R² values below (HAR 0.176, model 0.334, etc.) are from the *full* held-out sample and predate the fix this audit recommended. The patched `eval_vol_har.py` re-run on the common sample (n = 241) is now canonical: HAR MSE **0.177** / pooled R² **0.703**; pers 0.261 / 0.494; roll 0.320 / 0.380; model **0.324** / 0.372; shrink 0.340 / 0.341. See the Canonical Numbers section of `ICAIF_SUBMISSION_TODO.md` and the paper's `tab:vol_har`. The diagnosis and definitions below remain the reference for *why* the old column was wrong.

**Date:** 2026-07-30  
**Scope:** Why the paper’s held-out vol table ranks methods differently by MSE vs \(R^2\), and what a consistent definition produces.  
**Code:** `similarity_forecast/backtests.py::eval_vol_metrics`, `run_backtest.py` (vol branch), `scripts/analysis/core/eval_vol_har.py`.  
**Artifacts:** committed `results/regime_volatility/backtest.csv` (same held-out baseline numbers as the paper table); HAR column was produced by `eval_vol_har.py` on tag `vol_oos` (not in this clone — re-run flagged below).

---

## 1. What the paper reports

Table `tab:vol_har` (held-out 2017–2021):

| Method | MSE | \(R^2\) |
|--------|-----|---------|
| HAR | **0.176** | **−0.026** |
| Persistence | 0.270 | −0.463 |
| Rolling | 0.331 | −0.383 |
| Model | 0.334 | −0.241 |
| Shrinkage | 0.350 | −0.302 |

**MSE rank (best→worst):** HAR ≺ pers ≺ roll ≺ model ≺ shrink  
**\(R^2\) rank (best→worst):** HAR ≻ **model** ≻ shrink ≻ roll ≻ **pers**

Persistence is 2nd-best on MSE and *worst* on \(R^2\); rolling beats the model on MSE but loses on \(R^2\). That cannot happen under a single pooled \(R^2=1-\mathrm{SSE}/\mathrm{SST}\) on one common sample.

These baseline MSE/\(R^2\) values match, to reported precision, the **unweighted mean of per-anchor** columns in `results/regime_volatility/backtest.csv` on dates ≥ 2017-01-01 (n=247). HAR is appended by `eval_vol_har.py` the same way (mean of per-anchor HAR metrics).

---

## 2. Exact definitions in code

### 2.1 Per-anchor metrics (`eval_vol_metrics`)

For each forecast anchor, target and prediction are length-\(N\) vectors of **log realized vol** (one entry per asset):

```text
vol_mse = nanmean( (ŷ − y)² )
SS_res  = nansum( (ŷ − y)² )
ȳ       = nanmean(y)          # cross-sectional mean of that anchor’s targets
SST     = nansum( (y − ȳ)² )
vol_r2  = 1 − SS_res / SST
```

File: `similarity_forecast/backtests.py` (~302–338).

So each anchor’s \(R^2\) is **cross-sectional**: “better than predicting today’s cross-sectional mean log-vol,” not a time-series \(R^2\).

Minor robustness note (not the ranking bug here): `SST` is computed over all finite `y`, while `SS_res` only over assets with finite `ŷ−y`. If a method had NaN predictions for some assets, SST would be too large. On `regime_volatility`, implied per-anchor SST recovered from MSE and \(R^2\) is **identical across methods** (max |log-ratio| ~ 1e−14), so all methods share the same finite sample on every anchor.

### 2.2 How the table aggregates (`eval_vol_har.py` / `report.csv`)

```text
reported_MSE  = mean_over_anchors( vol_mse_a )
reported_R²   = mean_over_anchors( vol_r2_a )   # UNWEIGHTED
```

`eval_vol_har.py` `slice_report()` (~124–134) does exactly `.mean()` on each column. `run_backtest`’s `report.csv` `vol_error_mean` is the same aggregation.

### 2.3 Second issue specific to HAR merge

`slice_report` averages model/roll/pers/shrink over **all** rows in the date slice, but HAR’s `.mean()` skips NaNs. If HAR is missing on any anchors, the printed HAR metrics are on a **different date set** than the baselines. The printout even exposes this (`n=…, HAR non-null=…`) but does not restrict the common sample. Fix: evaluate every method on `df[df.har_vol_mse.notna()]` (and ideally require a shared finite mask).

---

## 3. Cause of the ranking disagreement

**Not** different samples or different SST across methods (verified on `regime_volatility`).

**Yes — inconsistent aggregation.** For each anchor \(a\),

\[
R^2_a = 1 - \frac{n_a\,\mathrm{MSE}_a}{\mathrm{SST}_a}.
\]

With a common sample, \(\mathrm{SST}_a\) is the same for every method, so **within each day** MSE and \(R^2\) rank methods identically. Averaging across days does **not** preserve that order:

- \(\overline{\mathrm{MSE}} = \mathrm{mean}_a(\mathrm{MSE}_a)\) weights every anchor equally.
- \(\overline{R^2} = \mathrm{mean}_a(R^2_a)\) also weights every anchor equally, but \(R^2_a\) is a **nonlinear** function of \(\mathrm{MSE}_a\) with day-dependent slope \(n_a/\mathrm{SST}_a\).

High-dispersion (high-SST) days move MSE a lot and \(R^2\) less (or differently) than calm days. Persistence tends to win absolute error on high-SST days; unweighted mean \(R^2\) is dragged down by many calm days where beating the cross-sectional mean is hard. Hence: **best MSE, worst mean \(R^2\)**.

Toy (same SST schedule, two methods): mean-MSE winner and mean-\(R^2\) winner can flip; the SST-weighted / pooled \(R^2\) agrees with total SSE ranking.

The paper’s verbal gloss (“negative \(R^2\) means worse than the sample mean”) is true **per anchor** (cross-sectional mean). The **table’s** mean of those \(R^2\) values is not a pooled “vs the sample mean” statistic and must not be ranked against mean MSE as if they were.

---

## 4. Consistent definition and corrected numbers

### Recommended definition (one sample, one ranking)

On a fixed set of anchors (held-out), with equal \(n\) per anchor (true here up to floating error):

\[
\overline{\mathrm{MSE}} = \mathrm{mean}_a(\mathrm{MSE}_a)
\]
\[
R^2_{\mathrm{pooled}} = 1 - \frac{\sum_a \mathrm{SSE}_a}{\sum_a \mathrm{SST}_a}
= 1 - \frac{\sum_a n\,\mathrm{MSE}_a}{\sum_a \mathrm{SST}_a}
\]

Equivalently, \(R^2_{\mathrm{pooled}}\) is the **SST-weighted** average of per-anchor \(R^2_a\). Under this definition, MSE and \(R^2\) cannot disagree on rank when the sample is shared.

Recovered from committed `results/regime_volatility/backtest.csv`, held-out ≥ 2017-01-01 (n=247), using \(\mathrm{SST}_a \propto \mathrm{MSE}_a/(1-R^2_a)\):

| Method | Mean MSE (unchanged) | Mean \(R^2\) (broken, paper) | **Pooled \(R^2\) (corrected)** |
|--------|----------------------|------------------------------|--------------------------------|
| Persistence | **0.270** | −0.463 | **+0.467** |
| Rolling | 0.331 | −0.383 | +0.347 |
| Model | 0.334 | −0.241 | +0.340 |
| Mix | 0.342 | −0.275 | +0.324 |
| Shrinkage | 0.350 | −0.302 | +0.309 |

Under the consistent definition: **persistence beats rolling beats model on both MSE and pooled \(R^2\)**. The paper’s claim that the model’s \(R^2\) (−0.241) is better than rolling (−0.383) while losing on MSE is an artifact and should be removed.

### HAR

HAR’s mean MSE **0.176** is still the headline loss for the model (nearly half of 0.334). Its **pooled** \(R^2\) is **not recomputable here**: `results/vol_oos/` (with HAR columns / residuals) is not in the committed tree. Because HAR shares the harness target and, once restricted to a common sample, the same per-anchor SST, **HAR must also rank first on pooled \(R^2\)** if it ranks first on mean MSE on that sample.

**Re-run needed (Devansh / scai4):**

```bash
python -m scripts.analysis.core.eval_vol_har --tag-dir results/vol_oos --split 2017-01-01
```

after the `eval_vol_har.py` fix below, and paste the common-sample mean MSE + pooled \(R^2\) into the paper table. Until then, safe prose: report **MSE only** (or MSE + pooled \(R^2\) for non-HAR rows from this audit) and do not claim a mean-\(R^2\) ranking that conflicts with MSE.

---

## 5. What to change in the paper / code

1. **Table `tab:vol_har`:** replace the \(R^2\) column with **pooled \(R^2\)** (or drop \(R^2\) and keep MSE + QLIKE). Do not publish unweighted mean \(R^2\) next to mean MSE.
2. **Prose (§vol_results):** delete “model’s held-out \(R^2\) is negative … rolling …” style comparisons that rely on mean-of-window \(R^2\). Keep: HAR roughly halves the model’s MSE; persistence also beats the model on MSE; the vol transfer fails.
3. **`eval_vol_har.py`:** (a) restrict all methods to anchors where HAR is finite; (b) print pooled \(R^2\) alongside mean MSE. (Patched in-repo with this audit.)
4. Optional harness follow-up: have `run_backtest` / report emit `vol_r2_pooled` for vol tags so `report.csv` cannot silently regress.

---

## 6. Reproduction (no parquets)

```bash
.venv/bin/python - <<'PY'
import numpy as np, pandas as pd
bt = pd.read_csv('results/regime_volatility/backtest.csv', parse_dates=['date'])
df = bt[bt.date >= '2017-01-01']
for m in ['model','roll','pers','shrink']:
    mse = df[f'{m}_vol_mse']; r2 = df[f'{m}_vol_r2']
    sst = mse / (1 - r2)
    print(m, 'mean_mse', float(mse.mean()), 'mean_r2', float(r2.mean()),
          'pooled_r2', float(1 - mse.sum()/sst.sum()))
PY
```

---

## 7. Bottom line for Mihai / Friday prose

| Question | Answer |
|----------|--------|
| Is there a sample mismatch across methods? | No (on committed vol backtest). |
| Why do MSE and \(R^2\) ranks disagree? | Table averages **per-anchor** MSE and **unweighted** per-anchor \(R^2\); those aggregates need not agree. |
| Corrected ranking (baselines)? | By MSE and pooled \(R^2\): pers ≻ roll ≻ model ≻ shrink. |
| Does HAR still win? | Yes on MSE (0.176 vs model 0.334); pooled HAR \(R^2\) needs a `vol_oos` re-run to print, but cannot reverse the MSE ordering on a common sample. |
