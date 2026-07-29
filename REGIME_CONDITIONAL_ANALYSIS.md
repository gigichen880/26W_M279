# Does the model win in particular regimes or conditions?

**Context:** Mihai's suggestion (2026-07-25 email): "It is still possible to not beat
all the baselines and still publish" — test whether the method wins in particular
regimes or conditions. This document reports that analysis on the honest harness
(`results/oos_final`: daily tranched GMVP returns, renormalized weights, GFC included,
tuned 2008–2016, held out 2017–2021).

**Reproduction:** `python -m scripts.analysis.core.conditional_gmvp_analysis --tag-dir results/oos_final`

**TL;DR:** There is a real, era-consistent *descriptive* pattern — the model earns its
Sharpe in calm markets and gives it back in stress — but it is the **opposite** of the
old paper narrative, and it is **not convertible into an ex-ante trading rule**: every
honest (tuning-era-selected, held-out-tested) switching/ensemble rule lands back at
persistence's Sharpe. The one ex-ante pocket that is consistent across eras is the
"Normal" regime, where the model beats every baseline in both eras, but it is ~15–20%
of days and not statistically significant.

---

## 1. Descriptive finding (valid as characterization, NOT tradeable)

Slicing daily tranched returns by what actually happened during the window:

| Slice (n days) | model | mix | shrink | pers | roll | best |
|---|---|---|---|---|---|---|
| Non-crisis pooled (2749) | **+1.435** | +1.383 | +1.335 | +1.378 | +1.306 | **model** |
| All crisis windows pooled (576) | −0.728 | −0.619 | −0.648 | −0.584 | **−0.552** | roll |
| GFC 2008-09..2009-03 (117) | −0.342 | +0.053 | +0.278 | −0.054 | **+0.471** | roll |
| COVID 2020-02..06 (104) | −0.097 | +0.054 | −0.116 | **+0.211** | +0.060 | pers |
| OOS future-corr HIGH tercile | −1.576 | −1.480 | −1.449 | −1.489 | **−1.360** | roll |
| OOS future-corr MID tercile | **+2.974** | +2.755 | +2.594 | +2.789 | +2.437 | **model** |
| OOS future-vol LOW tercile | **+2.789** | +2.605 | +2.530 | +2.586 | +2.376 | **model** |

Terminal wealth over non-crisis days: model 6.66 vs pers 5.22 — the model's entire
cumulative advantage is earned in calm markets. In every stress window it has the
highest realized vol and among the worst Sharpes.

**Interpretation:** similarity-weighted historical neighbors are informative when
covariance structure is locally stable; when correlations spike, the immediate past
(persistence/EWMA) is the better forecast and kernel-weighted history is a drag.
This *reverses* the earlier paper claim of "superior forecast accuracy in stress."
None of these slice differences is individually significant (block-bootstrap
p ≥ 0.16 everywhere; ~40 slices examined, so the best p-values are what multiple
testing would predict under the null).

## 2. Ex-ante conditioning (the tradeable question) — negative

The `realized_vol` / `avg_corr` columns in `backtest.csv` are computed on the
**future** horizon window (`compute_horizon_cross_sectional_stats(fut)` in
`run_backtest.py`) — they are regime-labeling descriptors, and conditioning on them
is look-ahead. A naive "hybrid" using future avg_corr produces OOS Sharpe 0.773 with
p ≈ 0.01–0.03 vs mix/shrink/model — entirely an artifact. Recomputing the same stats
on **trailing** windows (20d and 50d) and redoing everything honestly:

- Trailing→future predictability is weak (corr 0.75 for vol, **0.41 for avg corr**),
  so the calm-market edge cannot be captured ex ante with these variables.
- Tercile buckets of trailing vol/corr show **no era-consistent** model−pers gap
  (signs flip between 2008–2016 and 2017–2021).
- Hybrid switch rules (model when condition favorable, else pers; 13 candidates)
  tuned strictly on 2008–2016 and tested held-out: chosen rule (regime ∈ {Calm Bull,
  Normal}) gives OOS Sharpe **0.617** vs pers 0.635 (p = 0.78). Even the best rule
  in hindsight only reaches 0.709.
- Capital-split ensembles don't help either: model and pers daily returns are
  **0.954 correlated**, the Sharpe-vs-weight frontier is monotone in favor of pers,
  and the per-regime-tuned ensemble gives OOS 0.621 ≈ pers (p = 0.82).
- Regime confidence, and time-since-regime-switch: no pocket favoring the model.

## 3. The one era-consistent ex-ante pocket: regime "Normal"

Model − pers Sharpe gap by (ex-ante, filtered) regime:

| Regime | tuning 08–16 | held-out 17–21 |
|---|---|---|
| 0 Calm Bull | +0.09 | −0.15 |
| 1 High Stress | −0.30 | −0.04 |
| 2 Moderate Bull | −0.19 | −0.09 |
| 3 Normal | **+0.22** | **+0.13** |

In regime 3 the model beats **every** baseline in both eras (held-out: model −1.01 vs
mix −1.20, shrink −1.28, pers −1.14, roll −1.36; full-sample p = 0.058 vs mix, other
p ≥ 0.22). Two caveats: it is "losing least" in a bad-Sharpe state, on ~15–20% of
days, and it is not significant after any multiple-testing consideration. Honest
status: an interpretable, consistent pattern worth reporting, not a headline win.
(Matrix losses: even within regime 3, shrinkage/mix beat the model on Stein/KL;
there is no regime where the model wins the statistical metrics.)

## 4. What this means for the paper

1. The defensible regime-conditional content is **diagnostic, not performance**: the
   framework's own regimes explain *when* similarity forecasting helps (Normal) and
   when it hurts (High Stress), consistently across eras. That is a genuine
   interpretability contribution and it feeds the "statistical-vs-economic
   disconnect" framing (ADVISOR_SUMMARY option 2).
2. The calm-vs-stress reversal must replace the old "superior in stress" text
   (regime_labelling.md §6.4 edit should NOT be applied as drafted — the corrected
   claim runs the other way for economic performance).
3. The look-ahead hybrid (§2) is a useful cautionary example of how easily a
   "wins in condition X" result can be manufactured; the future-window stats in
   backtest.csv must never be used as conditioning variables.
4. If a conditional win is to be pursued further, it needs **exogenous ex-ante
   state variables** (VIX level/term structure, credit spreads) rather than trailing
   return statistics — trailing corr simply doesn't predict future corr well enough
   (r = 0.41). That is new data + one more honest tune/held-out cycle.

*Analysis run 2026-07-27 on `results/oos_final` (662 anchors, 3,325 daily returns,
2008-10-13..2021-12-27). Scratch scripts consolidated into
`scripts/analysis/core/conditional_gmvp_analysis.py`.*
