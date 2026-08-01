# Regime-Aware Similarity Covariance/Vol Forecasting — Findings for Advisor Review

**Prepared for:** Prof. Cucuringu · **Re:** ICAIF submission readiness · **Date:** 2026-07-10

## Bottom line

After correcting four evaluation-validity issues and adding the baselines the paper cites but never ran, **the model does not beat properly-constructed baselines out-of-sample on any target.** It is a statistical tie with all baselines on GMVP Sharpe, loses to Ledoit-Wolf on covariance likelihood losses (Stein/KL), and loses clearly to HAR on volatility. The "substantial/significant improvement over baselines" claim is not supported. The honestly-defensible contribution is a *modular regime-aware framework that matches standard estimators, with interpretable regime structure* — a methodological/exploratory result, not a performance win. **This memo is to decide how (or whether) to reframe for this cycle.**

All numbers below are **held-out 2017–2021**, with hyperparameters selected only on 2008–2016. Full reproduction code is committed (`ICAIF_REVIEW_FEEDBACK.md` has the details).

## What was corrected (all were inflating the original result)

1. **Equity-curve double-count.** The wealth curve compounded a 20-day holding return once every 5 days (~4×). Model terminal wealth 9.47 → **1.88** (≈36%/yr → ≈9%/yr). Ranking preserved but every wealth number was inflated.
2. **Silent sample loss / GFC excluded.** A single missing (pre-IPO) name NaN'd the whole day's portfolio return, dropping all of 2008–2013 (the GFC) from GMVP evaluation. Fixed by renormalizing weights over tradable names each day: valid GMVP dates **369 → 662** (62 in the GFC). Identity on full-universe days, so post-2014 numbers are unchanged.
3. **Non-standard Sharpe.** The reported Sharpe was a mean of per-window annualized Sharpes (noisy, ill-defined). Replaced with a standard whole-sample daily Sharpe on overlap-corrected (tranched) returns.
4. **In-sample tuning.** Hyperparameters were selected on 2015–2021, overlapping the test set. Re-tuned on 2008–2016 only; the grid re-selected essentially the paper's config (pca_k 40→48).

## The evidence (held-out 2017–2021)

**GMVP Sharpe (economic):** model 0.582 vs pers 0.635, EWMA 0.632, mix 0.597, roll 0.566, OAS 0.554, LW 0.549, shrink 0.546. Every model−baseline difference is a **statistical tie** (moving-block bootstrap, p = 0.70–0.78). The model also has the highest ex-post GMVP variance — it is the worst minimum-variance portfolio while being tied on Sharpe.

**Covariance accuracy (statistical):**

| | Stein ↓ | KL ↓ | Frobenius ↓ |
|---|---|---|---|
| Ledoit-Wolf | **839** | **420** | 0.0260 |
| OAS | 869 | 434 | 0.0265 |
| Shrink γ=.3 | 845 | 423 | 0.0246 |
| **Model** | 1095 | 548 | **0.0252** |
| Rolling / Persistence | ~1e6 | ~1e6 | 0.028–0.030 |

The model's strong-looking Stein/KL was only vs rolling/persistence and the fixed-γ strawman; **real Ledoit-Wolf beats it.** The model leads only on Frobenius, marginally.

**Volatility (statistical; corrected 2026-07-31 to common sample n=241 + pooled R², see `docs/VOL_METRIC_AUDIT.md`):** HAR MSE **0.177**, pooled R² **0.703** vs model MSE 0.324, pooled R² 0.372; pers 0.261/0.494, roll 0.320/0.380, shrink 0.340/0.341. HAR — the standard realized-vol baseline (Corsi 2009), cited but never run — roughly halves the model's error.

**Also relevant (from prior revision):** the Markov filter is empirically inert (raw vs filtered differ < 1e-8); regime persistence is 28–33% vs the 25% chance level.

## Reframing options

1. **Honest "matches, with interpretability" (recommended if submitting).** Position as: a modular regime-aware similarity framework that attains covariance/vol accuracy and GMVP performance *on par with* Ledoit-Wolf/OAS/EWMA/HAR, with the added value of interpretable, economically-labeled regimes. Report the ties honestly; lead with the framework and regime analysis, not a performance claim. Feasible for this cycle; modest contribution.
2. **Methodological "statistical-vs-economic disconnect."** Make the thesis that covariance accuracy does not determine GMVP outcome (persistence has catastrophic Stein/KL yet best GMVP). Novel angle; harder to sell as positive.
3. **Defer / rework.** Treat this cycle's data as a null result; pursue the genuinely open directions (exogenous macro/cross-asset regime conditioning; a portfolio objective matched to the model's strength) for a stronger next submission.

## What I recommend deciding

- Is a "matches standard estimators + interpretable regimes" contribution above the ICAIF bar, or do we hold for a stronger result?
- If submitting: which framing (1 vs 2), and are we comfortable foregrounding the ties and the negative vol R²?
- Note the double-blind constraint and that the framework builds directly on Cartea et al. (2023).

*Reproducibility: fixes + analyses committed to the repo (`build_equity_curves.py`, `oos_split_report.py`, `oos_significance.py`, `eval_extra_baselines.py`, `eval_vol_har.py`); runs on scai4. See `ICAIF_REVIEW_FEEDBACK.md` for the full issue-by-issue record.*
