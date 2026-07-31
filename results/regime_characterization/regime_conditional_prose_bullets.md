# Regime-conditional story — Friday prose bullets (Track B / Bree)

Revised 2026-07-30 after referee pass (B4): descriptive / exploratory language only.
Do **not** say the model “earns its edge” in Regime 3 (or anywhere). Paper uses
**Regimes 1–4** only (Regime 3 = legacy “Normal”). Source of truth:
`REGIME_CONDITIONAL_ANALYSIS.md` + Canonical Numbers.

## Bullets for Results prose

- **Descriptive calm-vs-stress pattern (not a trading rule).** Slicing the honest
  tranched daily GMVP series by concurrent crisis windows, the model’s best
  whole-sample Sharpe among methods appears on non-crisis days (**1.435**), with
  terminal wealth **6.66** vs persistence **5.22** on that slice. In every
  crisis/stress window the model is among the worst Sharpes and has the highest
  realized portfolio vol. This is a *characterization* of where cumulative P&amp;L
  accrued, and it reverses any prior “superior in stress” claim; it is **not**
  an ex-ante selectable edge.
- **Ex-ante conditioning fails.** Horizon `realized_vol` / `avg_corr` in the
  backtest are look-ahead descriptors; honest trailing-window hybrids and
  ensembles tuned on 2008–2016 and tested 2017–2021 collapse back to persistence
  (chosen hybrid OOS Sharpe **0.617** vs pers **0.635**).
- **Exploratory regime pocket (Regime 3), not a headline.** Model−persistence
  Sharpe gaps (tuning / held-out): Regime 1 **+0.09 / −0.15**, Regime 2
  **−0.19 / −0.09**, Regime 3 **+0.22 / +0.13**, Regime 4 **−0.30 / −0.04**.
  Regime 3 is the only regime with a positive gap in *both* eras (~15–20% of
  days), but the pocket is small and **insignificant after multiple testing**.
  Report it as an exploratory, regime-conditional diagnostic consistent with
  interpretability — not as where the method “wins” or “earns” performance.
- **Framing for the paper.** Regime labels support describing *when* similarity
  forecasting looks better or worse in-sample across eras; they do **not**
  upgrade the overall claim beyond held-out parity with persistence/EWMA/LW.

## Regime-label stability (bonus check on `results/oos_final/backtest.csv`)

Hard `dominant_regime` remapped to paper Regimes 1–4 via
`paper_regime_numbering.json`. Soft filtered posteriors in this export are
degenerate (max \(p_k\approx 0.25\) every day), so stability is about the hard
assignment only.

| Check | Result |
|---|---|
| Consecutive-anchor persistence (same paper regime) | **72.0%** of transitions |
| Self-transition by regime (signature table) | R1 0.772 · R2 0.664 · R3 0.625 · R4 0.771 |
| Share of days, 1st vs 2nd half of sample | R1 13.6→41.2 · R2 26.2→13.3 · R3 26.8→14.2 · R4 33.4→31.2 |
| Soft max-prob ≈ 1/K | **100%** of anchors (filtered \(\alpha\) uniform — hard labels still used) |

**Prose use:** regimes are moderately persistent at the hard-assignment level
(diagonal ~0.63–0.77) but mix shares shift across the sample (Regime 1 becomes
more prevalent later). Do not overclaim “stable latent states”; the signatures
are descriptive summaries of the hard assignments under the frozen
hyperparameters.

## Numbers used (Canonical)

| Paper regime | Legacy (internal) | Gap tune | Gap held-out |
|---|---|---|---|
| 1 | Calm Bull | +0.09 | −0.15 |
| 2 | Moderate Bull | −0.19 | −0.09 |
| 3 | Normal | +0.22 | +0.13 |
| 4 | High Stress | −0.30 | −0.04 |
