# Regime-conditional story — Friday prose bullets (Track B / Bree)

One-paragraph bullet summary for co-writing Results. Claims stay within Canonical
Numbers / `REGIME_CONDITIONAL_ANALYSIS.md`. Paper uses **Regimes 1–4** only
(Regime 3 = legacy "Normal").

## Bullets for Results prose

- **Descriptive calm-market edge (not tradeable).** On non-crisis days the model
  posts the best Sharpe among methods (**1.435**) and compounds to terminal
  wealth **6.66** vs persistence **5.22**; that is where the cumulative advantage
  is earned. In every crisis/stress window the model is among the worst Sharpes
  and has the highest realized portfolio vol — reversing any prior “superior in
  stress” claim.
- **Ex-ante conditioning fails.** Horizon `realized_vol` / `avg_corr` in the
  backtest are look-ahead descriptors; honest trailing-window hybrids and
  ensembles tuned on 2008–2016 and tested 2017–2021 collapse back to persistence
  (chosen hybrid OOS Sharpe **0.617** vs pers **0.635**).
- **Only era-consistent pocket: Regime 3.** Model−persistence Sharpe gaps
  (tuning / held-out): Regime 1 **+0.09 / −0.15**, Regime 2 **−0.19 / −0.09**,
  Regime 3 **+0.22 / +0.13**, Regime 4 **−0.30 / −0.04**. Regime 3 is the sole
  positive pocket in both eras (~15–20% of days) but is insignificant after
  multiple testing — report as interpretable diagnostics, not a headline win.
- **Framing for the paper.** Regime-conditional content supports interpretability
  (when similarity helps vs hurts) and feeds the statistical-vs-economic
  disconnect; it does **not** upgrade the overall claim beyond parity.

## Numbers used (Canonical)

| Paper regime | Legacy (internal) | Gap tune | Gap held-out |
|---|---|---|---|
| 1 | Calm Bull | +0.09 | −0.15 |
| 2 | Moderate Bull | −0.19 | −0.09 |
| 3 | Normal | +0.22 | +0.13 |
| 4 | High Stress | −0.30 | −0.04 |
