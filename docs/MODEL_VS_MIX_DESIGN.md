# What “model” vs “mix” includes (covariance vs volatility)

## Covariance

**“Model” (reported as `model_*` in backtest):**
- **No continuous ensemble** with baselines. The model series is:
  - The **pipeline output** (regime-aware kNN forecast), optionally with **in-pipeline** `output_shrink_toward_diag` (blend with the forecast’s own diagonal for stability — regularization, not ensemble with roll/shrink).
  - **Guardrail:** when the pipeline’s trace ratio is out of bounds, we **replace** that day’s forecast with **shrink** (so that day “model” = 100% shrink). So it’s binary: pipeline or shrink, not a smooth blend.
- So covariance **“model” = pipeline only** (plus optional diagonal shrinkage and guardrail replacement). It is **not** blended with roll or shrink.

**“Mix”:**
- Explicit ensemble: convex combination of model + shrink + pers via `cov_mix_weights` (e.g. shrink 0.55, pers 0.2, model 0.25).

---

## Volatility

**“Model” (reported as `model_*` in backtest):**
- **Does include an explicit blend** with baselines in the backtest:
  - `vol_hat_use = (1 − α − β)·vol_hat + α·vol_roll + β·vol_shrink`
  - With current config: `vol_dampen_toward_roll: 0.5`, `vol_dampen_toward_shrink: 0.15` → **“model” = 35% kNN + 50% roll + 15% shrink.**
- So volatility **“model”** is already an ensemble (pipeline + roll “momentum” + shrink). That’s why vol model can beat baselines: we blend the kNN forecast with roll and shrink before evaluating.

**“Mix”:**
- Further blend: `vol_mix = (1 − mix_lambda)·vol_shrink + mix_lambda·vol_hat_use`, so mix is shrink vs (the already-blended) model.

---

## Summary

| Task        | “Model” includes ensemble with baselines? | Notes                                                                 |
|------------|-------------------------------------------|-----------------------------------------------------------------------|
| Covariance | No                                        | Pipeline only (+ optional diagonal shrinkage; guardrail → replace with shrink). |
| Volatility | Yes                                       | vol_hat_use = kNN + roll + shrink (config: 35% + 50% + 15%).         |

So for **cov** the only explicit multi-method ensemble is **“mix”**. For **vol**, **“model”** already embeds a roll/shrink blend (momentum + stability); “mix” adds another layer on top of that.

---

## Volatility: dampening vs correlation vs performance

**The tension**
- **With dampening** (current): vol “model” beats baselines on MSE/MAE/RMSE, but forecast correlation (model vs roll/shrink/pers) is **high** → little diversification; ensembling doesn’t add much beyond what the blend already does.
- **Without dampening** (raw kNN only): vol model **underperforms** all baselines on every metric. So “isolating” the vol model would fix correlation at the cost of worse forecasts.

**Recommendation: keep dampening; don’t drop it for diversification.**

1. **Dampening is for performance, not diversification.** We blend with roll/shrink because vol is persistent and kNN alone is noisy. The goal is **better MSE**, not uncorrelated errors. High correlation after dampening is expected: we made the forecast look more like roll/shrink on purpose.
2. **Don’t remove dampening in production** just to get lower correlation. That would give you a “purer” model but worse realized accuracy. The right framing is: *we blend for stability/performance; we do not rely on ensemble diversification for vol.*
3. **No need to add more ensemble.** High correlation means further combining model with baselines (e.g. fancier mix weights) won’t diversify error risk. The current “model” (with dampening) and “mix” are enough; no need to push ensembling further.
4. **Optional diagnostic (once).** If you want to *see* whether raw kNN is less correlated with baselines: set `vol_dampen_toward_roll: 0` and `vol_dampen_toward_shrink: 0`, re-run the vol backtest, then run the forecast-correlation script. You’ll likely see lower correlation (model vs roll/shrink) but worse metrics. That documents the tradeoff and justifies keeping dampening for the main results.

**Bottom line:** Keep the current vol setup. Treat dampening as **stability/performance blending**, not as diversification. Report high correlation as “forecasts move together; ensembling does not materially diversify error risk,” and avoid over-selling ensemble benefits. Optional: one run with dampening off to document the correlation/performance tradeoff in the appendix or supplement.
