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
