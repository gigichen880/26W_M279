# Why the volatility model can underperform baselines (MSE) and how to improve

## Why MSE can be worse than roll / pers / shrink

1. **Volatility is highly persistent**  
   The best predictor of future realized vol is often **past realized vol**. So:
   - **Persistence (pers)** uses past-horizon realized vol as the forecast → very strong baseline.
   - **Rolling (roll)** uses the lookback window’s realized vol → also strong.
   - **Shrink** stabilizes that with a constant or structure → often best or close.
   So we’re comparing against baselines that are already well suited to the target.

2. **kNN adds variance**  
   The model forecasts by averaging **log-vol of k similar past windows**. Even if that mean were unbiased:
   - With small **k** (e.g. 7), the average has **high variance** (only 7 numbers).
   - Roll/shrink use the **whole** lookback and shrinkage, so they have lower variance.
   So the model can lose on MSE purely from **higher forecast variance**, not only bias.

3. **Similarity ≠ same future vol**  
   We use **VolStatsEmbedder** (past vol distribution, trend, concentration). “Similar” in that space doesn’t guarantee “similar future log-vol”. So the kNN average can be noisier or biased relative to the conditional mean that MSE targets.

4. **No direct MSE training**  
   We’re not minimizing MSE; we’re doing similarity + average. So there’s no guarantee the procedure is optimal for MSE.

---

## Levers that help (implemented or config)

### 1. **Dampen model toward roll and shrink** (recommended)

- **`vol_dampen_toward_roll`**  
  `vol_hat_use = (1 - damp_roll - damp_shrink)*vol_hat + damp_roll*vol_roll + damp_shrink*vol_shrink`  
  Higher values (e.g. **0.5**) make the “model” forecast closer to roll → lower variance and often **better MSE**.

- **`vol_dampen_toward_shrink`**  
  Same formula; e.g. **0.15** adds 15% shrink. Helps when shrink is the best baseline.

- **Current defaults in `configs/regime_volatility.yaml`**  
  - `vol_dampen_toward_roll: 0.5`  
  - `vol_dampen_toward_shrink: 0.15`  
  So “model” = 35% kNN + 50% roll + 15% shrink. Tune these if you want more or less baseline.

### 2. **More neighbors**

- **`backtest.k_neighbors`**  
  Increased from 7 to **15** so the kNN average is over more past windows → **lower forecast variance** and often better MSE. Try 15–25.

### 3. **Softer similarity (larger tau)**

- **`model.tau`**  
  Larger **tau** → softer similarity (weights more even across neighbors) → smoother, lower-variance forecast. Try e.g. **3.0** if you have 2.0 now.

### 4. **Mix series**

- **`mixing.mix_lambda`**  
  “Mix” = (1 - mix_lambda)*shrink + mix_lambda*vol_hat_use.  
  Keeping **mix_lambda** low (e.g. **0.2**) makes the **mix** series close to shrink, which often has good MSE.

---

## What to expect after tuning

- With **strong dampening** (e.g. 50% roll + 15% shrink), the **“model”** series can get close to or beat roll/shrink on **Vol MSE**, because it’s mostly a blend of strong baselines plus a small kNN component.
- If you **lower dampening** again to give kNN more weight, MSE may worsen but you might capture more regime-specific variation; that’s a bias–variance tradeoff.
- **Persistence** can remain very hard to beat on pure MSE; beating it consistently may require a different design (e.g. loss or target tailored to vol).

---

## Quick config checklist for better vol MSE

| Parameter                      | Suggested range / value | Effect                          |
|-------------------------------|--------------------------|---------------------------------|
| `vol_dampen_toward_roll`      | 0.4 – 0.6                | Stronger pull toward roll       |
| `vol_dampen_toward_shrink`    | 0.1 – 0.2                | Slight pull toward shrink       |
| `backtest.k_neighbors`        | 15 – 25                  | Smoother kNN, lower variance    |
| `model.tau`                   | 2.0 – 3.5                | Softer similarity               |
| `mixing.mix_lambda`           | 0.15 – 0.25              | Mix stays close to shrink       |

Re-run the vol backtest and the statistical comparison to see the impact on Vol MSE / MAE / RMSE.
