# Why is GMVP variance worse for the model?

## 1. What we mean by “better” covariance

Yes: **if we forecast the covariance matrix well, we expect the GMVP to have low realized variance.**

- **GMVP** = global minimum variance portfolio. Weights are \( w \propto \Sigma^{-1} \mathbf{1} \) (from the forecast \(\Sigma\)).
- **Realized variance** in the backtest = variance of the portfolio returns over the **future** window: \( r_t = \sum_i w_i R_{t,i} \), then `gmvp_var = Var(r_1, ..., r_H)`. That equals \( w' \Sigma_{\text{true}} w \), where \(\Sigma_{\text{true}}\) is the **sample covariance of the future returns** (same for all methods).
- So **lower `gmvp_var`** = your chosen \(w\) is closer to the true minimum-variance portfolio for that window. Better covariance forecasts should yield \(w\) closer to that optimum and hence **lower** realized variance.

So the direction is correct: we want **low** GMVP variance, and it’s concerning that the model often has **higher** realized variance than shrink/roll.

---

## 2. How GMVP variance is computed in the backtest

- For each anchor date we have a **future** return matrix `fut` (shape horizon × assets).
- Each method produces a forecast covariance and then **GMVP weights** \(w\):
  - **model**: from regime model’s \(\hat{\Sigma}\)
  - **mix**: from mix of model + shrink
  - **roll / pers / shrink**: from rolling, persistence, or shrinkage covariance.
- **Realized stats** (including `gmvp_var`) come from `hold_period_portfolio_stats(fut, w)` in `similarity_forecast/backtests.py`:
  - Portfolio returns: `rp = fut @ w`
  - `gmvp_var = sample variance of rp` over that horizon (i.e. \(w' \Sigma_{\text{true}} w\) with \(\Sigma_{\text{true}} = \text{cov}(\text{fut})\)).

So we are comparing the **same** out-of-sample window; only the weights (and thus the forecast) differ. Higher `gmvp_var` for the model means its \(w\) is worse for that \(\Sigma_{\text{true}}\).

---

## 3. Why the model can have higher GMVP variance despite better Frobenius

### (a) Frobenius (and Stein/KL) are on \(\Sigma\), not on what GMVP cares about

- We evaluate forecasts with **Frobenius**, **Stein**, **KL** on the matrix \(\Sigma\).
- The GMVP depends on **\(\Sigma^{-1}\)** in the direction of \(\mathbf{1}\): \(w \propto \Sigma^{-1}\mathbf{1}\).
- A **small error in \(\Sigma\)** can become a **large error in \(\Sigma^{-1}\)** in that direction (e.g. if the error is in a low-eigenvalue direction). So “better Frobenius” does **not** guarantee “better \(w\)” or “lower \(w'\Sigma_{\text{true}} w\)”.

So it’s possible to have **better matrix loss** but **worse GMVP variance**.

### (b) Regime model can give more extreme or unstable weights

- The regime model’s \(\hat{\Sigma}\) can change more across time (regime shifts) than shrink or roll.
- That can lead to:
  - **More extreme** \(w\) (e.g. concentrated in a few assets),
  - **Higher turnover** (we see the model has higher turnover),
  - When the regime view is **wrong**, those extreme weights can produce **large realized variance** in some windows.

So the model may “try harder” to minimize variance in its forecast, but when that forecast is off, the **realized** variance can be worse.

### (c) Shrinkage stabilizes the inverse and often helps realized variance

- **Shrink** (and **mix**) pull the covariance toward a stable target (e.g. diagonal or constant correlation).
- That tends to:
  - **Stabilize \(\Sigma^{-1}\)** (less sensitivity to estimation error),
  - **Smooth weights** (less concentration, less turnover),
  - In practice often **lower realized variance** even when the raw forecast isn’t “closest” in Frobenius.

So shrink can win on GMVP variance **without** winning on Frobenius.

### (d) Guardrail and “model = shrink” on bad days

- When the model fails the **trace-ratio guardrail**, we replace \(\hat{\Sigma}\) with the shrink covariance for that day. So on those days, model and shrink have the **same** \(w\) and the **same** `gmvp_var`.
- The **difference** in average GMVP variance comes from days when the model **is** used. On those days, the model’s more aggressive \(\hat{\Sigma}\) can sometimes produce worse \(w\) and higher realized variance.

---

## 4. What would align GMVP variance with “better” covariance?

To make low GMVP variance a direct target, you’d need one or more of:

- **Loss in portfolio space**: e.g. train or select models to minimize **realized** \(w'\Sigma_{\text{true}} w\) (or a proxy), not only Frobenius/Stein/KL on \(\Sigma\).
- **Structured loss on \(\Sigma\)**: e.g. losses that emphasize the direction \(\Sigma^{-1}\mathbf{1}\) (e.g. certain spectral or inverse-based metrics), not only full-matrix norms.
- **More shrinkage / mixing**: the **mix** already often has lower variance than the **model**; increasing mix weight toward shrink typically reduces variance further at the cost of less “model” signal.

---

## 5. Quick diagnostics you can run

- **Time series of `gmvp_var`**: Plot `model_gmvp_var` vs `shrink_gmvp_var` (and optionally `mix_gmvp_var`) over time. See **when** the model is much worse (e.g. in volatile or regime-shift periods).
- **Correlation with regime**: Check whether model’s variance disadvantage is larger in high-regime-uncertainty or right after regime switches (e.g. `regime_prob_*` or regime assignment).
- **Weight concentration**: Compare \(\|w\|_2\) or max \(|w_i|\) across methods; if the model’s weights are more concentrated, that can explain both higher turnover and higher variance when wrong.

If you want, we can add a small script under `scripts/analysis/diagnostics/` that reads the backtest output and plots `gmvp_var` by method over time and (optionally) vs regime assignment.
