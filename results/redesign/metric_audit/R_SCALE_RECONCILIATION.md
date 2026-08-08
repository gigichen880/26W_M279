# R_raw / R_cond scale reconciliation

## Is there annualization (*252)?
**No.** `conditioned_gmvp_variance_regret` stores
`R_raw = w' Σ_real w − w*' Σ_real w*` with **no** scaling by 252.
Grep of redesign metrics / runner / robustness scripts finds no annualization of these portfolio variances.

## Why are table means ~0.023–0.060?
Same heavy-tail pattern as Frobenius. Per-anchor values on ordinary dates are ~10^{-5}.

| Method | mean R_raw | median R_raw |
|---|---:|---:|
| EWMA | 0.023 | 7.5e-5 |
| D0 | 0.060 | 9.5e-5 |

Single-date recompute (2021-10-07, EWMA): stored R_raw equals `w'Σw − w*'Σw*` exactly (~9.6e-7); `Δ×252` does **not** match the stored value.

## Paper wording
Report as **mean per-anchor excess variance of daily portfolio returns** under the constructed H-day sample covariance (**not annualized**). Means are outlier-dominated; medians are O(10^{-5}). After excluding future |r|>0.5 windows, mean R_cond falls to ~10^{-4}.
