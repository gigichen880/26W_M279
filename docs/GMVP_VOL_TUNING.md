# Improving model cumulative advantage on GMVP vol

**Goal:** Raise the cumulative advantage of the **model** vs roll/shrink/mix on `gmvp_vol` (lower model GMVP volatility = positive advantage).

**Pipeline:** Model GMVP vol is the realized daily volatility of the portfolio built from the **model’s** forecast covariance (`Sigma_hat`), after guardrail and floor. Roll/mix/shrink use their own covariances. So model vol is driven by:

1. **Stability of the forecast** (guardrail, floor, kNN, regimes)
2. **Concentration of GMVP weights** (more ridge → more diversified → often lower vol)

---

## Config levers (regime_covariance.yaml)

| Parameter | Current | Effect on model gmvp_vol | Suggested for lower vol |
|-----------|--------|---------------------------|--------------------------|
| **guardrail.trace_ratio_lo / hi** | 0.4, 2.5 | When trace(Sigma_hat)/trace(S_roll) ∉ [lo, hi], model uses S_shrink. Tighter band → replace with shrink more often → more stable. | e.g. **0.6, 2.0** or **0.7, 1.8** |
| **stability.floor_eps** | 5e-4 | Diagonal ridge on matrix used for GMVP weights. Larger → less extreme weights → usually **lower** realized vol. | e.g. **1e-3** or **2e-3** |
| **backtest.k_neighbors** | 10 | More neighbors → smoother Sigma_hat → less jumpy weights. | e.g. **14** or **18** |
| **backtest.long_only** | false | Long-only GMVP → less extreme positions, often lower turnover and vol. | **true** to try |
| **model.n_regimes** | 4 | Fewer regimes → smoother regime mix; more → can be jumpier. | e.g. **3** to test |
| **model.transition_estimator** | soft | Soft uses full posteriors; hard uses argmax. Soft is usually smoother. | keep **soft** |

**Note:** `mixing.mix_lambda` only affects the **mix** path, not the **model** path. To improve *model* gmvp_vol you need to stabilize the model’s covariance (guardrail, floor, k, regimes) or the GMVP construction (floor_eps, long_only).

---

## Quick changes to try

1. **Tighten guardrail** so the model falls back to shrink more often when the raw forecast is extreme:
   - `trace_ratio_lo: 0.6`, `trace_ratio_hi: 2.0`
2. **Increase floor_eps** so GMVP weights are less concentrated:
   - `floor_eps: 1.0e-3` or `2.0e-3`
3. **Increase k_neighbors** for a smoother forecast:
   - `k_neighbors: 14` or `18`
4. Optionally set **long_only: true** and/or **n_regimes: 3** and re-run the backtest, then regenerate `cumulative_advantage_model_gmvp_vol.png` with `run_all`.

After changing config, re-run:
```bash
python run_backtest.py --config configs/regime_covariance.yaml
python -m scripts.analysis.run_all --target covariance
```
