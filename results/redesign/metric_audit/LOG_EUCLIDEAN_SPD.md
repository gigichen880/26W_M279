# Pre-log SPD projection vs common conditioner

## Code path (frozen redesign)

1. **Target construction** (`build_covariance_target` → `cov_from_returns_imputed`):
   - impute NaNs (asset-wise window mean)
   - sample covariance
   - add ridge `1e-8 * I`
   - `project_to_spd(..., eps=1e-8)` absolute eigenvalue clip

2. **Log-Euclidean aggregation** (`SimilarityForecaster.eps = 1e-12`):
   ```
   Acc += w_i * logm_spd(project_to_spd(T_i, eps=1e-12), eps=1e-12)
   Σ̂ = project_to_spd(expm(Acc), eps=1e-12)
   ```
   Intrinsic numerical requirement of matrix log/exp; **distinct from** evaluation conditioner `C_η`.

3. **Common evaluation conditioner** `C_η` (`eta=0.01` official):
   - relative floor `ε = η * tr(Σ)/N` for Stein/NLL/GMVP only
   - **not** used inside log-Euclidean aggregation

## Single-date check (2017-01-09)
- Math sample cov (imputed, no ridge): min eig=-3.639e-18, n_pos@1e-12=19
- Built target (ridge+ε=1e-8): min eig=1.000e-08, ||built−raw||_F=1.000e-07
- After ε_log=1e-12: ||·−built||_F=1.703e-17
- After C_η: min eig=3.004e-06, relative ε=3.004e-06

## Frobenius evaluation
`raw_frob` compares `Σ̂` to `S_real = build_covariance_target(fut)` (ridge+ε=1e-8 constructed target), **not** pure rank-deficient sample cov and **not** `C_η(Σ)`.
