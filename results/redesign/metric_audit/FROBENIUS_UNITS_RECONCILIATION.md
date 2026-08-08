# Frobenius units reconciliation

## End-to-end single-date trace (2017-01-09)
- decimal returns; |Σ_real|_F ≈ 0.010; |Σ̂-Σ|_F ≈ 0.0099 (matches audit ~1e-2 scale)

## Why Table means are ~10–18
- **Not a display multiplier bug and not squared-vs-unsquared confusion in the robustness panel.**
- Per-date unsquared Frobenius is heavy-tailed: D0 median ≈ **0.017**, mean ≈ **10.3**.
- ≈16% of OOS dates have Frobenius > 1 (often ≫ 100), dominating the mean.
- Ranking by **mean** Frobenius: D0 best; ranking by **median** Frobenius: see frob_robust_summary.csv.

## Recommendation for paper
Report **median** (and optionally trimmed/mean-on-cleaned dates) alongside mean, or state that mean Frobenius is outlier-dominated.
Do **not** claim a 10^3 display factor unless explicitly introduced.
