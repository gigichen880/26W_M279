# Final Scientific Findings

Frozen OOS and redesign experiments are unchanged. This document is the scientific checklist aligned with the submission manuscript `paper/main.tex`.

## Positioning

Similarity-based covariance forecasting with optional regime-aware conditioning (D0 → D2 → D5), evaluated under decision-aware criteria. **Not** a “new model wins” claim.

## Finding 1 — Why did the old regimes collapse?

**Representation-geometry failure**, not high dimension alone.

- Legacy PCA+SVD \(D=240\) unscaled → FCM \(\max p\equiv 0.25\), centroid contrast ≈ 0.
- Joint standardization of the same 240 features → \(\max p\approx 0.53\).
- PCA-only at \(D\in\{5,10,15,48\}\) still near-uniform.
- Market-state \(D=6\) (log mean vol, log vol-of-vol, mean pairwise corr, top corr eigenvalue, EW lookback return, lag-1 EW autocorr) → \(\max p\approx 0.64\)–\(0.67\).

## Finding 2 — Did regimes add incremental value?

**No robust incremental decision value** (2013–2016). D5−D0 point estimate slightly favors D5 but bootstrap CIs cover 0 for \(B\in\{4,8,12,20\}\) anchors. Frozen winner by pre-specified simplicity: **similarity-only D0**.

## Finding 3 — Frozen OOS vs classical estimators

D0 has **lowest median Frobenius** (0.017) but **worst** conditioned Stein, NLL, mean \(R^{\mathrm{raw}}\), and mean \(R^{\mathrm{cond}}\). EWMA/persistence lead decision risk; Ledoit–Wolf/OAS lead Stein/NLL. \(R\) means ~0.02–0.06 are heavy-tail dominated (medians ~\(10^{-5}\)); **not annualized**.

## Finding 4 — Common conditioning

Does **not** remove the discrepancy across \(\eta\in\{0.001,0.01,0.05\}\).

## Finding 5 — Inverse-sensitive losses vs decisions

Stein/NLL flag D0’s pathology that Frobenius misses, but do **not** reproduce classical GMVP rankings. Descriptive Spearman (\(n=7\)): Stein–NLL ≈ 1; either vs \(R^{\mathrm{cond}}\) ≈ 0.25; median Frob vs \(R^{\mathrm{cond}}\) ≈ 0.

## Finding 6 — Mechanism

Not “small eigenvalues only.” D0 worst top-band log-MSE, inverse-Frobenius ~\(10^{15}\) vs ~\(10^{12}\), cond. number of \(C_\eta(\hat\Sigma)\) ~9700 vs LW ~127, highest leverage / \(\max|w|\).

## Finding 7 — Constrained GMVP

\(\|w\|_1\le 2\): D0 improves but remains last.

## Finding 8 — Post-hoc extreme-return filter

Exclude future windows with any \(|r|>0.5\) (194/247): D0 mean Frob → 0.024; still best median Frob; still last on Stein and \(R^{\mathrm{cond}}\).
