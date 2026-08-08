# Pre-expansion audit package (Frobenius / PCA / bootstrap)

## 1. Frobenius units — reconciled

| Check | Result |
|---|---|
| Single-date \(\|\Sigma\|_F\), \(\|\hat\Sigma-\Sigma\|_F\) | \(\sim10^{-2}\) (decimal returns) |
| Table means \(\sim10\)–\(18\) | **Heavy-tailed outlier dates**, not a \(10^3\) multiplier |
| D0 median / mean | **0.017 / 10.33** |
| Squared vs unsquared | Robustness panel stores both; paper reports unsquared |

Artifacts: `results/redesign/metric_audit/FROBENIUS_UNITS_RECONCILIATION.md`, `frob_robust_summary.csv`.

## 2. PCA whitening — verified

Frozen D0 uses `PCAOnlyEmbedder(whiten=True)` (sklearn). Kernel: \(w_i\propto\exp(-d_i/d_{(k)})\).

## 3. Bootstrap units — clarified + sensitivity

Bootstrap resamples **evaluation anchors** (stride 5). `block=20` ⇒ ≈100 trading days. Sensitivity \(B\in\{4,8,12,20\}\): D5−D0 never supported.

Artifact: `bootstrap_block_sensitivity.csv`, `BOOTSTRAP_UNITS.md`.

## 4. Finding 5 — narrowed

Inverse-sensitive losses **flag D0**; they do **not** fully reproduce classical decision rankings. Rank correlations in `metric_rank_correlations.csv`.

## 5. Manuscript

Expanded Overleaf source: `paper/main.tex` (backup: `paper/main.tex.bak_pre_redesign_rewrite`). Compiles with `tables/` + `figs/` + `refs.bib`.
