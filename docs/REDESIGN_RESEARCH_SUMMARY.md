# Redesign Research Summary (audited)

See also: [`REDESIGN_METRIC_AUDIT.md`](REDESIGN_METRIC_AUDIT.md), [`FINAL_SCIENTIFIC_FINDINGS.md`](FINAL_SCIENTIFIC_FINDINGS.md), [`PAPER_REWRITE_NOTES.md`](PAPER_REWRITE_NOTES.md).

## Frozen OOS (2017–2021, η=0.01) — audited units

Raw Frobenius below is **unsquared** \(\|\hat\Sigma-\Sigma\|_F\).

| Method | Raw Frob | Cond Stein | NLL | \(R^{\mathrm{raw}}\) | \(R^{\mathrm{cond}}\) | Leverage |
|---|---:|---:|---:|---:|---:|---:|
| EWMA | 17.98 | 1.08e6 | 5.10e5 | **0.0230** | **0.0231** | 2.71 |
| Persistence | 18.28 | 1.53e6 | 7.19e5 | 0.0246 | 0.0247 | 1.87 |
| Ledoit–Wolf | 14.31 | **5.06e4** | **2.35e4** | 0.0248 | 0.0249 | 1.69 |
| OAS | 17.62 | 6.57e4 | 3.06e4 | 0.0251 | 0.0252 | 1.91 |
| Rolling | 18.16 | 1.13e6 | 5.32e5 | 0.0260 | 0.0261 | 2.74 |
| Shrink | 16.09 | 9.93e4 | 4.65e4 | 0.0364 | 0.0364 | 1.77 |
| D0 PCA8 | **10.33** | 3.56e7 | 1.68e7 | 0.0601 | 0.0603 | 3.34 |

Constrained \(\|w\|_1\le 2\): D0 \(R^{\mathrm{cond}}\) → 0.030 (still last; EWMA ≈ 0.016).

## Paper

- Source: [`paper/redesign/main.tex`](../paper/redesign/main.tex)
- Title: *When Better Matrix Fit Does Not Mean Better Risk Forecasts: Lessons from Similarity-Based Covariance Modeling*
- Framing: decision-aware evaluation / failure-mode study (not a winning regime method)
