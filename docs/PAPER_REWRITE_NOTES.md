# Paper Rewrite Notes

## Title (chosen)

**When Better Matrix Fit Does Not Mean Better Risk Forecasts: Lessons from Similarity-Based Covariance Modeling**

Alternatives considered: “Decision-Aware Evaluation…”, “Failure Modes in Regime-Aware…”

## Contribution bullets

1. Diagnose soft-regime collapse as a **representation-geometry** failure (heterogeneous scaling + PCA unsuitable for FCM), with train/query membership diagnostics.
2. Nested protocol separating similarity from regimes: regimes add **no robust incremental decision value** (block-bootstrap CIs).
3. Frozen OOS: similarity wins **raw Frobenius** but loses on conditioned Stein/KL/NLL and GMVP risk vs classical estimators; common \(C_\eta\) shows this is not an asymmetric-floor artifact.
4. Mechanistic evidence: inverse-Frobenius error, condition numbers, and leverage—not Sharpe—explain the matrix/decision disagreement.

## Sections vs old manuscript

| Old | New |
|---|---|
| Regime-aware method pitch | Evaluation problem + case study |
| Five-stage FCM+Markov pipeline as method | Similarity core + optional D2/D5 ablations; old pipeline as failure mode |
| Sharpe-led tables | Frobenius / Stein / NLL / \(R^{\mathrm{raw}}\) / \(R^{\mathrm{cond}}\) |
| Vague statistical–economic disconnect | Eigenstructure + inverse pathology |
| Volatility/HAR transfer | Removed |
| Wealth curves | Removed |

## Source

- Old preserved at [`paper/main.tex`](../paper/main.tex)
- New: [`paper/redesign/main.tex`](../paper/redesign/main.tex)

## 2026-08-08 update (Overleaf `paper/main.tex`)
- Audits: Frobenius mean vs median reconciled; PCA whitening=True; bootstrap in anchor units + sensitivity.
- Finding 5 narrowed (diagnostic, not universal decision surrogate).
- Expanded manuscript with Data section, kernel/PCA formulas, eigenstructure + constrained + rank-corr tables.
- Backup of previous Overleaf narrative: `paper/main.tex.bak_pre_redesign_rewrite`.
