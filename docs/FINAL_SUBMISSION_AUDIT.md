# Final Submission Audit — ICAIF manuscript

**Canonical source:** `paper/main.tex`  
**Canonical PDF:** `paper/main.pdf`  
**Local ACM class:** `paper/acmart.cls` (**v2.19**, 2026-06-27)  
**Archived old manuscript:** `paper/archive/main_old.tex`, `paper/archive/main_old.pdf`  
**Mirror:** `paper/main_updated.tex` / `paper/main_updated.pdf` (kept in sync)

**Title:** When Better Matrix Fit Does Not Mean Better Risk Forecasts: Lessons from Similarity-Based Covariance Modeling

---

## Claims

- [x] Abstract claims appear in Results (collapse; SIM-Overlap/SIM-Pred no robust increment; SIM best median Frob / worst Stein–NLL–GMVP; common conditioning; inverse pathology).
- [x] Contributions have direct empirical evidence (collapse / incremental / oos / oosboot / dataqual / eigen / constrained / rankcorr).
- [x] No regime-performance claim exceeds bootstrap evidence (SIM-Pred CI covers 0; simplicity rule freezes SIM).
- [x] No Sharpe-based model-win language (only “Sharpe is not used for selection”).
- [x] Collapse attributed to representation geometry, not high dimension alone.
- [x] Mechanism not “small eigenvalues alone” (SIM worst top-band; LW worst bottom band yet best Stein).
- [x] Constrained leverage text matches Table: SIM worse than all except shrinkage (not “remains last / ranking unchanged”).
- [x] Inverse-Frobenius appears in eigenstructure table and prose.
- [x] No D0/D2/D5 or “legacy/earlier/final estimator” chronology language.

## Leakage / protocol

- [x] Neighbor eligibility: \(a\le t-H-g\) with \(H=20\), \(g=10\).
- [x] Representation / library / regimes fit on expanding past-only history.
- [x] SIM-Pred occupancy is predictive–predictive with daily transitions.
- [x] Candidate set enumerated; OOS architecture/HPs frozen after 2016.
- [x] Classical estimators are external benchmarks.
- [x] GMVP \(w(A)\), top/bottom 10% spectral bands, and equal-weight mixing projection defined.

## Metrics / evidence polish

- [x] Unsquared Frobenius; median primary / mean secondary.
- [x] Stein on \(C_\eta\); KL omitted (\(\mathrm{Stein}=2\,\mathrm{KL}\)).
- [x] \(R^{\mathrm{raw}}\) selection continuity explained; \(R^{\mathrm{cond}}\) emphasized for evaluation.
- [x] Lookback+future extreme-return filter table + mean Frobenius collapse for SIM.
- [x] Paired OOS block-bootstrap CIs for headline contrasts (full + cleaned panels).
- [x] Cartea positioned as inspiration / not replication; SSRN 4652980 + DOI.
- [x] Citations: Bezdek FCM, Künsch block bootstrap, Chen et al.\ OAS, RiskMetrics.

## Numeric spot-check (η=0.01 OOS)

| Claim | Result |
|---|---|
| Best median Frob | SIM = 0.017 |
| Best mean \(R^{\mathrm{cond}}\) | EWMA = 0.0231 |
| Best Stein | Ledoit–Wolf ≈ 5.06e4 |
| Worst \(R^{\mathrm{cond}}\) / Stein | SIM |
| SIM Inv.\ Frob.\ / cond.\ no.\ / lev | ≈2.25e15 / 9693 / 3.34 |
| Constrained \(R^{\mathrm{cond}}_{L\le2}\) | SIM 0.0302; Shrinkage 0.0332 (last) |
| SIM-Pred−SIM supported at any B∈{4,8,12,20}? | No |

## Visual / submission QA

- [x] Clean compile from scratch with local `acmart` v2.19 (CreatorTool confirms `2026/06/27 v2.19`).
- [x] Page count ≤8 including references (ICAIF limit); currently 6 pages.
- [x] Anonymous Author(s) / Anon.\ header; no author identity intended in metadata.
- [x] Figures have ACM `\Description{...}` text; Fig.~1 uses Unscaled (not Legacy); Fig.~2 labels SIM (not D0).
- [x] Inv.\ Frob.\ reported as squared; SIM-Pred uses HMM `transmat_`; bootstrap blocks calendar-preserving with filter masks.
- [x] Overleaf upload set: `main.tex`, `refs.bib`, `acmart.cls` (**required** for v2.19), `ACM-Reference-Format.bst`, `figs/*`, `tables/*` used by `main.tex`.
