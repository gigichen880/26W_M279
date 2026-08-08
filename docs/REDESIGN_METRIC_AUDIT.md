# Redesign Metric Audit

Artifacts: [`results/redesign/metric_audit/`](../results/redesign/metric_audit/) (`audit.json` passed hard assertions).

## A1. Common conditioning \(C_\eta\)

Implementation: [`similarity_forecast/redesign/conditioner.py`](../similarity_forecast/redesign/conditioner.py)

\[
\epsilon(\Sigma)=\eta\frac{\mathrm{tr}(\Sigma)}{N},\qquad
C_\eta(\Sigma)=Q\,\mathrm{diag}(\max(\lambda_i,\epsilon(\Sigma)))\,Q^\top.
\]

**Verified**

- Forecast threshold \(\epsilon_t^f\) uses \(\mathrm{tr}(\hat\Sigma_t)\) only.
- Realized threshold \(\epsilon_t^r\) uses \(\mathrm{tr}(\Sigma_t^{\mathrm{real}})\) only.
- Changing the target does not change \(C_\eta(\hat\Sigma)\).
- Output is symmetric and SPD; \(\lambda_{\min}(C_\eta(\Sigma))\ge\epsilon(\Sigma)\).

**Not a bug.** Official OOS uses \(\eta=0.01\).

## A2. Stein / KL

From [`similarity_forecast/backtests.py`](../similarity_forecast/backtests.py):

- Stein: \(\mathrm{tr}(\hat S^{-1}S)-\log\det(\hat S^{-1}S)-n\) with \(S=\)true, \(\hat S=\)forecast.
- KL: \(\mathrm{KL}(N(0,S_{\mathrm{true}})\|N(0,S_{\mathrm{hat}}))\).

**Verified:** \(L_{\mathrm{Stein}}(S,S)=0\), \(\mathrm{KL}(S\|S)\approx 0\).

Redesign evaluates both on \((C_\eta(\hat\Sigma),C_\eta(\Sigma^{\mathrm{real}}))\) with independent floors.

## A3. Frobenius units (why old ~0.03 vs redesign ~10²–10³)

| Source | Definition |
|---|---|
| Old manuscript / `frobenius_error` | \(\|A-B\|_F=\sqrt{\sum_{ij}(A_{ij}-B_{ij})^2}\) |
| Redesign panel `raw_frob` (pre-audit) | \(\|A-B\|_F^2=\sum_{ij}(A_{ij}-B_{ij})^2\) |

Returns are **decimal** daily returns (median \(|r|\approx 0.01\)). Typical 20-day sample-cov diagonal \(\sim 2\times 10^{-4}\); typical \(\|S\|_F\sim 10^{-2}\).

**Conclusion:** scale mismatch is primarily **squared vs unsquared Frobenius**, not a decimal/percent return bug. Rankings under \(x\mapsto x^2\) are identical for nonnegative values; **absolute levels are not comparable** to the old paper. Going forward, report **unsquared** \(\|A-B\|_F\) as `raw_frob` in tables (and keep squared only if labeled).

## B. Variance “regret” naming

Frozen selection metric (unchanged):

\[
w=w(C_\eta(\hat\Sigma)),\quad w^\star=w(C_\eta(\Sigma^{\mathrm{real}})),
\]
\[
R^{\mathrm{raw}}=w^\top\Sigma^{\mathrm{real}}w-w^{\star\top}\Sigma^{\mathrm{real}}w^\star.
\]

Because \(w^\star\) minimizes variance under \(C_\eta(\Sigma^{\mathrm{real}})\), not raw \(\Sigma^{\mathrm{real}}\), \(R^{\mathrm{raw}}\) is **not guaranteed nonnegative**. Prefer the name

> **raw realized excess variance versus the conditioned oracle**

Robustness-only companion (not used for selection):

\[
R^{\mathrm{cond}}=w^\top C_\eta(\Sigma^{\mathrm{real}})w-w^{\star\top}C_\eta(\Sigma^{\mathrm{real}})w^\star\ge 0.
\]

## A4. Frobenius mean vs median (critical reconciliation)

End-to-end trace (2017-01-09): \(|\Sigma^{\mathrm{real}}\|_F\approx0.010\), \(\|\hat\Sigma-\Sigma\|_F\approx0.0099\) — matches audit \(\sim10^{-2}\).

OOS panel unsquared Frobenius for D0: **median 0.017**, **mean 10.3**. About 16% of dates have Frobenius \(>1\) (some \(\sim140\)), associated with residual extreme returns (\(|r|\) up to \(\sim10\)) in the parquet despite the paper’s intended \(|r|>0.5\) cleaning rule. **No \(10^3\) display multiplier.** Means are outlier-dominated; medians are the honest central tendency.

See `results/redesign/metric_audit/FROBENIUS_UNITS_RECONCILIATION.md`.

## A5. PCA whitening

Frozen D0 uses `PCAOnlyEmbedder` with **`whiten=True`** (sklearn). Verified in code and runtime. (Old pipeline used `whiten=False`.)

## A7. Pre-log SPD vs \(C_\eta\)

Three distinct stabilizations (see `LOG_EUCLIDEAN_SPD.md`):

1. **Target construction:** ridge \(10^{-8}I\) + `project_to_spd(eps=1e-8)` → \(\Sigma^{\mathrm{real}}\) used for libraries and raw Frobenius.
2. **Log-Euclidean:** `project_to_spd` / `logm_spd` with \(\epsilon_{\log}=10^{-12}\) (intrinsic to matrix log).
3. **Evaluation conditioner \(C_\eta\):** relative floor \(\eta\cdot\mathrm{tr}/N\) for Stein/NLL/GMVP only — **not** inside aggregation.

D0 does **not** receive a method-specific relative floor beyond \(\epsilon_{\log}\).

## A6. Bootstrap block units

Incremental bootstrap resamples **evaluation anchors** (stride-5 dates), not daily returns. `block=20` ⇒ **20 anchors ≈ 100 trading days**. Sensitivity over \(B\in\{4,8,12,20\}\) anchors: D5−D0 CI still covers 0 in all cases (`bootstrap_block_sensitivity.csv`).

## Bugs found?

| Item | Verdict |
|---|---|
| Target leakage into forecast floor | None |
| Stein/KL identity | Correct |
| Frobenius mean ≫ \(10^{-2}\) | Heavy-tailed dates / residual extremes — use **median**; not a secret multiplier |
| Squared vs unsquared (earlier panel) | Fixed in robustness panel; both reported |
| \(R^{\mathrm{raw}}\) called “regret” | Naming overclaim; metric preserved |
| L1 constraint undoing | Fixed for constrained robustness |
| Bootstrap “block=20 days” wording | Misleading — clarify **anchor** units |

No bug requires invalidating the frozen OOS architecture choice.
