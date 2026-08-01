# ICAIF Submission — Current State of Pipeline & Results

*Internal working doc. Describes the pipeline and evaluation **as they stand now**, after the validity corrections, cross-checked against `results/oos_final/`, `results/regime_cov_renorm/`, `results/vol_oos/`, and the `similarity_forecast/` code. All portfolio and forecast numbers are the honest, out-of-sample figures unless noted otherwise.*

**Where things stand.** The pipeline is a modular regime-aware similarity forecaster evaluated on a clean out-of-sample split: hyperparameters are selected on 2008–2016 and every headline number is held-out 2017–2021. Against properly constructed baselines (Ledoit-Wolf, OAS, EWMA/RiskMetrics, HAR), the model does **not** beat baselines on any target. It is a statistical tie with all baselines on GMVP Sharpe; on covariance matrix losses it is on par with shrinkage (slightly worse than Ledoit-Wolf on Stein/KL, slightly better on Frobenius); on volatility it trails HAR with a negative held-out R². The defensible contribution is a **modular regime-aware framework that matches standard estimators, with interpretable regime structure** — a methodological/exploratory result, not a performance win. How (or whether) to frame this for the current cycle is an open author decision (see `ADVISOR_SUMMARY.md`).

---

## Part 1: Current results

### 1. Out-of-sample GMVP performance is a statistical tie with every baseline

Hyperparameters are selected on the tuning window (2008–2016) via the Phase-2 joint grid (32 configs, `configs/ablation_phase2_tune0816.yaml`) and held fixed for the 2017–2021 held-out evaluation. The selected config — fuzzy_cmeans, l1, **pca_k=48**, tau=2.0, k=10 — is the paper's config except pca_k (40→48), so re-tuning re-selects essentially the same model. Full 2008–2021 backtest under tag `oos_final`.

Honest daily tranched Sharpe (overlap-corrected, GFC included):

| Period | Model | Pers | Mix | Roll | Shrink |
|---|---|---|---|---|---|
| Tuning 2008–2016 (in-sample) | 0.694 | **0.750** | 0.733 | 0.687 | 0.685 |
| **Held-out 2017–2021 (OOS)** | 0.582 | **0.635** | 0.597 | 0.566 | 0.546 |
| Full 2008–2021 | 0.654 | **0.707** | 0.682 | 0.641 | 0.632 |

On the point estimate, persistence leads the model out-of-sample on daily Sharpe (0.635 vs 0.582) and terminal wealth (1.511 vs 1.501). But a moving-block bootstrap (block=20, 20k resamples, `scripts/analysis/core/oos_significance.py`) shows the held-out model-vs-baseline Sharpe gap is **not significant for any baseline**: model−pers Δ=−0.053 p=0.70 CI[−0.27,+0.20]; model−shrink Δ=+0.036 p=0.72; model−roll Δ=+0.016 p=0.86; model−mix Δ=−0.015 p=0.91. The accurate statement is **parity — no significant difference on GMVP Sharpe versus any baseline** (neither a loss nor a win; p≈0.7 also reflects low power on ~1255 noisy days).

### 2. The model has the highest ex-post GMVP variance

GMVP is a variance-minimization exercise, and the model produces the highest ex-post GMVP variance of the five core methods (9.88×10⁻⁵ vs 8.78 rolling, 8.88 mix, 8.93 persistence, 9.15 shrink; `report.csv`). Any Sharpe parity therefore comes from the return side — which a covariance forecast does not predict and GMVP does not optimize — so the model is the *worst* minimum-variance portfolio while being *tied* on Sharpe. This is the natural referee attack surface: the covariance forecast yields the worst min-variance portfolio and any return effect is unexplained (possible factor tilt). Not yet decomposed into factor exposures.

### 3. Equity curve and Sharpe estimator are now correctly constructed

The evaluation now emits per-day GMVP returns (`run_backtest.py` → `results/<tag>/gmvp_daily_returns.parquet`, covariance path), and `scripts/analysis/core/build_equity_curves.py` builds both a tranched (all-anchor, overlap-corrected) and a non-overlapping daily curve plus a whole-sample annualized Sharpe (`equity_curve_summary.csv`).

This replaces two earlier artifacts:
- **Wealth curves are no longer double-counted.** The prior curve compounded a 20-trading-day holding return once every 5 days (stride=5), compounding each period ~4×. Correctly stitched terminal wealth (all methods): Model **1.88**, Pers **1.82**, Mix **1.68**, Shrink **1.42**, Roll **1.41** — ranking preserved, model and persistence nearly tied. The old as-plotted 9.47 (≈36%/yr for a min-variance book over 7.3 yr) was the artifact; ~9.1%/yr is the honest figure.
- **Sharpe is now a standard whole-sample daily Sharpe** on the tranched series, replacing the earlier mean-of-per-window-annualized-Sharpes (noisy, never defined).

### 4. The GFC is now inside the portfolio evaluation

Portfolio returns are computed over observed names each day with weights renormalized over the tradable universe (`gmvp_daily_returns_renorm` in `backtests.py`, wired into `hold_period_portfolio_stats` and the daily artifact). Previously a single missing (pre-IPO) name NaN'd the whole day via `fut @ w`, dropping all of 2008–2013 — so the GFC the abstract advertises contributed zero portfolio evidence.

- Valid GMVP dates: **369 → 662** (263 pre-2014 dates recovered, **62 in the 2008–2009 GFC**).
- The renormalization is identity on fully-observed windows: on the 369 previously-valid dates, `model_fro` and `regime_assigned` are byte-identical and 356/369 GMVP values match to ~1e-15; only 13 partial-universe windows legitimately change. This introduces a disclosed time-varying universe.
- Consequence: including the GFC and removing the double-count compresses the field. Even under the paper's own per-window metric, the model's lead over persistence collapses from 1.57-vs-1.00 to 1.57-vs-1.53; on the honest daily tranched Sharpe the field sits at 0.63–0.71 (tag `regime_cov_renorm`), nothing like the paper's headline gap.

### 5. The novel components are inert or small and untested

- **The Markov filter is empirically inert** on this data: raw vs filtered regime posteriors differ by < 10⁻⁸. It is an architectural option, not a working contribution here.
- **The regime layer's measured delta** (K=1→K=4) is ~8% Sharpe (0.996→1.079), on the tuning sample, at a non-final config, with no significance test. The block-bootstrap test of K=4 vs K=1 at the final config on held-out data — the single experiment that directly tests the contribution — is not yet run.
- **Regime persistence is near-chance**: diagonal persistence is 28–33% vs a 25% chance level, so "persistent and interpretable market regimes" overstates what the data shows.

### 6. Against real baselines the model does not dominate on any axis

Real Ledoit-Wolf, OAS, and EWMA are now implemented (`baseline_ledoit_wolf`/`baseline_oas`/`baseline_ewma_cov` in `backtests.py`) and evaluated faithfully on the same OOS harness/anchors (`scripts/analysis/core/eval_extra_baselines.py`, tag `oos_final`, held-out 2017–2021). The prior "shrinkage" baseline was a fixed γ=0.3 blend toward the diagonal — a strawman relative to the Ledoit-Wolf estimator the paper cites.

| Method | GMVP Sharpe | Stein | KL | Frobenius |
|---|---|---|---|---|
| Persistence | 0.635 | 2.2e6 | 1.1e6 | 0.0295 |
| EWMA(0.94) | 0.632 | 1.1e6 | 5.5e5 | 0.0272 |
| Model | 0.582 | 1095 | 548 | **0.0252** |
| Rolling | 0.566 | 1.1e6 | 5.5e5 | 0.0280 |
| **Ledoit-Wolf** | 0.549 | **839** | **420** | 0.0260 |
| OAS | 0.554 | 869 | 434 | 0.0265 |
| Shrink γ=.3 | 0.546 | 845 | 423 | 0.0246 |

On Stein/KL, real Ledoit-Wolf and OAS **beat** the model (LW Stein 839 vs model 1095) — the model's apparent Stein/KL edge existed only against rolling/persistence and the γ=0.3 strawman. The model leads only on Frobenius, marginally. On GMVP Sharpe, EWMA(0.94)≈persistence, both above the model, and all pairwise differences are statistical ties (block bootstrap p=0.70–0.78). Against properly constructed estimators the model is on par — tie on GMVP, ~LW-family on matrix losses, slightly worse on Stein/KL, slightly better on Frobenius.

**Volatility target + HAR** (tag `vol_oos`; faithful pooled walk-forward HAR / Corsi 2009, `scripts/analysis/core/eval_vol_har.py`, refit every 20d, no look-ahead). *Corrected 2026-07-31 — common sample n=241, pooled R² per `docs/VOL_METRIC_AUDIT.md`; the earlier full-sample figures (HAR 0.176/−0.026, model 0.334/−0.241) and the per-anchor-mean R² column are retracted.* Held-out 2017–2021 vol MSE / pooled R²: **HAR 0.177 / 0.703** (best) vs pers 0.261/0.494, roll 0.320/0.380, **model 0.324 / 0.372**, shrink 0.340/0.341. HAR nearly halves the model's MSE; both metrics rank methods identically by construction. The old paper vol claim (MSE 0.233, R² 0.16) was in-sample only. Neither the covariance nor the volatility target gives the model a convincing OOS win over its own cited baselines.

---

## Part 2: Still open in the write-up

The empirics above are settled in code and results. These items are text/analysis work the paper still needs, and describe the present gap between the repo and the draft.

**Paper-text corrections (mostly no new code):**

7. **Survivorship/look-ahead disclosure.** Universe selection requires ≥85% availability in 2015–2021 and includes mega-caps known ex post (NVDA) for a backtest starting 2008. Cross-method comparisons on the shared universe stay internally fair, but absolute performance is inflated — needs a prominent disclosure in Section 4.1, not an implicit one.
8. **Table 6 statistics.** Persistence Sharpe p=0.052 is starred `*` under a p<0.05 threshold (doesn't qualify); Roll Frobenius p=0.019 is starred `**` under p<0.01. Also the sign convention flips meaning within one table (+ good for Sharpe, bad for Frobenius). Fix stars; restructure so positive always favors the model or split columns.
9. **Number provenance.** Sharpes of 0.996/1.079 (K-ablation), 1.943 (Phase 2), 1.573 (full backtest), 1.815 (normal periods) appear without a consistent sample/config label; `CURRENT_STATE.md` says 1.041, which matches none of them. Add a single "which sample, which config" table (and reconcile repo vs paper before the code link goes live). All of these are now superseded by the honest OOS numbers in Part 1.
10. **Stability-enhancement tuning.** γ=0.12 and λ=0.08 ("tuned to balance…", "selected to minimize realized GMVP variance") need an explicit statement of which sample they were tuned on; if the evaluation sample, that is leakage and must be folded into the OOS split. Same for the mixture weights, asserted "fixed a priori" two sentences before a Sharpe-variance sweep on the same data.
11. **Regime-persistence diagnostic (small code task).** Consecutive anchors share 47–49 of 50 window days, so their embeddings are nearly identical; near-chance hard-assignment flipping suggests cluster instability (FCM memberships near the simplex center) rather than fast regime switching. Check membership entropy and whether flips occur between near-tied memberships, and report FCM/GMM seed sensitivity (currently no cross-seed variability is reported anywhere).
12. **Leverage reporting.** GMVP here is long-short with unreported, time-varying gross exposure of Σ⁻¹1, which confounds Sharpe and turnover across methods. Report gross leverage per method; consider adding the long-only variant (already in the codebase) as a robustness table.

**Submission mechanics (do regardless of framing):**

13. **Format.** The draft is 19 pages, single-column article class, with "Supervisor: Mihai Cucuringu" on the title page. ICAIF uses the ACM sigconf two-column template with a hard page limit (recent full papers ~8 pages + references; confirm this year's CFP). Needs ~half the content cut — Section 4 (data QC) and the ablation narrative are the natural compression targets (move detail to an appendix/arXiv version).
14. **Anonymization.** ICAIF is double-blind. The title page carries names, emails, supervisor, and footnote 1 links a personal GitHub. Use an anonymized repo (e.g. anonymous.4open.science), strip the acknowledgment, and phrase the Cartea et al. (2023) relationship neutrally ("builds on") since the supervisor is a co-author and the paper could otherwise de-anonymize itself.
15. **Citation hygiene.** Cartea et al. (2023) is cited as "Working paper / preprint" with the title repeated — get the SSRN number or published version. Add related work a regime-focused reviewer expects: Pelletier (2006) regime-switching correlations, jump-model regime detection (Nystrup et al.), and recent ICAIF regime-detection papers.

---

## Part 3: Scope — directions not pursued this cycle

Proposals to add signals/complexity (macro, cross-asset, learned metrics, fancier regimes) are performance-engineering ideas. The current blocker is not insufficient performance — it is that the model already matches baselines on a now-clean evaluation; adding degrees of freedom on top would raise an in-sample number reviewers trust less. Current position on each:

| Idea | Fixes a reviewer objection? | Risk | Verdict |
|---|---|---|---|
| Macro (FRED-MD), lagged | Partially — answers "regimes are endogenous" | Scope blow-up | v2 / camera-ready |
| Cross-asset (VIX, bonds, credit) | Same; economic interpretation | **Look-ahead/circularity** if not strictly lagged | Strongest idea, still v2 |
| Learned / Mahalanobis metric | No | More params → amplifies overfitting attack | Skip this submission |
| Temporal / trajectory matching | No (ties to Path Shadowing) | Research-level; own validation | v2 |
| Adaptive kNN | Marginally (crisis) | Another threshold to tune | Cheap, low priority |
| Novelty detection / confidence fallback | Marginally | Already partly in codebase (blend-to-pers) | Low priority |
| Fancier regime models (VAE, time-varying A) | No — conflicts with own finding | Filter already inert | Skip / premature |
| Feature eng. (skew, kurt, vol-of-vol, dispersion) | No | Dimensionality (kNN curse) | Marginal |
| Learned embeddings (autoencoder, contrastive) | No | Largest overfitting surface; small N | Skip this submission |

Current position:
1. Any added feature raises the headline and lowers its credibility until there is a genuine OOS win to build on — which there currently is not.
2. The "high-impact" cross-asset features (VIX, realized skew-kurt, vol-of-vol) are contemporaneous risk measures, legitimate only if strictly lagged; VIX in particular risks dissolving the novelty into "VIX did the work" and would then require a VIX-conditioned baseline to rule out.
3. Regime upgrades contradict the project's own findings (filter inert, persistence near-chance); the right move on regimes is diagnostic (issue 11), not a bigger architecture.
4. Can't warn about the kNN curse of dimensionality and then concatenate VIX + bonds + credit + macro + skew + kurt.
5. "Regimes are endogenous (market-return-only)" is a legitimate limitation to state now; macro/cross-asset regime conditioning (`P(s_t=k | z_t, M_t)`) is the correct spine for a v2 with its own OOS evaluation, not a bolt-on.
