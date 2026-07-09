# ICAIF Reviewer-Style Critique & Fix List

*Internal working doc — not for submission. Review of the current paper draft cross-checked against `results/regime_covariance/` outputs and `similarity_forecast/` code.*

**Verdict:** In its current form this would likely get "reject" or at best "major revision" — not because the idea is weak, but because the headline evidence is mostly in-sample, the portfolio metric works against you, and the paper isn't in submission format. All of these are fixable.

---

## Part 1: Fatal-if-unfixed issues

### 1. The portfolio evaluation is almost entirely inside the tuning window (worse than the paper discloses)

The paper honestly flags that hyperparameters were tuned on 2015–2021, which overlaps the evaluation window, and says "the genuinely out-of-sample portion is the pre-2015 period." But I verified against `results/regime_covariance/backtest.csv`: valid GMVP dates run 2014-03-10 to 2021-06-29, and only 42 of the 369 GMVP dates fall before 2015. So ~89% of the portfolio evidence — including the headline Sharpe 1.57 — sits inside the period the configuration was selected on. The "out-of-sample" pre-2015 portion barely exists for portfolio metrics. A careful reviewer will reconstruct this from Figure 4 (flat until 2014) and the date counts in Section 6, and it undermines the central empirical claim.

**Fix (pick one, in descending order of strength):**
- Re-tune on pre-2015 data only (or 2015–2018), and report 2015–2021 (or 2019–2021) as genuinely held-out. This is the clean fix and your ablation infrastructure makes it cheap.
- Alternatively, do anchored walk-forward tuning (re-select config each year using only past data).
- At absolute minimum, report the tuned-period and held-out-period results in separate columns and move this disclosure from a mid-paragraph aside into the results table itself. But note this weak version probably still draws the criticism.

**Fix status (DONE — decisive, 2026-07-08/09).** Split: tune 2008–2016, hold out 2017–2021. Re-ran the Phase-2 joint grid (32 configs) on 2008–2016 only (`configs/ablation_phase2_tune0816.yaml`, scai4). Selected config (highest tuning Sharpe + Pareto): fuzzy_cmeans, l1, **pca_k=48**, tau=2.0, k=10 — identical to the paper's config except pca_k 40→48, so re-tuning re-selects essentially the same model. Ran the full 2008–2021 backtest with it (tag `oos_final`) and sliced the honest daily tranched Sharpe:

| Period | Model | Pers | Mix | Roll | Shrink |
|---|---|---|---|---|---|
| Tuning 2008–2016 (in-sample) | 0.694 | **0.750** | 0.733 | 0.687 | 0.685 |
| **Held-out 2017–2021 (OOS)** | 0.582 | **0.635** | 0.597 | 0.566 | 0.546 |
| Full 2008–2021 | 0.654 | **0.707** | 0.682 | 0.641 | 0.632 |

**Persistence beats the model out-of-sample on daily Sharpe (0.635 vs 0.582, model ranks 3rd) and on terminal wealth (1.511 vs 1.501).** Even in-sample on the honest metric persistence leads; the model only "wins" on the noisy per-window Sharpe it was selected on. Combined with issues 3+4, the paper's central claim of "statistically significant / substantial improvements in risk-adjusted portfolio performance" over persistence does **not** survive honest accounting. The defensible paper is now a *negative/nuanced* result: regime-aware similarity conditioning matches the shrinkage/rolling baselines but does not beat realized-covariance persistence for GMVP on this universe. This needs a decision from the authors on reframing (see summary).

### 2. The model is worst-in-class at the objective the portfolio is built for

GMVP is a variance-minimization exercise, and the model has the highest ex-post GMVP variance of all five methods (9.88×10⁻⁵ vs 8.78 rolling, 8.88 mix, 8.93 persistence, 9.15 shrink — confirmed in `report.csv`). The Sharpe advantage therefore comes entirely from the return side, which a covariance forecast doesn't predict and GMVP doesn't optimize. The standard referee attack: "your covariance forecast produces the worst minimum-variance portfolio; the Sharpe gain is an unexplained return effect, possibly a factor tilt or luck." The paper currently buries this in a half-sentence ("higher realized variance and turnover").

**Fix:** Confront it head-on. (a) Report ex-post variance with significance tests as a co-primary metric and acknowledge the model loses on it. (b) Investigate where the return comes from — factor exposures of the weight differences (market beta, size, low-vol tilt), or at least show the return gain is stable across subperiods. (c) Consider reframing: if your story is "regime-conditioning captures economically relevant structure," a mean-variance or risk-parity evaluation, or Diebold-Mariano tests on the matrix losses, may serve the claim better than GMVP alone.

### 3. [CONFIRMED] The wealth curve double-counts overlapping holding windows (~4× inflation); the Sharpe metric is nonstandard

**Audit result (2026-07, verified against `backtest.csv` + code):**
- `hold_period_portfolio_stats` returns `gmvp_cumret = prod(1 + rp) - 1` over the **full 20-trading-day holding window**.
- `plot_equity_curves` (in `scripts/analysis/core/visualize_backtest_results.py`) builds the curve as `np.cumprod(1 + gmvp_cumret)` over evaluation rows.
- Evaluation rows are spaced **5 trading days apart** (stride=5), but each `gmvp_cumret` spans 20 trading days → every period is compounded ~4×.
- Reproduced the paper number exactly: as-plotted terminal wealth = **9.474** (paper: 9.47), implying **36%/yr CAGR** for a min-variance portfolio over 7.3 years — implausible.
- Honest non-overlapping stitch (anchors ≥20 trading days apart): terminal wealth **1.88** (model), **9.1%/yr CAGR** — plausible.

Corrected terminal wealth, all methods: Model 9.47→**1.88**, Pers 8.42→**1.82**, Mix 6.77→**1.68**, Shrink 4.40→**1.42**, Roll 4.32→**1.41**. Ranking survives, but every terminal-wealth figure in the paper is a ~4× artifact and Figure 4's shape is wrong. Note the honest curve makes model and persistence nearly tied (1.88 vs 1.82), reinforcing issue 5.

Separately, `gmvp_sharpe` is a mean of per-window annualized Sharpes (each from 20 daily obs, ×√252) averaged across dates — noisy, nonstandard, never defined in the paper. This averaging is *not* affected by the compounding bug, but still needs replacing with a proper strategy-level Sharpe.

**Fix:** (a) Rebuild the equity curve from a properly stitched non-overlapping (or 1/H-tranched overlapping) return series — ideally re-instrument the backtest to emit daily portfolio returns so a true daily curve and whole-sample Sharpe can be computed. (b) Define the Sharpe estimator explicitly in the paper and report the whole-sample annualized Sharpe. (c) Correct every terminal-wealth number and redraw Figure 4 with its construction stated in the caption.

**Fix status (code landed, pending batched re-run with issue 1):**
- `run_backtest.py` now emits `results/<tag>/gmvp_daily_returns.parquet` — per-day GMVP returns per method, tagged by anchor + calendar date (covariance path only).
- `scripts/analysis/core/build_equity_curves.py` builds both a **tranched** (all-anchor, overlap-corrected) and a **non-overlapping** daily curve, plus whole-sample annualized Sharpe + terminal wealth (`equity_curve_summary.csv`). Falls back to a non-overlapping stitch of `*_gmvp_cumret` until the re-run produces the per-day artifact.
- `plot_equity_curves` (viz pipeline) rewritten to stitch non-overlapping windows instead of `cumprod` over overlapping rows.
- Honest numbers from the fallback (existing data): terminal wealth model **1.88**, pers **1.82**, mix 1.68, shrink 1.42, roll 1.41; whole-sample (window-level) Sharpe model **0.58** vs pers **0.55** — model and persistence nearly tied, consistent with the block bootstrap.
- Remaining: re-run backtest to produce the per-day artifact + true daily tranched Sharpe, then update Figure 4 and every terminal-wealth/Sharpe number in the paper (batched with the issue-1 tuning-split re-run).

### 4. Silent sample selection: 293 of 662 dates dropped, and the GFC never enters the portfolio results

The 369-date restriction isn't a minor detail. In `hold_period_portfolio_stats`, a single missing asset return makes the whole day's portfolio return NaN, and dates where all 20 holding days are NaN are dropped. That's why nothing before March 2014 survives — which means the 2008 crisis, which the abstract advertises as part of the test ("includes GFC"), contributes zero portfolio evidence. The paper attributes the flat pre-2014 curve to "expanding-window warm-up," which is not the actual mechanism.

**Fix:** Disclose the true mechanism, and preferably fix it: compute portfolio returns over the observed-asset subset each day (renormalize weights over non-missing assets), which should recover most of 2008–2014 and let you actually test the GFC claim. If the GFC results are bad, report them — the crisis analysis section already sets up that narrative honestly.

**Fix status (DONE + validated, scai4 re-run to tag `regime_cov_renorm`):**
- Confirmed: all 293 dropped GMVP dates were NaN purely from `fut @ w` propagation; the *entire* 2008–2013 span (263 dates) had valid matrix losses but zero valid GMVP. Not a data limitation — the bug.
- Fixed via `gmvp_daily_returns_renorm` (in `backtests.py`, wired into `hold_period_portfolio_stats` and the daily-artifact instrumentation): each day, invest over observed names and renormalize weights (policy chosen: renormalize over tradable names; introduces a disclosed time-varying universe).
- Validated: valid GMVP dates **369 → 662** (263 pre-2014 recovered, **62 in the 2008–2009 GFC**). On the 369 previously-valid dates, `model_fro` and `regime_assigned` are byte-identical and 356/369 GMVP values are identical to ~1e-15 (renorm = identity on fully-observed windows); only 13 partial-universe windows legitimately change. The ΔSharpe=41 outlier is the noisy per-window metric on a ~0-variance window, not a bug.

**Headline consequence (couples issues 2, 3, 5).** On the honest whole-sample **daily tranched Sharpe** (all anchors, no double-count, GFC included): Model **0.68**, Pers **0.71**, Mix 0.69, Roll 0.64, Shrink 0.63 — **persistence edges out the model; the field compresses to 0.63–0.71**, nothing like the paper's 1.57-vs-1.00. Even under the paper's own per-window metric, including the GFC collapses the model's lead over persistence from 1.57-vs-1.00 to 1.57-vs-1.53. Terminal wealth still ranks model top (3.67 vs pers 3.45) but that reflects the model taking more variance. Net: the paper's central "substantial improvement in risk-adjusted performance" claim does not survive honest accounting on the current (still in-sample) config. Re-tuning (issue 1) is next; this is the pre-tuning honest baseline.

### 5. The novel component's contribution is small, measured in-sample, and never tested for significance

The paper is commendably honest that (a) the Markov filter is empirically inert (differences < 10⁻⁸), and (b) the regime layer itself — the delta over Cartea et al.-style similarity-only — is the K=1→K=4 gain of ~8% Sharpe (0.996 → 1.079), on the tuning sample, at a non-final configuration, with no significance test. Meanwhile the block bootstrap shows the model is not distinguishable from persistence. So the referee's summary is: "the novel components are either inert (filtering) or worth an untested 8% in-sample (regimes), and the full model doesn't significantly beat the strongest baseline." That's the crux of the accept/reject decision.

**Fix:**
- Run the block bootstrap on the K=4 vs K=1 paired difference at the final configuration on held-out data. This is the single most important missing experiment — it directly tests your contribution. If it's significant, lead with it.
- Rewrite the abstract and intro to stop selling Markov filtering ("Markov transition smoothing, yielding a regime-conditioned mixture-of-local-experts estimator") as a contribution — Section 6.4 admits it does nothing on this data. Sell it as an architectural option, or drop it from the pitch.
- The abstract claims "persistent and interpretable market regimes"; Section 6.4 says diagonal persistence is 28–33% vs a 25% chance level. Fix the abstract.

### 6. Baselines are too weak for ICAIF

The "shrinkage" baseline is a fixed γ=0.3 blend toward the diagonal — a strawman relative to the Ledoit-Wolf estimators you cite (LW has an optimal, data-driven intensity and a market-factor target). There is no EWMA/RiskMetrics, no DCC(-NL), no factor-model covariance, no 1/N portfolio, and the volatility section has no HAR — the universal realized-vol baseline, which you also cite. Reviewers at a finance-AI venue will check exactly this list.

**Fix (minimum viable set):** actual Ledoit-Wolf (sklearn's `LedoitWolf` is a drop-in), EWMA covariance (λ=0.94), and HAR for the volatility section. DCC-NL if time permits. If the model only beats weak baselines, better to know now than from the reviews.

---

## Part 2: Serious but straightforward issues

7. **Survivorship/look-ahead in universe construction.** Selection requires ≥85% availability in 2015–2021 and hand-picks mega-caps known ex post (NVDA!) for a backtest starting 2008. Cross-method comparisons on the shared universe remain internally fair, but absolute performance is inflated. Disclose this prominently in Section 4.1, not implicitly.

8. **Statistical reporting errors in Table 6.** Persistence Sharpe p=0.052 is starred `*` under a stated threshold of p<0.05 (it doesn't qualify), and Roll Frobenius p=0.019 is starred `**` (threshold p<0.01). Reviewers catch these and it damages trust in everything else. Also the sign convention flips meaning within one table (+ is good for Sharpe, bad for Frobenius) — restructure so positive always favors the model, or split columns.

9. **Number provenance.** Sharpes of 0.996/1.079 (K-ablation), 1.943 (Phase 2), 1.573 (full backtest), 1.815 (normal periods) appear without a consistent sample/config label; the K-ablation numbers use a different base config than the Phase 2 winner, on an unstated sample. Add a single "which sample, which config" table, or footnote every number. (`CURRENT_STATE.md` says 1.041, which matches none of them — make sure the repo and paper agree before the code link goes live.)

10. **Stability-enhancement tuning.** γ=0.12 and λ=0.08 are described as "tuned to balance…" and "selected to minimize realized GMVP variance" — on which sample? If on the evaluation sample, that's more leakage; state it and fold it into the tuning-window fix (issue 1). Same for the mixture weights: "fixed a priori" is asserted, but a Sharpe-variance sweep on the same data is described two sentences later; a reviewer will not read that as a priori.

11. **Near-chance regime persistence is mechanically suspicious, not just a limitation.** Consecutive anchors share 47–49 of 50 window days, so their embeddings are nearly identical; hard assignments flipping at near-chance rates suggests cluster instability (FCM memberships hovering near the simplex center, or label noise), not fast regime switching. Investigate before a reviewer does: check membership entropy and whether hard assignments flip between near-tied memberships. Also report seed sensitivity for FCM/GMM — the clustering is stochastic and no variability across seeds is reported anywhere.

12. **GMVP is long-short with unreported leverage.** Gross exposure of unconstrained Σ⁻¹1 weights can be large and time-varying, which confounds both Sharpe and turnover comparisons across methods. Report gross leverage per method, and consider adding the long-only variant (already in your codebase) as a robustness table.

---

## Part 3: Submission mechanics (do these regardless)

13. **Format:** 19 pages, single-column article class, with "Supervisor: Mihai Cucuringu" on the title page. ICAIF uses the ACM sigconf two-column template with a hard page limit (full papers have been 8 pages + references; check this year's CFP). You need to cut roughly half the content — Section 4 (data QC) and the ablation narrative are the natural compression targets (move detail to an appendix/arXiv version).

14. **Anonymization:** ICAIF is double-blind. The title page has names, emails, supervisor, and footnote 1 links your personal GitHub. Use an anonymized repo (e.g., anonymous.4open.science) and strip the acknowledgment. Note also that your supervisor is a co-author of Cartea et al. (2023), your primary reference — phrase the relationship neutrally ("builds on," not insider framing) so the paper doesn't de-anonymize itself.

15. **Citation hygiene:** Cartea et al. (2023) is cited as "Working paper / preprint" with the title repeated in the note — get the SSRN number or the published version. Add missing related work a regime-focused reviewer will expect: Pelletier (2006) regime-switching correlations, jump-model regime detection (Nystrup et al.), and recent ICAIF regime-detection papers — citing the venue's own literature matters for fit.

---

## Priority order

1. **Audit the equity curve for overlap double-counting** (issue 3) — if Figure 4 is inflated, everything downstream changes.
2. **Fix the tuning/evaluation split** (issue 1) — re-tune on pre-2015 or 2015–2018; this rescues the paper's validity.
3. **Block-bootstrap the K=4 vs K=1 difference at the final config** (issue 5) — this is the test of your actual contribution.
4. **Add real baselines: Ledoit-Wolf, EWMA, HAR** (issue 6).
5. **Fix the NaN-driven date dropping so the GFC enters the portfolio evaluation** (issue 4).
6. Rewrite abstract/claims (filtering inert, regimes not persistent, variance not improved), fix Table 6 stars, add number-provenance labels.
7. Convert to ACM template, cut to page limit, anonymize.

The honest self-assessments already in the paper (block bootstrap, inert filter, the "honest magnitude" paragraph) are genuinely to your credit and unusual — keep that tone. The work needed is not more honesty, it's restructuring the experiments so the headline numbers are the honest ones.

---

## Part 4: On "add more signals/complexity" proposals (macro, cross-asset, learned metrics, fancier regimes)

**Headline:** These are performance-engineering ideas, and the paper will not be rejected for insufficient performance — it will be rejected for the validity problems in Part 1. Adding features/learned components on top of an evaluation that already leaks produces a *bigger in-sample number that reviewers trust less*. Validity first; signals second. Do not spend the runway to the deadline on model complexity.

**Triage (ICAIF lens):**

| Idea | Fixes a reviewer objection? | Risk | Verdict |
|---|---|---|---|
| Macro (FRED-MD), lagged | Partially — answers "regimes are endogenous" | Scope blow-up; doesn't touch leakage | v2 / camera-ready |
| Cross-asset (VIX, bonds, credit) | Same; economic interpretation | **Look-ahead/circularity** if not strictly lagged | Strongest idea, still v2 |
| Learned / Mahalanobis metric | No | More params → amplifies "just overfitting?" attack | Skip for this submission |
| Temporal / trajectory matching | No (ties to Path Shadowing) | Research-level; own validation | v2 |
| Adaptive kNN | Marginally (crisis) | Another threshold to tune | Cheap, low priority |
| Novelty detection / confidence fallback | Marginally | Already partly in codebase (blend-to-pers) | Low priority |
| Fancier regime models (VAE, time-varying A) | No — conflicts with own finding | Filter already inert; a rewrite | Skip / premature |
| Feature eng. (skew, kurt, vol-of-vol, dispersion) | No | Dimensionality (kNN curse) | Marginal |
| Learned embeddings (autoencoder, contrastive) | No | Largest overfitting surface; small N | Skip for this submission |

**Key ICAIF-specific caveats:**
1. **Amplifies the #1 problem.** With ~89% of GMVP evidence inside the tuning window, every added feature raises the headline and lowers its credibility. Cannot add degrees of freedom until the evaluation is genuinely OOS. This ordering is non-negotiable.
2. **Look-ahead in the "high-impact" features.** VIX / realized skew-kurt / vol-of-vol are contemporaneous risk measures; legitimate only if strictly lagged. Adding VIX also risks dissolving the novelty into "VIX did the work" — which then *requires* a VIX-conditioned baseline to rule out.
3. **Regime upgrades contradict the paper's own findings.** The filter is already inert and persistence is near-chance. The right move on regimes is *diagnostic* (is near-chance persistence real, or clustering instability from 47–49/50 overlapping window days? — see Part 2 issue 11), not a bigger architecture.
4. **Dimensionality self-contradiction.** Can't warn about the kNN curse of dimensionality and then concatenate VIX + bonds + credit + macro + skew + kurt + vol-of-vol. Each block must earn its place or it degrades neighbor quality.
5. **Where it's genuinely right — but as a different paper.** "Regimes are endogenous (market-return-only)" is a legitimate limitation worth adding to the Discussion now. Macro/cross-asset conditioning (regime-level `P(s_t=k | z_t, M_t)`) is the correct spine for a *v2*, with its own OOS evaluation — not a bolt-on. Frame the current endogenous regimes as a deliberate clean baseline and name macro conditioning as the natural extension: get credit for the idea without owing the experiments.

**What to actually do now:** the Part 3 → Priority order, not this list. The only overlaps worth doing for this submission are (a) framing endogenous regimes as an honest limitation, and (b) possibly a VIX-conditioned *baseline to beat* (not a feature to absorb).
