# ICAIF 2026 Submission Plan — Devansh & Bree

**Paper:** Regime-Aware Similarity-Based Covariance Forecasting
**Deadline:** Sunday **Aug 2, AoE** (no rebuttal; notification Sept 27)
**Internal deadline:** full draft to Mihai by **Friday Jul 31, night** so he can comment over the weekend

**How we're working:**
- **Wed–Thu:** repo work in two parallel tracks (below) — Devansh on Track A, Bree on Track B. The tracks touch different scripts, results folders, and figures, so we don't block each other.
- **Friday:** prose written **together**, once every table and figure exists.
- **Weekend:** Mihai's comments + submit.

---

## Agreed framing (from Jul 28 call with Mihai)

Combine options A + B. The paper claims two things, and only these two:

1. **An interpretable regime-aware forecasting framework that matches standard covariance estimators out-of-sample.** Not "beats" — matches. Parity is the honest result; interpretability is the value-add.
2. **A statistical-vs-economic disconnect finding (headline):** our model wins the statistical matrix metric (Frobenius) while producing the *worst* minimum-variance portfolio; Ledoit-Wolf/OAS win Stein/KL but don't win the portfolio contest either. Metric choice materially changes the ranking of covariance forecasters.

**Regime labels:** no more human-readable names ("Calm Bull", "High Stress", …). They become **Regimes 1–4**, each characterized by a multi-feature **signature** (Mihai's suggestion): realized vol, avg pairwise correlation, mean return, persistence, % of days. Descriptive stats, no narrative labels.

---

## Step 0 — Unblock Bree (Devansh, Wednesday morning, FIRST)

- [x] Commit + push the honest-results work so Bree can pull: `results/oos_final/`, `results/regime_cov_renorm/`, `results/figs_regime_similarity/`, `REGIME_CONDITIONAL_ANALYSIS.md`, `scripts/analysis/core/conditional_gmvp_analysis.py`, and other uncommitted analysis scripts (local commits `3f97bda`, `f8a37b6` are also unpushed)
- [x] Bree: clone repo, `pip install -r requirements.txt`, confirm the results CSVs open *(core deps in `.venv`; full `requirements.txt` matplotlib wheel failed on py3.13 — figures Thu need a binary matplotlib install)*
- [x] `data/processed/returns_universe_100.parquet` is gitignored — Devansh sends it directly (Drive/AirDrop) in case any regeneration script needs raw returns *(present locally)*
- [x] Agree on the regime numbering once (see mapping at bottom) so tables and figures from both tracks use identical labels *(locked in `results/regime_characterization/paper_regime_numbering.json`)*

---

## Track A — Devansh: paper scaffold + portfolio & baseline results

**Wednesday**
- [x] Sync local `paper/` clone to Overleaf head (it drifts — hard reset to origin first)
- [x] Convert draft to **ACM sigconf, double-blind**; get it compiling
- [x] Purge every April-era claim (see Dead Numbers) — delete now so nothing stale survives into Friday
- [x] Post section skeleton in Overleaf (headings + one-liners). During Wed–Thu **only Devansh edits `main.tex`**; all Track B output lands as separate include files (`tables/*.tex`, `figs/regime/*`) to avoid conflicts

**Thursday**
- [x] Baseline comparison table (LaTeX): held-out GMVP Sharpe for all 8 methods + bootstrap ties (from `results/oos_final/`)
- [x] Matrix-loss table: Frobenius / Stein / KL by method — the disconnect in one table
- [x] Disconnect figure or table: statistical rank vs portfolio rank reversal, visible at a glance
- [x] Equity curves figure from honest harness (tranched, GFC included) — `scripts/analysis/core/build_equity_curves.py`
- [x] Vol-target comparison (HAR vs model) — one small table
- [x] Collect disclosure facts: survivorship (universe selected on 2015–21 availability, backtest starts 2008), all tuning on 2008–2016 only (incl. γ=0.12, λ=0.08, mix weights), long-short GMVP leverage

## Track B — Bree: regime characterization & regime-conditional results

Everything here reads from committed results CSVs (`results/oos_final/`, `results/regime_cov_renorm/`, `REGIME_CONDITIONAL_ANALYSIS.md`); ask Devansh if a script needs the raw parquet.

**Wednesday**
- [x] Read `REGIME_CONDITIONAL_ANALYSIS.md` end-to-end — it is the source of truth for everything in this track
- [x] Build the **regime signature table** (Regimes 1–4 × features: realized vol, avg correlation, mean return, persistence, % of days) as a standalone LaTeX include — from `results/regime_characterization` outputs / the conditional analysis script *(→ `paper/tables/regime_signature.tex`; rebuild via `python -m scripts.analysis.regime.build_regime_signature_table`)*

**Thursday**
- [x] Regenerate **regime timeline** and **transition-matrix heatmap** with numbered regime labels (scripts in `scripts/analysis/regime/`; drop the name-mapping step) → export to `paper/figs/regime/` *(also mirrored in `results/regime_characterization/figs/`; rebuild via `python -m scripts.analysis.regime.build_icaif_regime_figures`)*
- [x] **Per-regime performance table**: model − persistence Sharpe gap by regime, tuning era vs held-out (numbers in Canonical Numbers below) as a LaTeX include *(→ `paper/tables/regime_performance.tex`; rebuild via `python -m scripts.analysis.regime.build_regime_performance_table`)*
- [x] One-paragraph bullet summary of the regime-conditional story for Friday's prose: calm-market edge (descriptive, not tradeable), loses in every stress window, Regime "Normal" is the only era-consistent positive pocket (insignificant after multiple testing) *(→ `results/regime_characterization/regime_conditional_prose_bullets.md`; paper label = Regime 3)*
- [x] Sanity-check every number in your tables against Canonical Numbers below; flag any mismatch to Devansh immediately (a mismatch means we're reading a stale results file) *(gaps match Canonical; signature↔transition persistence match; **flag:** `results/oos_final/gmvp_daily_returns.parquet` missing locally so gaps cannot be recomputed from scratch — table uses Canonical Numbers)*

**Thursday night — merge checkpoint (both): DONE (Overleaf 95966d8)** — all tables/figures pushed to Overleaf, compile clean, regime numbering consistent across both tracks. This is the gate for Friday.

---

## Friday — prose, together

Working session (co-write or rapid ping-pong on Overleaf), in this order:

- [ ] **Results prose** around the finished tables (parity result → disconnect → regime-conditional)
- [ ] **Introduction**: motivation → framework → parity → disconnect as the hook
- [ ] **Related work**: shrinkage estimators (Ledoit-Wolf, OAS, EWMA, HAR) · regime-switching models · forecast-evaluation literature (statistical loss vs portfolio outcome — positions the disconnect)
- [ ] **Methodology trim** to fit the 8-page budget — keep the 5-stage pipeline, cut derivations that don't serve the two claims
- [ ] **Limitations + Conclusion**
- [ ] **Abstract** (written last, once results prose is frozen — promises exactly the A+B claims, nothing stronger)
- [ ] **Anonymization sweep**: advisor-connected work (e.g. Cartea, Cucuringu & Jin 2023) cited in neutral third person; no "our prior work"
- [ ] **Page cut + full proofread** against Canonical / Dead Numbers
- [ ] **Email Mihai** with the Overleaf link — Friday night

## Weekend — both

- [ ] **Saturday:** triage Mihai's comments — number/figure fixes vs prose fixes, split on the spot
- [ ] **Sunday:** final compliance pass (page limit, anonymity, template) → **submit with buffer before the AoE cutoff**

---

## Canonical numbers (use ONLY these)

Held-out = 2017–2021; tuning era = 2008–2016. All Sharpes are standard whole-sample daily Sharpes on tranched (overlap-corrected) GMVP returns.

**Held-out GMVP Sharpe:** model **0.582** · persistence **0.635** · EWMA **0.632** · mix 0.597 · rolling 0.566 · OAS 0.554 · Ledoit-Wolf 0.549 · shrinkage 0.546
**Significance:** all pairwise gaps are statistical ties — moving-block bootstrap (block = 20-day horizon); model − persistence Δ = −0.053, p = 0.70, 95% CI [−0.27, +0.20]
**Matrix losses (held-out, corrected 2026-07-30 from oos_final backtest.csv):** Ledoit-Wolf Stein 839 / KL 420 beats model 1095 / 548; model Frobenius 0.0252 is best **only among the properly constructed estimators** (LW 0.0260, OAS 0.0265, EWMA 0.0272, roll 0.0280, pers 0.0295) — fixed-γ shrink (0.0246) and mix (0.0248, Stein 860 / KL 430) are lower still. Do not write "model wins Frobenius outright."
**Ex-post GMVP variance (corrected 2026-07-30 — old 9.9 vs 8.8–9.2 values were from the stale regime_covariance tag):** held-out oos_final per-window means ×10⁻⁵: model **11.47** (highest = worst min-variance portfolio) vs shrink 9.99, mix 9.78, pers 9.64, roll 9.54
**Held-out turnover (mean one-way L1):** model 0.28 · mix 0.28 · roll 0.24 · pers 0.45 · shrink 0.18
**Volatility target:** HAR beats model — MSE 0.176 vs 0.334
**Terminal wealth (corrected 2026-07-30 — old 1.88/1.82 was the stale tag's 93-window nonoverlap stitch):** oos_final tranched full-sample 2008–2021: model **3.60** vs persistence **3.45** (~10%/yr both; all five core methods 2.99–3.60); full-sample tranched Sharpe model 0.654 vs pers 0.707
**Phase-2 tuning objective (2008–2016 grid, replaces the flagged 1.943):** selected fuzzy_cmeans / l1 / pca_k=48 / tau=2.0 / k=10 at **1.646** mean per-window Sharpe (a selection diagnostic, never a headline Sharpe); FCM grid-average 1.61 vs Ward 1.52
**Per-regime (model − persistence Sharpe gap, tuning / held-out):** "Calm Bull" +0.09 / −0.15 · "High Stress" −0.30 / −0.04 · "Moderate Bull" −0.19 / −0.09 · "Normal" **+0.22 / +0.13** (only era-consistent positive pocket, ~15–20% of days, insignificant after multiple testing)
**Descriptive calm-market edge:** non-crisis Sharpe 1.435 (best of all methods), TW 6.66 vs pers 5.22; model loses in every crisis/stress window

## Dead numbers (must NOT appear anywhere)

Retracted April figures — if any survives in the draft, it's a bug:

- Sharpe **1.041** (or any "model beats all baselines" claim) — was in-sample + nonstandard Sharpe
- Terminal wealth **9.47** (or ~4× anything) — equity double-count
- "**Superior in stress / crisis**" — reversed by the honest analysis; the model wins in *calm* markets and loses in stress
- Wilcoxon **p < 0.001** significance claims — replaced by bootstrap ties (p ≥ 0.70)
- Vol RMSE 0.233-era volatility numbers — superseded; HAR wins the vol comparison
- Any result conditioning on `realized_vol` / `avg_corr` from backtest outputs — computed over the *future* window (look-ahead); the "hybrid OOS 0.773" result is fake

## Named-regime → numbered-regime map

**Locked** (Step 0; verified against `results/oos_final` signature table — use `results/regime_characterization/paper_regime_numbering.json`):
- "Calm Bull" → Regime 1 (cluster 0) · "Moderate Bull" → Regime 2 (cluster 2) · "Normal" → Regime 3 (cluster 3) · "High Stress" → Regime 4 (cluster 1)
- Paper prints **Regimes 1–4 only**; legacy names are internal bookkeeping for Canonical Number checks.

---

## Definition of done (Friday night)

- Compiles on ACM sigconf, ≤ page limit, fully anonymized
- Both claims (parity + disconnect) in abstract, intro, and conclusion — nothing stronger anywhere
- Every number matches Canonical Numbers; zero Dead Numbers present
- Regimes numbered with a signature table; identical numbering in every table and figure
- Disclosures present (survivorship, tuning sample, leverage)
- Mihai notified with the Overleaf link
