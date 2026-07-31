# Evaluation-spec answers (Bree B2)

**Audience:** Devansh → Methods reproducibility paragraph / pseudocode (D4).  
**Format:** question → answer → `file:line` evidence.  
**Config frozen for headlines:** `results/oos_final/config_used.yaml`.

---

### Q1. Exact target construction — is covariance of days \(t+1,\ldots,t+h\)?

**Answer:** Yes. For each raw anchor index \(t\) (= last day of the lookback window), the future window is the half-open slice `R[t+1 : t+H+1]` (exactly \(H\) calendar rows after the anchor). The covariance target is the sample covariance of those future returns (`ddof=1`), then projected to SPD.

**Evidence:**
- Window construction: `similarity_forecast/pipeline.py:115-118` (`fut = slice(anchor + 1, anchor + H + 1)`).
- Target: `similarity_forecast/target_objects.py:29-37` (`CovarianceTarget.target` → `cov_from_returns` + `project_to_spd`).
- `oos_final`: `horizon: 20`, `lookback: 50`, `ddof: 1` in `results/oos_final/config_used.yaml:10-12,27`.

---

### Q2. Neighbor eligibility — is the condition \(t_i + h \le t\)?

**Answer:** Stricter. A neighbor with anchor \(a\) is eligible only if

\[
a \le t - H - g
\]

with `neighbor_gap` \(g=10\) and \(H=20\) in `oos_final`, i.e. \(a \le t-30\). That implies the neighbor’s future window ends at \(a+H \le t-g = t-10\), so labels are available with a gap buffer beyond mere non-overlap (\(t_i+h\le t\)).

**Evidence:**
- Filter: `similarity_forecast/pipeline.py:328-336` (`cutoff = raw_anchor - H - gap`, `anchor_rows_[idx] <= cutoff`).
- Config: `results/oos_final/config_used.yaml:46` (`neighbor_gap: 10`).

---

### Q3. Do PCA / clustering / transition matrices refit walk-forward in 2017–2021?

**Answer:** Yes — estimation continues; hyperparameters stay frozen. On `refit_mode: days` with `refit_every_days: 20`, the forecaster is rebuilt on the expanding training prefix whenever 20 calendar days have elapsed since the last refit. That rebuild re-fits the embedder (PCA), regime clustering (FCM), transition matrix, and kNN index on all eligible windows up to the refit date. Tuned knobs (\(k\), \(\tau\), `pca_k`, fuzziness, \(\gamma\), mix weights, etc.) are not re-selected on the held-out window.

**Evidence:**
- Refit gate: `run_backtest.py:445-499` (`do_refit` when `(anchor_date - last_refit_date).days >= refit_every_days`, then rebuild pipeline).
- Config: `results/oos_final/config_used.yaml:49-51` (`refit_mode: days`, `refit_every_days: 20`).
- Pipeline `fit()` rebuilds embeds, targets, regimes, transitions, kNN: `similarity_forecast/pipeline.py:132+`.

---

### Q4. Eigenvalue floor (`stability.floor_eps`) — value and where applied?

**Answer:** `floor_eps: 0.0015` with `apply_floor_to: gmvp_only`. Every method’s covariance is passed through `_spd_floor` before GMVP weight construction. Matrix-loss metrics (Frobenius / Stein / KL) use the **unfloored** model forecast when `apply_floor_to == "gmvp_only"` (floored copies are for portfolio weights only).

**Evidence:**
- Config: `results/oos_final/config_used.yaml:71-72`.
- Floor helper: `run_backtest.py:61` (`_spd_floor`).
- Applied to GMVP inputs: `run_backtest.py:709-714`.
- Comment that Fro/Stein/KL keep unfloored model when `gmvp_only`: `run_backtest.py:700`.

---

### Q5. Fuzzy exponent?

**Answer:** Fuzzy C-Means fuzziness \(m=2.0\) (`regime_clustering.params.fuzziness`).

**Evidence:** `results/oos_final/config_used.yaml:20-26`; mirrored in `configs/regime_covariance.yaml:30`.

---

### Q6. Order of diagonal shrinkage vs log-Euclidean mixing?

**Answer:** **Log-Euclidean (regime-mixture) aggregation first, then diagonal shrinkage.**

1. Per-regime neighbor targets are aggregated with the log-Euclidean aggregator and mixed by filtered regime weights \(\alpha\) → `yhat`, then SPD-postprocessed.
2. Optional output shrink toward diagonal is applied afterward:  
   \(\hat\Sigma \leftarrow (1-\gamma)\hat\Sigma + \gamma\,\mathrm{diag}(\hat\Sigma)\) with \(\gamma=0.12\).

There is no diagonal shrink *inside* the log-Euclidean mean; shrink is a post-aggregation stabilizer on the final matrix.

**Evidence:**
- Aggregate then tensordot \(\alpha\): `similarity_forecast/pipeline.py:368-374`.
- Then `output_shrink_toward_diag`: `similarity_forecast/pipeline.py:375-380`.
- Aggregator name `logeuc`, \(\gamma=0.12`: `results/oos_final/config_used.yaml:31,40-41`.

---

### Q7. Does confidence-fallback-to-persistence ever fire? (`guardrail_stats.json` says 0%)

**Answer:** Distinguish two mechanisms:

| Mechanism | What `guardrail_stats.json` reports | Fires on `oos_final`? |
|-----------|-------------------------------------|------------------------|
| Trace-ratio / invalid **guardrail** (replace model with shrink) | `pct_guardrail_triggered: 0.0` | **No** (0/662 days; mode `invalid_only`) |
| **Confidence blend model→persistence** | `pct_model_blended_to_pers: 100.0`, `model_pers_blend_lambda_mean ≈ 0.65` | **Yes, every day** |

Soft filtered regime posteriors in the export are degenerate-uniform (`max α_k ≈ 0.25` ⇒ adjusted confidence ≈ 0). Contributing factor: `alpha_smooth_frac: 0.08` blends filtered \(\alpha\) toward uniform (`similarity_forecast/pipeline.py:311-315`). With `model_blend_to_pers_strength: 0.35` and threshold `0.75`, uncertainty is 1 every day, so

\[
\lambda_{\mathrm{model}} = 1 - 0.35\cdot 1 = 0.65
\quad\Rightarrow\quad
\hat\Sigma \leftarrow 0.65\,\hat\Sigma_{\mathrm{model}} + 0.35\,S_{\mathrm{pers}}.
\]

So: the *guardrail* never fires; the *confidence→persistence blend* fires at full configured strength on **100%** of anchors. The reported “model” series is continuously \(0.65\cdot\hat\Sigma_{\mathrm{pipeline}}+0.35\cdot S_{\mathrm{pers}}\). Do not tell reviewers “no persistence fallback.”

Note: mean stored `guardrail_trace_ratio` ≈ 2.33 can sit near/above the band edge, but mode `invalid_only` means **ratio alone does not trigger replacement**; prior trace-rescale is intended to pull scale into band before the check (`run_backtest.py:620-634`).

**Evidence:**
- Stats: `results/oos_final/guardrail_stats.json` (entire file).
- Blend math: `run_backtest.py:642-672`.
- Config strength/threshold/power: `results/oos_final/config_used.yaml:73-75`.
- Soft-prob degeneracy check: every anchor `max(regime_prob_*)≈0.25` in `results/oos_final/backtest.csv` (see B4 stability note).

---

## Additional reproducibility facts (likely referee follow-ups)

### Q8. What is “mix” vs “model”?

**Answer:** `model` is the pipeline forecast after output diagonal shrink and the model↔pers confidence blend above. `mix` is a separate convex combination of covariances using `cov_mix_weights` (oos_final: model 0.25, shrink 0.35, pers 0.40, roll 0.0), not the same object as `model`.

**Evidence:** `results/oos_final/config_used.yaml:63-68`; mix construction `run_backtest.py:674-694`.

### Q9. Anchor frequency / stride?

**Answer:** Evaluation anchors every `backtest.stride: 5` trading days; training windows inside `fit` use `model.sample_stride: 3`.

**Evidence:** `results/oos_final/config_used.yaml:28,45`.

### Q10. Long-only?

**Answer:** No — `long_only: false` (long–short GMVP).

**Evidence:** `results/oos_final/config_used.yaml:47`.

### Q11. kNN metric and \(k\)?

**Answer:** `knn_metric: l1`, `k_neighbors: 10`, regime temperature `tau: 2.0`, `pca_k: 48`.

**Evidence:** `results/oos_final/config_used.yaml:14,29-30,35,44`.

### Q12. Transition estimator?

**Answer:** Soft transitions with Laplace smoother `trans_smooth: 10.0`. At predict time, default filtered weighting runs one-step \(\alpha_t \propto (\alpha_{t-1} A)\odot \pi_t\).

**Evidence:** `results/oos_final/config_used.yaml:16-17`; filter path `similarity_forecast/pipeline.py:297-306`; soft \(A\) counts `similarity_forecast/regimes.py:80-111`.

### Q13. Full model post-processing stack (order)?

**Answer:** log-Euclidean kNN aggregate → SPD postprocess → diagonal shrink (\(\gamma=0.12\)) → trace rescale into \([0.5, 2.5]\times\mathrm{tr}(S_{\mathrm{roll}})\) → guardrail (invalid→shrink; never on OOS) → confidence→pers blend (\(\lambda\approx 0.65\)) → eigenvalue floor for GMVP only.

**Evidence:** `similarity_forecast/pipeline.py:368-380`; `run_backtest.py:620-714`; guardrail bounds `results/oos_final/config_used.yaml:58-61`.

### Q14. PCA embedding dimension / fit-window quirk?

**Answer:** `pca_k=48` is the PCA component count; the returned embedding concatenates PCA coords with within-window SVD features (explained-variance ratios, log singular values, score mean/std), each padded to `k`, so the feature width is \(5\times\) `pca_k`. Pipeline past at anchor \(t\) is `R[t-L+1:t+1]` (includes day \(t\)); `PCAWindowEmbedder` fit windows use `R[anchor-L:anchor]` (ends the day before) — same length \(L=50\), shifted by one day.

**Evidence:** `similarity_forecast/embeddings.py:285-294,380-414`; pipeline past `similarity_forecast/pipeline.py:116-117`.

### Q15. GMVP definition?

**Answer:** Long–short allowed. Unconstrained GMVP \(w\propto\Sigma^{-1}\mathbf{1}\) with ridge \(10^{-8}\). Weights held fixed over the \(H\)-day future window; daily returns via `gmvp_daily_returns_renorm`; window Sharpe annualized with \(\sqrt{252}\).

**Evidence:** `similarity_forecast/backtests.py:417-452,518-557`; holding loop `run_backtest.py:748-752`.

### Q16. Persistence baseline window?

**Answer:** Persistence uses the *previous* horizon window \([t-H+1,\ldots,t]\) as the forecast of \([t+1,\ldots,t+H]\).

**Evidence:** `similarity_forecast/backtests.py:202-207`.

---

## Quick reference — `oos_final` knobs

| Knob | Value |
|------|--------|
| Lookback / horizon | 50 / 20 |
| Regimes / FCM \(m\) | 4 / 2.0 |
| PCA \(k\) / KNN | 48 / L1, \(k=10\), \(\tau=2.0\) |
| Neighbor gap | 10 (\(\Rightarrow a+H+10\le t\)) |
| Aggregator | log-Euclidean → then diag shrink \(\gamma=0.12\) |
| Refit | every 20 calendar days, expanding |
| Floor | 0.0015, GMVP only |
| Guardrail | `invalid_only` → **0% fire** |
| Pers blend | strength 0.35 → **100% fire**, \(\lambda\approx 0.65\) |
| Eval stride / burn-in | 5 / 252 |
| \(n\) anchors | 662 (2008-10-10 … 2021-11-26) |

---

## One-paragraph Methods paste (starter for D4)

Hyperparameters (including \(k=10\), \(\tau=2\), `pca_k=48`, FCM fuzziness \(m=2\), output diagonal shrink \(\gamma=0.12\), mix weights, and `floor_eps=1.5\times10^{-3}\)) are selected on 2008--2016 only and then frozen. During 2017--2021 the PCA embedder, FCM regimes, transition matrix, and kNN index continue to refit every 20 calendar days on the expanding history. The forecast target at anchor \(t\) is the sample covariance of returns on days \(t+1,\ldots,t+H\) (\(H=20\)). Neighbors must satisfy \(a\le t-H-g\) with gap \(g=10\). Regime-conditional neighbor covariances are combined with a log-Euclidean aggregator and then shrunk toward their diagonal (\(\gamma=0.12\)). An eigenvalue floor of \(1.5\times10^{-3}\) is applied only when forming GMVP weights (`apply_floor_to=gmvp_only`); reported matrix losses use the unfloored forecast. The trace-ratio guardrail never triggers on `oos_final`, but because filtered regime weights are effectively uniform, the configured confidence blend replaces 35\% of the model covariance with persistence on every anchor — so the reported model series is continuously \(0.65\cdot\hat\Sigma_{\mathrm{pipeline}}+0.35\cdot S_{\mathrm{pers}}\).
