# Redesign Stage-24 Diagnosis

Status: **hypotheses strongly supported by existing evidence**; collapse root-cause attribution finalized after the PC/SVD train–query decomposition (`results/redesign/collapse_decomp/`).

## Q1 — How is the current PCA embedding fit and scaled?

[`PCAWindowEmbedder`](../similarity_forecast/embeddings.py):

- Expanding past-only windows of length \(L=50\); per-asset demean; `StandardScaler` on flattened \(L\times N\); sklearn `PCA` with `whiten=False`.
- Query uses the same scaler/PCA (`transform`).
- Refit every 20 calendar days in walk-forward.
- **Critical detail:** `embed()` concatenates PC coords with within-window SVD features (`top_evr`, `log_sv`, `score_mean`, `score_std`). With `pca_k=48`, output dim is **\(D=240\)**, and the SVD block is **not** re-standardized relative to the PCA block.

Canon config: [`results/oos_final/config_used.yaml`](../results/oos_final/config_used.yaml).

## Q2–Q3 — Why are held-out fuzzy memberships nearly uniform?

**Not** a train/query scaler bug; **not** a broken FCM formula ([`FuzzyCMeansRegimeClusterer`](../similarity_forecast/regime_clustering.py)).

**Working hypothesis:** representation-geometry failure — high dimension, heterogeneous unscaled feature blocks, their interaction, and/or train–query drift.

Evidence already in hand:

- oos_final: `max π ≈ 0.25` on every row; filtered \(\alpha\) ≡ raw \(\pi\).
- Always-on persistence blend fires 100% (`pct_model_blended_to_pers: 100%`).
- 6-D `EconomicStateEmbedder` restores peaked memberships elsewhere in the tree.

### Collapse decomposition result (`results/redesign/collapse_decomp/`)

Fit 2008–2012; query predict 2013–2016; both via `cmeans_predict` (same path as walk-forward queries).

| Representation | Dim | Train max \(p\) | Query max \(p\) | Query centroid contrast |
|---|---:|---:|---:|---:|
| PCA+SVD legacy | 240 | 0.250 | 0.250 | ~0 |
| PCA-only | 48 | 0.250 | 0.250 | ~0 |
| PCA+SVD **jointly standardized** | 240 | 0.531 | 0.525 | ~2.2 |
| PCA-only | 15/10/5 | ~0.25 | ~0.25 | ~0 |
| Market state | 6 | 0.670 | 0.640 | ~3.9 |

**Interpretation (updated):** collapse is **not** “high-\(D\) alone.” Joint standardization of the hybrid restores moderate membership peaking, so **heterogeneous unscaled SVD blocks** were a major contributor. But PCA-only even at \(D=5\) still yields near-uniform FCM memberships with near-zero centroid contrast (centers essentially equidistant), so **PCA geometry also fails to support \(K=4\) soft regimes** under FCM. Market-state features produce informative train *and* query memberships with no large train/query asymmetry in this window — arguing against pure extrapolation drift as the sole story for the old failure (at least under `cmeans_predict` for both).

## Q4 — Where does the \(K=4\) effect come from?

With query \(\alpha\equiv 1/K\), expert mixing is uninformative. Residual \(K>1\) structure is almost entirely **historical neighbor regime weights** \(\Pi_{ik}\) ([`RegimeAwareWeights`](../similarity_forecast/regime_weighting.py)).

Redesign must separate:

1. **Historical stratification** — partitioning analogs into latent groups.
2. **Current-state prediction** — informative \(p_t\) / \(\bar p_{t,H}^{\mathrm{pred}}\).

## Q5 — Reuse cleanly

| Component | Path |
|---|---|
| Neighbor eligibility \(a\le t-H-g\) | `pipeline.py` |
| ExactKNN / kernels | `core.py` |
| SPD / log-Euclidean | `core.py`, aggregators |
| Baselines (roll/pers/EWMA/LW/OAS) | `backtests.py` |
| Walk-forward cadence | `run_backtest.py` |

## Q6 — Delete rather than patch

- Confidence→persistence blend (static 35% mixer)
- `alpha_smooth_frac` as regime hygiene
- High-D PCA+SVD hybrid as regime state (pending decomp)
- Claiming Markov filter as active under \(\pi\equiv 1/K\)
- Hard argmax storytelling from flat soft probs
- Always-on `cov_mix_weights` inside the “model” claim
- Leverage gate / publishability overlays as core method

## Q7 — Leakage control

**Correct** in the current harness: neighbor gap, expanding fit, past-only PCA, future used only for metrics.

Gap fixed in redesign: tune / val / test split is a **first-class** `ExperimentRunner` flag (2008–2012 invent; 2013–2016 select; 2017–2021 locked test). Hyperparameters freeze after 2016; **parameters keep expanding** OOS.

## Redesign architecture

```text
compact state → similarity-only → optional regime → C_η(forecast & target) → decision eval
```

Primary decision metric: **conditioned GMVP variance regret** (weights from \(C_\eta\); quadratic form on raw \(\Sigma^{\mathrm{real}}\)).
