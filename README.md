# Similarity-Based Covariance Forecasting (US Equities)

Regime-aware **kNN similarity** in embedding space forecasts covariance (or volatility), evaluated in a **walk-forward** backtest with **GMVP** portfolio metrics and classical baselines (rolling, persistence, shrinkage).

**Core code:** `similarity_forecast/` (embeddings, regimes, aggregation) · **`run_backtest.py`** (full evaluation loop).

---

## Quick start

```bash
pip install -r requirements.txt
```

Large processed returns (`data/processed/*.parquet`) are gitignored—obtain data separately if you clone fresh.

### Main backtest (primary entry point)

```bash
# Covariance: model, mix, roll, pers, shrink + matrix & GMVP metrics
python run_backtest.py --config configs/regime_covariance.yaml

# Volatility target (scalar vol metrics; no GMVP)
python run_backtest.py --config configs/regime_volatility.yaml
```

**Outputs (canonical layout):** `results/<tag>/` with `tag` from the config (e.g. `regime_covariance`).

| Artifact | Description |
|----------|-------------|
| `backtest.parquet`, `backtest.csv` | Per–refit-date metrics for all methods |
| `report.csv` | Means by method (Fro, Stein, GMVP Sharpe/var, turnover, …) |
| `config_used.yaml` | Resolved config snapshot |

Typical runtime: on the order of **minutes** for the default date range and stride (machine-dependent).

### Regenerate figures from an existing backtest

```bash
python -m scripts.analysis.run_all --target covariance
python -m scripts.analysis.run_all --target volatility
```

This runs visualization, statistical comparison, and regime scripts against **`results/<tag>/backtest.*`** (does **not** re-run the backtest). Options: `--skip-stats`, `--skip-viz`, `--skip-regime`, `--skip-ablation-figs`.

**Individual steps (covariance):**

```bash
python -m scripts.analysis.core.visualize_backtest_results --config configs/viz_regime_covariance.yaml
python -m scripts.analysis.core.visualize_statistical_comparison --input results/regime_covariance/backtest.parquet --target covariance
```

---

## Key results (where to look)

### Tables

| File | Use |
|------|-----|
| `results/regime_covariance/report.csv` | **Primary** summary: mean errors + GMVP stats by method |
| `results/regime_covariance/figs/statistical_comparison/*_meanbars.png` | Paired **t**-tests: model/mix vs baselines |
| `results/regime_covariance/figs/statistical_comparison/forecast_correlation.png` | Correlation of forecasts across methods |

### Figures (covariance)

| Path | Content |
|------|---------|
| `figs/raw_temporal/equity_curves_gmvp.png` | Cumulative GMVP wealth (methods configurable via `equity_methods` in `configs/viz_regime_covariance.yaml`) |
| `figs/raw_temporal/method_overlays.png` | Metric time series |
| `figs/raw_temporal/covariance_error_fro_timeseries.png` | Frobenius error over time |
| `figs/regime/` | Regime timeline, transition matrix, performance by regime |

### Paper-style numbers (terminal wealth & mean horizon Sharpe)

```bash
python -m scripts.analysis.summarize_gmvp_equity_stats --input results/regime_covariance/backtest.csv
```

Chains `{method}_gmvp_cumret` into terminal wealth and reports mean `{method}_gmvp_sharpe` with **% vs baselines**.

### Phase 2 joint grid (shortlist + Pareto)

After `ablation_phase2_joint` runs: `ablation_summary.csv`, `pareto/pareto_frontier.csv`, Pareto scatter plots. **Paper-ready bar charts** (16 configs after deduping `l1` ≡ `lp_p1`):

```bash
python -m scripts.analysis.ablation.grid_ablation_metric_figs \
  --summary results/regime_covariance/ablation_phase2_joint/ablation_summary.csv \
  --report
```

→ `figs_grid_metrics_report/` + `deduped_grid_summary.csv`.

### Mixing weights sweep (optional)

```bash
python -m scripts.analysis.sweep_cov_mix_weights --two-way --step 0.1 --plot
```

Writes `results/<tag>/sweep_mix_weights/sweep_mix_weights.csv` and Pareto-style picks.

---

## Pipeline overview

1. **Embed** rolling return windows (`PCAWindowEmbedder`, `CorrEigenEmbedder`, …).
2. **Regime assignment** (GMM, fuzzy c-means, etc.) → soft memberships \(\pi_t\).
3. **Filter** regime posteriors with a transition matrix (optional soft transitions).
4. **kNN** in embedding space with kernel \(\kappa \propto \exp(-d/\tau)\).
5. **Aggregate** neighbor covariances (e.g. log-Euclidean) with regime-aware weights.

**GMVP** weights are computed from each method’s covariance (or precision) forecast; **mix** blends model + shrink (+ optional persistence) via `mixing.mix_lambda` or `mixing.cov_mix_weights`.

Details: [docs/MODEL_VS_MIX_DESIGN.md](docs/MODEL_VS_MIX_DESIGN.md), [docs/DIAGNOSIS_GMVP_VARIANCE.md](docs/DIAGNOSIS_GMVP_VARIANCE.md), [docs/REGIME_ANALYSIS_GUIDE.md](docs/REGIME_ANALYSIS_GUIDE.md).

---

## Hyperparameter & design experiments

| Workflow | Command / config |
|----------|------------------|
| **Phased covariance** (marginals → joint shortlist → Pareto) | `python -m scripts.analysis.ablation.run_phased_covariance_ablation --phase 1` (or `0`, `2`, `3`) · configs `ablation_phase1_covariance.yaml`, `ablation_phase2_joint_shortlist.yaml` |
| **Joint τ × k + Pareto** | `python -m scripts.analysis.ablation.run_joint_gmvp_grid` · `configs/ablation_joint_gmvp_grid.yaml` |
| **Pareto only** (existing grid summary) | `python -m scripts.analysis.ablation.pareto_gmvp_report --summary <path>/ablation_summary.csv` |
| **One-at-a-time** design axes | `python -m scripts.analysis.ablation.run_ablation --config configs/ablation_covariance.yaml` |
| **K regimes** | `python -m scripts.analysis.ablation.run_k_ablation --config configs/regime_covariance.yaml` |

Phase 2 default grid size in YAML may differ from your run; see `ablation_phase2_joint_shortlist.yaml`.
`read_parquet` failures: `scripts/analysis/utils/backtest_io.py` falls back to CSV when present.

---

## Data

- **Returns:** `data/processed/returns_universe_100.parquet` (100-stock universe; see `data/universes/`).
- **Coverage:** ~98.5% mean availability (see `data/universes/` metadata).

---

## Repository layout

```
26W_M279/
├── configs/              # regime_covariance.yaml, viz_*, ablation_*
├── similarity_forecast/  # Core forecasting pipeline
├── run_backtest.py       # Main evaluation entry point
├── scripts/analysis/     # Visualization, ablation, diagnostics
├── results/              # Outputs (often gitignored partially)
├── docs/                 # Design notes, figure catalog, organization rules
└── data/                 # Raw/processed (large files gitignored)
```

**Organization rules:** [docs/FILE_ORGANIZATION_RULES.md](docs/FILE_ORGANIZATION_RULES.md)  
**Figure catalog:** [docs/FIGURES_CATALOG.md](docs/FIGURES_CATALOG.md)  
**Extended results narrative:** [docs/RESULTS_FINAL_REPORT.md](docs/RESULTS_FINAL_REPORT.md)

---

## Other utilities

| Script | Purpose |
|--------|---------|
| `python run_regime_covariance.py` | Small demo / single-anchor test (not the full backtest) |
| `python -m scripts.analysis.case_study_neighbors --config configs/regime_covariance.yaml --date YYYY-MM-DD` | Neighbor diagnostics for a given anchor |
| `python -m scripts.analysis.ablation.run_regime_clustering_ablation` | Clustering backend comparison |

---

## References

_Add your proposal / key papers here._

---

*Last updated: March 2026*
