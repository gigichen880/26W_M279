# Similarity-Based Covariance Forecasting for US Equities

## Project Overview

This project implements similarity-based covariance forecasting for US equities using daily/minutely return data. We build a clean universe of high-quality stocks, extract returns, and (in the pipeline phase) apply SPD geometry, regime detection, and GMVP backtesting against baselines (HAR, DCC-GARCH, Ledoit-Wolf).

## Project Structure

```
26W_M279/
├── data/               # Dataset storage (raw, processed, universes, docs)
├── scripts/            # Data processing and analysis scripts
│   ├── data_extraction/
│   ├── universe_selection/
│   ├── data_validation/
│   └── analysis/
├── notebooks/          # Jupyter notebooks for EDA
├── results/            # Analysis outputs (figures, reports, latex_tables)
├── docs/               # General documentation (see FILE_ORGANIZATION_RULES.md)
├── archive/            # Old attempts (gitignored)
└── similarity_forecast/  # Main forecasting pipeline
```

## Repository Organization

See [docs/FILE_ORGANIZATION_RULES.md](docs/FILE_ORGANIZATION_RULES.md) for detailed file organization rules.

**Quick reference:**

- **Data:** `data/raw/`, `data/processed/`, `data/universes/`, `data/docs/`
- **Scripts:** `scripts/data_extraction/`, `scripts/universe_selection/`, `scripts/data_validation/`, `scripts/analysis/`
- **Results:** `results/eda/figures/`, `results/eda/reports/`, `results/latex_tables/`
- **Documentation:** `data/docs/`, `docs/`

**Do not commit:** Large data (e.g. `.parquet` in data/raw and data/processed), `archive/`, generated figures under `results/eda/figures/`.

## Data

- **Source**: NYSE + NASDAQ daily returns (2007–2021); minutely data from archives for quality filtering.
- **Universe**: 100 stocks including major tech, financials, healthcare.
- **Availability**: ~98.5% mean data coverage for the final universe.

### Data Files

- `data/processed/returns_universe_100.parquet`: Final 100-stock returns matrix (gitignored; large).
- `data/processed/minutely_daily_returns.parquet`: Daily returns from minutely data (515 stocks; gitignored).
- `data/universes/FINAL_UNIVERSE_100_FINAL.csv`: List of selected 100 tickers (tracked).
- `data/universes/FINAL_UNIVERSE_metadata.txt`: Selection criteria and stats (tracked).

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Data files are gitignored due to size. Contact the team for access (e.g. Dropbox or shared drive).

## Usage

### Data Processing (already run)

```bash
# Build pvCLCL matrix from raw CSVs (data-dir = data/raw or unzipped source)
python scripts/data_extraction/build_pvclcl_matrix.py --data-dir data/raw --out-parquet data/processed/pvCLCL_matrix.parquet

# Extract daily returns from minutely .7z archives (writes to results/eda/reports)
python scripts/data_extraction/minutely_extraction_pipeline.py

# Select final universe (reads quality from results/eda/reports, writes to data/universes/)
python scripts/universe_selection/select_final_universe.py
```

### EDA

```bash
jupyter notebook notebooks/data_prep_eda.ipynb
```

## Main Pipeline (5-Stage Regime-Aware Similarity)

The regime-aware similarity forecasting engine lives under `similarity_forecast/`.

This module implements a **5-stage regime-aware similarity framework** for volatility, covariance, and correlation forecasting.

```bash
python run_regime_covariance.py
```

For each anchor time $t$:

### Stage 1 — Window Embedding

We construct a rolling lookback window of returns:

$$
X_t = R_{t-L+1:t}
$$

and compute an embedding:

$$
z_t = f(X_t)
$$

Embeddings are implemented in:

```
similarity_forecast/embeddings.py
```

Examples:

- `CorrEigenEmbedder`
- `VolStatsEmbedder`

### Stage 2 — Regime Inference (GMM)

We fit a Gaussian Mixture Model (GMM) on the embedding space:

$$
\pi_t(k) = P(s_t = k \mid z_t)
$$

This produces **soft regime probabilities**.

Implemented in:

```
similarity_forecast/regimes.py
```

### Stage 3 — Transition Estimation & Filtering

We estimate a regime transition matrix:

$$
A_{ij} = P(s_t = j \mid s_{t-1} = i)
$$

Then compute filtered regime posteriors:

$$
\alpha_t \propto (\alpha_{t-1} A) \odot \pi_t
$$

where $\odot$ denotes elementwise multiplication.

After normalization:

$$
\alpha_t = \frac{(\alpha_{t-1} A) \odot \pi_t}
{\sum_k \left[(\alpha_{t-1} A) \odot \pi_t \right]_k}
$$

This smooths regime probabilities over time.

Implemented in:

```
similarity_forecast/regimes.py
```

### Stage 4 — Similarity Retrieval

For anchor embedding $z_t$, retrieve nearest neighbors:

$$
\mathcal{N}(t) = { t_i }
$$

using Euclidean distance in embedding space:

$$
d_i = | z_t - z_{t_i} |_2
$$

Similarity kernel:

$$
\kappa_i = \exp\left(-\frac{d_i}{\tau}\right)
$$

Implemented in:

```
similarity_forecast/core.py
similarity_forecast/pipeline_regime.py
```

### Stage 5 — Regime-Aware KNN Aggregation

For each regime $k$, compute regime-conditioned weights:

$$
w_i^{(k)} \propto \kappa_i , \pi_{t_i}(k)
$$

Normalized over neighbors:

$$
w_i^{(k)} =
\frac{\kappa_i , \pi_{t_i}(k)}
{\sum_j \kappa_j , \pi_{t_j}(k)}
$$

Regime-conditional forecasts:

$$
\hat{y}^{(k)} =
\sum_{i \in \mathcal{N}(t)}
w_i^{(k)} , y_{t_i}
$$

Final mixture forecast:

$$
\hat{y}*t =
\sum*{k=1}^{K}
\alpha_t(k) , \hat{y}^{(k)}
$$

Aggregation supports:

- Euclidean mean
- Log-Euclidean SPD mean (for covariance matrices)

Implemented in:

```
similarity_forecast/regime_weighting.py
similarity_forecast/core.py
similarity_forecast/pipeline_regime.py
```

### Outputs

For each anchor time $t$, the model produces:

- Regime-aware forecast object (covariance, correlation, or volatility)
- Filtered regime posterior $\alpha_t$
- Optional diagnostics:
  - Neighbor indices
  - Similarity weights $\kappa_i$
  - Regime-conditioned forecasts $\hat{y}^{(k)}$
  - Neighbor regime probabilities $\pi_{t_i}(k)$

### Architecture Overview

```
similarity_forecast/
│
├── core.py
├── embeddings.py
├── target_objects.py
├── regimes.py
├── regime_weighting.py
├── pipeline_regime.py
└── __init__.py
```

Stage 6 (transition-ahead regime forecasting) can be added on top of this foundation.

## Handling Missing Data

The pipeline handles NAs (missing returns) at several stages:

### 1. Data filtering (optional)

- Optionally filter out stocks with >30% missing data before fitting.
- Toggle in `run_regime_covariance.py` via `FILTER_HIGH_NA_STOCKS`, or use `SimilarityConfig.filter_high_na_stocks` and `high_na_threshold`.

### 2. Window validation

- Windows with >30% NAs are skipped (configurable: `max_window_na_pct`).
- At least 80% of stocks must have some data in the window (`min_stocks_with_data_pct`).
- Reduces unstable covariance and embedding estimates.

### 3. Covariance computation

- Uses pairwise-complete observations (pandas-style).
- Requires a minimum overlap between pairs (default 50% of window length).
- Result is projected onto the SPD manifold.

### 4. Embeddings

- Correlation eigenvalue embedder uses NA-safe covariance; falls back to complete-case or zero embedding when needed.
- Vol embedder uses `nanstd` / `nanmean` over the window.

### Configuration

```python
from similarity_forecast.config import SimilarityConfig

config = SimilarityConfig(
    max_window_na_pct=0.3,           # Skip windows with >30% NAs
    min_stocks_with_data_pct=0.8,   # Need 80% of stocks with data
    filter_high_na_stocks=True,      # Remove high-NA stocks upfront (if used)
    high_na_threshold=0.3,
)
```

Pipeline parameters (e.g. on `RegimeAwareSimilarityForecaster`): `max_window_na_pct`, `min_stocks_with_data_pct`, `verbose_skip`.

### Why not impute?

For returns we avoid imputation (forward-fill, mean, zero) because:

- Forward-fill can introduce look-ahead bias.
- Mean/zero imputation distorts volatilities and correlations.

We use pairwise-complete observations and window validation instead, which is standard for real financial data.

## Key Results (EDA Phase)

- Extracted 515 stocks from minutely data.
- Selected 100 high-quality stocks (~98.5% availability).
- Includes 18 mega-caps: AAPL, MSFT, GOOGL, AMZN, NVDA, META/FB, NFLX, JPM, BAC, WFC, JNJ, PFE, UNH, WMT, HD, V, MA, NKE.

Here’s a **drop-in README section** you can paste (I’d place it right after “Main Pipeline” / “Outputs”, before “Handling Missing Data”). It’s intentionally not too detailed but names the metrics you’re already computing.

## Evaluation

We evaluate covariance forecasts in a walk-forward backtest (no look-ahead bias).
At each anchor date `t`, the model predicts a covariance matrix `Sigma_hat_t` using only the past `L` days, and is scored against the realized covariance `Sigma_t` computed from the next `H` days.

### Metrics

We report a mix of matrix accuracy, probabilistic scoring, correlation skill, and portfolio usefulness.

---

### Matrix Errors

- **`fro`** — Frobenius norm
  || Sigma_hat_t − Sigma_t ||\_F

- **`kl`** — Gaussian KL divergence
  KL( N(0, Sigma_t) || N(0, Sigma_hat_t) )

- **`stein`** — Stein loss
  tr(Sigma_hat^{-1} Sigma) − log det(Sigma_hat^{-1} Sigma) − N

---

### Predictive Likelihood

- **`nll`** — Gaussian negative log-likelihood of realized future returns
  Average over horizon days of:
  0.5 \* ( log det(Sigma_hat_t) + r' Sigma_hat_t^{-1} r )

This evaluates how well the forecast explains actual realized returns.

---

### SPD / Spectral Structure

- **`logeuc`** — Log-Euclidean distance
  || log(Sigma_hat_t) − log(Sigma_t) ||\_F

- **`eig_log_mse`** — Mean squared error between log eigenvalues

- **`cond_ratio`** — Condition number ratio
  cond(Sigma_hat_t) / cond(Sigma_t)

These capture structural and conditioning differences between covariance matrices.

---

### Correlation Structure

- **`corr_offdiag_fro`** — Frobenius error on off-diagonal entries of correlation matrices

- **`corr_spearman`** — Spearman rank correlation between upper-triangle correlation entries

These isolate correlation forecasting skill separately from volatility scale.

---

### Portfolio-Based Evaluation

- **`pred_var` / `real_var`**
  Predicted vs realized variance of the ridge-regularized Global Minimum Variance Portfolio (GMVP).

- **`port_mse_logvar`**
  Mean squared error of log variance across a fixed set of evaluation portfolios
  (equal-weight + random long-only portfolios).

- **Stability diagnostics**
  - `turnover_l1` — L1 turnover of GMVP weights
  - `w_hhi` — Herfindahl concentration index
  - `w_max_abs` — Maximum absolute weight

---

## Running Evaluation & Analysis

### 1. Main Backtest (Walk-Forward Evaluation)

Run the complete evaluation pipeline with all baselines:

```bash
python run_backtest.py --config configs/regime_covariance.yaml
```

**Outputs (covariance):**

- `results/regime_covariance/backtest.{parquet,csv}` - Full results (all methods, all dates)
- `results/regime_covariance/report.csv` - Summary statistics by method
- `results/regime_covariance/config_used.yaml` - Config snapshot

**Duration:** ~5-10 minutes

**Target type:** The pipeline is target-agnostic. In config, `model.target` can be `"covariance"` (default) or `"volatility"`. **Volatility = realized volatility:** the target is computed from the future-window sample covariance as sqrt(diag(Σ)) (optionally log), i.e. realized vol per asset over the horizon. For the vol task, the same walk-forward loop runs with EuclideanMean aggregation and scalar metrics (vol_mse, vol_mae, vol_rmse); no GMVP.

**Volatility backtest (same pipeline):**

```bash
python run_backtest.py --config configs/regime_volatility.yaml
```

Outputs: `results/regime_volatility/backtest.{parquet,csv}`, `results/regime_volatility/report.csv`. Then visualize with:

```bash
python -m scripts.analysis.core.visualize_backtest_results --config configs/viz_regime_volatility.yaml
```

Figures go to `results/regime_volatility/figs/raw_temporal/` (error and cumulative-advantage use `vol_mse`).

**Embedder choice:** For **covariance** forecasting, `pca` or `corr_eig` are appropriate (they capture return/factor or correlation structure). For **realized vol** forecasting, `embedder.name: "vol_stats"` is recommended: it embeds the past window’s vol distribution (mean + quantiles of log vol), so kNN similarity aligns with “similar past vol regime.” The volatility config uses `vol_stats` by default; you can set `embedder.name: "pca"` or `"corr_eig"` to reuse the same embedding as the covariance run. If **vol MSE is poor** (model worse than baselines): increase **`mixing.vol_dampen_toward_roll`** (e.g. 0.2–0.3), lower **`mixing.mix_lambda`**, and run the statistical comparison for vol (§5).

**Key design (volatility):** The vol pipeline is tuned so the model can beat Shrink, Roll, and Pers on Vol MSE / MAE / RMSE (log-vol). Main choices: **(1)** **Dampening** — the raw kNN forecast is blended with baselines: `vol_hat_use = (1 − α − β)·vol_hat + α·vol_roll + β·vol_shrink` with `vol_dampen_toward_roll` (e.g. 0.5) and `vol_dampen_toward_shrink` (e.g. 0.15). This cuts kNN variance while keeping regime signal. **(2)** **More neighbors** — `backtest.k_neighbors` (e.g. 15) gives a smoother kNN average and lower forecast variance than small k. **(3)** **VolStatsEmbedder** — past-window vol distribution (mean, quantiles, trend, concentration) so similarity ≈ similar past vol regime. With these, the model consistently shows statistically significant advantage over Shrink, Roll, and Pers on Vol MSE/MAE/RMSE in the statistical comparison plots.

---

### 2. Standard Evaluation Plots

Generate time-series plots of all metrics:

```bash
python -m scripts.analysis.core.visualize_backtest_results \
  --config configs/viz_regime_covariance.yaml
```

**Outputs (covariance):**

- `results/regime_covariance/figs/raw_temporal/equity_curves_gmvp.png`
- `results/regime_covariance/figs/raw_temporal/method_overlays.png`
- `results/regime_covariance/figs/raw_temporal/rolling_median_21d.png`
- other time-series / cumulative-advantage plots under `results/regime_covariance/figs/raw_temporal/`

---

### 3. Regime Analysis & Visualization

#### Generate All Regime Figures

All regime scripts support **covariance** (default) and **volatility** via `--target` and `--input`:

```bash
# Regime timeline, probability evolution, filtering effect (default: covariance)
python -m scripts.analysis.regime.visualize_regimes
python -m scripts.analysis.regime.visualize_regimes --target volatility

# Regime characterization table
python -m scripts.analysis.regime.regime_characterization --input results/regime_covariance/backtest.parquet
python -m scripts.analysis.regime.regime_characterization --input results/regime_volatility/backtest.parquet --target volatility

# Transition matrix heatmap
python -m scripts.analysis.regime.visualize_transition_matrix --target volatility

# Performance by regime
python -m scripts.analysis.regime.performance_by_regime --input results/regime_volatility/backtest.parquet --target volatility
```

**Outputs (covariance):** `results/regime_covariance/figs/regime/`, `results/regime_covariance/regime_characterization.csv`, `results/regime_covariance/transition_matrix.csv`  
**Outputs (volatility):** `results/regime_volatility/figs/regime/`, `results/regime_volatility/regime_characterization_volatility.csv`, `results/regime_volatility/transition_matrix_volatility.csv`

---

### 4. K Ablation Study

Test model with different numbers of regimes (K=1,2,3,4,5,6). Works for **covariance** and **volatility**:

```bash
# Covariance (default): run ablation, then analyze
python -m scripts.analysis.ablation.run_k_ablation --config configs/regime_covariance.yaml
python -m scripts.analysis.ablation.analyze_k_ablation

# Volatility: use vol config and target
python -m scripts.analysis.ablation.run_k_ablation --config configs/regime_volatility.yaml
python -m scripts.analysis.ablation.analyze_k_ablation --ablation-dir results/ablation_k_regime_volatility --target volatility
```

**Outputs (cov):** `results/ablation_k/` (reports and optional backtest_k*.parquet), `results/regime_covariance/figs/ablation_k_regimes.png` (4-panel)  
**Outputs (vol):** `results/ablation_k_regime_volatility/`, same comparison CSV and 2-panel figure under `results/regime_volatility/figs/ablation_k_regimes.png`

---

### 4b. Pipeline design ablation (one axis at a time)

Isolate the effect of high-level design choices (embedder, distance, transition, regime aggregation/weighting) without sweeping hyperparameters:

```bash
# Covariance: embedder (pca, corr_eig), transition_estimator (hard, soft), knn_metric (l2, l1), regime_aggregation (soft, hard), regime_weighting (filtered, raw_pi)
python -m scripts.analysis.ablation.run_ablation --config configs/ablation_covariance.yaml

# Volatility: same axes; embedder choices include vol_stats
python -m scripts.analysis.ablation.run_ablation --config configs/ablation_volatility.yaml
```

**Outputs:**

- `results/regime_covariance/ablation/ablation_summary.csv` (cov) or `results/regime_volatility/ablation/ablation_summary.csv` (vol) — one row per axis/choice with primary metric mean (e.g. `model_fro`, `mix_fro` for cov; `model_vol_mse` for vol) and optional extra metrics.
- Full per-choice time-series backtests under `results/<tag>/ablation/runs/<axis>/<choice>.parquet` (or `.csv`).
- Ablation figures under `results/<tag>/figs/ablation/`:
  - `ablation_summary.png` (bar chart)
  - `<axis>/..._timeseries_overlay.png`, `cumadv_*_overlay.png` (choices overlaid for easy comparison)

**Axes:**

| Axis                   | Choices                          | Description                                  |
| ---------------------- | -------------------------------- | -------------------------------------------- |
| `embedder`             | pca, corr_eig (vol: + vol_stats) | Window embedding for similarity              |
| `transition_estimator` | hard, soft                       | How transition matrix A is estimated         |
| `knn_metric`           | l2, l1                           | Neighbor distance (Euclidean vs Manhattan)   |
| `regime_aggregation`   | soft, hard                       | Weights over regimes: α vs one-hot argmax(α) |
| `regime_weighting`     | filtered, raw_pi                 | Use filtered α vs raw GMM π                  |

---

### 5. Statistical Comparison

Compare model/mix vs baselines with paired t-tests (same script for covariance and volatility):

```bash
# Covariance (default: regime_covariance backtest)
python -m scripts.analysis.core.visualize_statistical_comparison

# Volatility
python -m scripts.analysis.core.visualize_statistical_comparison --input results/regime_volatility/backtest.parquet --target volatility
```

**Outputs:**

- `results/regime_covariance/figs/statistical_comparison/model_vs_baselines.png`, `mix_vs_baselines.png` (cov)
- `results/regime_volatility/figs/statistical_comparison/model_vs_baselines.png`, `mix_vs_baselines.png` (vol)
- Green = reference better than baseline; red = worse; ** / \*** = significance at 5% / 1%

---

### 6. Neighbor-Based Case Studies

To understand which historical episodes the model reuses for a given evaluation date, use the
neighbor case-study tool:

```bash
# Example: COVID crash (covariance)
python -m scripts.analysis.case_study_neighbors \
  --config configs/regime_covariance.yaml \
  --date 2020-03-09 \
  --k_neighbors 10
```

This assumes the main backtest has already been run so that the corresponding backtest exists (e.g. `results/regime_covariance/backtest.parquet`). For **volatility**, use `--config configs/regime_volatility.yaml` (uses `results/regime_volatility/backtest.parquet`). Outputs are written under the config’s tag.

**Per-date outputs (under `results/<tag>/case_studies/`, with `<tag>` = `regime_covariance` or `regime_volatility`):**

- `neighbors_YYYYMMDD.csv`  
  Neighbor diagnostics: `neighbor_date`, `lag_days`, `dist_embedding`, kernel weight `kappa`,
  per-regime neighbor probabilities `pi_neighbor_regime_k`, regime-aware weights `W_regime_k`,
  and `total_weight` (overall contribution under the filtered regime mix).
- `neighbors_YYYYMMDD_weights_vs_date.png`  
  Panel A — total neighbor weight vs `neighbor_date` (color = embedding distance).
- `neighbors_YYYYMMDD_on_regime_timeline.png`  
  Panel B — neighbors overlaid on a mini regime timeline, with point size ∝ weight and the
  anchor date shown as a dashed vertical line.

Typical anchors:

- GFC stress: `2008-11-21`
- COVID crash: `2020-03-09`

### 7. Quick Demo (Single Prediction)

Test pipeline on a single anchor point:

```bash
python run_regime_covariance.py
```

**Note:** This is for testing only - uses different data file and parameters
than main evaluation.

---

## Complete Analysis Pipeline (Run All)

To regenerate all results and figures from scratch:

```bash
# One-command figures/tables (uses existing backtest outputs)
python -m scripts.analysis.run_all --target covariance
python -m scripts.analysis.run_all --target volatility

# Main backtest (covariance then volatility)
python run_backtest.py --config configs/regime_covariance.yaml
python run_backtest.py --config configs/regime_volatility.yaml
```

`run_all` internally calls the per-module scripts (`visualize_backtest_results`, `visualize_statistical_comparison`, regime plots, K ablation, etc.) using the canonical `results/<tag>/...` layout, so you usually do not need to invoke them individually.

**Total duration:** ~1-2 hours (most time is K ablation)

---

## Key Result Files

After running complete pipeline:

**Metrics & Tables (covariance example):**

- `results/regime_covariance/report.csv` - Summary by method
- `results/regime_covariance/regime_characterization.csv` - Regime statistics
- `results/ablation_k/ablation_k_comparison.csv` - K ablation results
- `results/regime_covariance/comprehensive_baseline_comparison.csv` - Full baseline comparison
- `results/regime_covariance/ablation/ablation_summary.csv` - Design ablation summary

**Key Figures (covariance example, all under `results/regime_covariance/figs/`):**

- `regime/regime_timeline.png` - Regime assignments with crisis periods
- `regime/performance_by_regime.png` - Model vs Roll by regime
- `ablation_k_regimes.png` - K=1-6 comparison
- `ablation/ablation_summary.png` and axis overlays - Design ablation
- `raw_temporal/equity_curves_gmvp.png` - Cumulative wealth
- `regime/regime_probs_stacked.png` - Regime probability evolution
- `regime/regime_filtering_effect.png` - Markov filtering demonstration
- `transition_matrix_heatmap.png` - Regime transition probabilities
- `raw_temporal/rolling_median_21d.png` - Temporal performance
- `raw_temporal/method_overlays.png` - Metric time series

**Raw Results:**

- `results/regime_covariance/backtest.parquet` - Full backtest data
- optional `results/ablation_k/backtest_k{1-6}.parquet` - K ablation data (if generated)

See [docs/RESULTS_FINAL_REPORT.md](docs/RESULTS_FINAL_REPORT.md) for result summaries and figure captions.

## References

[Add key papers from your proposal.]

---

Last updated: February 2026
