# Similarity-Based Covariance Forecasting for US Equities

## Project Overview

This project implements similarity-based covariance forecasting for US equities using daily/minutely return data. We build a clean universe of high-quality stocks, extract returns, and (in the pipeline phase) apply SPD geometry, regime detection, and GMVP backtesting against baselines (HAR, DCC-GARCH, Ledoit-Wolf).

## Team

- Devansh Mishra
- Zhiyi Chen

## Project Structure

```
26W_M279/
├── data/               # Dataset storage (raw, processed, universes)
├── scripts/            # Data processing scripts
│   ├── data_extraction/
│   └── universe_selection/
├── notebooks/          # Jupyter notebooks for EDA
├── results/            # Analysis outputs (figures, reports)
├── archive/            # Old attempts (gitignored)
└── similarity_forecast/           # Main forecasting pipeline
```

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
python run_regime_similarity.py
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

* `CorrEigenEmbedder`
* `VolStatsEmbedder`


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

* Euclidean mean
* Log-Euclidean SPD mean (for covariance matrices)

Implemented in:

```
similarity_forecast/regime_weighting.py
similarity_forecast/core.py
similarity_forecast/pipeline_regime.py
```

### Outputs

For each anchor time $t$, the model produces:

* Regime-aware forecast object (covariance, correlation, or volatility)
* Filtered regime posterior $\alpha_t$
* Optional diagnostics:

  * Neighbor indices
  * Similarity weights $\kappa_i$
  * Regime-conditioned forecasts $\hat{y}^{(k)}$
  * Neighbor regime probabilities $\pi_{t_i}(k)$

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
- Toggle in `run_regime_similarity.py` via `FILTER_HIGH_NA_STOCKS`, or use `SimilarityConfig.filter_high_na_stocks` and `high_na_threshold`.

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

* **`fro`** — Frobenius norm
  || Sigma_hat_t − Sigma_t ||_F

* **`kl`** — Gaussian KL divergence
  KL( N(0, Sigma_t) || N(0, Sigma_hat_t) )

* **`stein`** — Stein loss
  tr(Sigma_hat^{-1} Sigma) − log det(Sigma_hat^{-1} Sigma) − N

---

### Predictive Likelihood

* **`nll`** — Gaussian negative log-likelihood of realized future returns
  Average over horizon days of:
  0.5 * ( log det(Sigma_hat_t) + r' Sigma_hat_t^{-1} r )

This evaluates how well the forecast explains actual realized returns.

---

### SPD / Spectral Structure

* **`logeuc`** — Log-Euclidean distance
  || log(Sigma_hat_t) − log(Sigma_t) ||_F

* **`eig_log_mse`** — Mean squared error between log eigenvalues

* **`cond_ratio`** — Condition number ratio
  cond(Sigma_hat_t) / cond(Sigma_t)

These capture structural and conditioning differences between covariance matrices.

---

### Correlation Structure

* **`corr_offdiag_fro`** — Frobenius error on off-diagonal entries of correlation matrices

* **`corr_spearman`** — Spearman rank correlation between upper-triangle correlation entries

These isolate correlation forecasting skill separately from volatility scale.

---

### Portfolio-Based Evaluation

* **`pred_var` / `real_var`**
  Predicted vs realized variance of the ridge-regularized Global Minimum Variance Portfolio (GMVP).

* **`port_mse_logvar`**
  Mean squared error of log variance across a fixed set of evaluation portfolios
  (equal-weight + random long-only portfolios).

* **Stability diagnostics**

  * `turnover_l1` — L1 turnover of GMVP weights
  * `w_hhi` — Herfindahl concentration index
  * `w_max_abs` — Maximum absolute weight

---

### Running Evaluation & Plots

```bash
# Run backtest (writes results/*.csv and results/*.parquet)
python run_backtest_regime_similarity.py

# Visualize results
python scripts/plot_backtest.py \
  --csv results/regime_similarity_backtest.csv \
  --outdir results/figs_regime_similarity
```

## Next Steps

1. Implement similarity-based forecasting pipeline (Done)
2. SPD geometry (log-Euclidean representations) (Done)
3. Regime detection (Done)
4. GMVP backtesting.
5. Comparison vs baselines (HAR, DCC-GARCH, Ledoit-Wolf).

## References

[Add key papers from your proposal.]

---

Last updated: February 2026
