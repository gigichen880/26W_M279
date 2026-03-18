# Context Embedding Table

Embedders map a lookback window of returns `(L, N)` to a fixed-size vector for kNN similarity. Config key: `embedder.name` in regime configs.

| Family | Name (config) | Class | What it captures |
|--------|----------------|--------|------------------|
| 1 | `corr_eig` | CorrEigenEmbedder | Top-k **log-eigenvalues** of the window correlation matrix (correlation structure, not covariance scale). |
| 2 | `vol_stats` | VolStatsEmbedder | **Cross-sectional log-vol** distribution: mean, std, quantiles (e.g. 5th, 25th, 50th, 75th, 95th), IQR; **vol trend** (1st vs 2nd half of window); **vol concentration** (HHI). Fixed dimension independent of N. Used for realized-vol forecasting. |
| 3 | — | HybridStateEmbedder | **Multi-view:** (A) factor strength (top-k log singular values + EVR + entropy); (B) vol regime (mean/std/quantiles of per-asset vol); (C) serial dependence (lag-1 autocorr mean/std); (D) correlation (mean off-diag corr, top eigenvalue, eigengap); (E) tail proxy (mean/std of \|returns\|). *Present in code; not currently selectable via config.* |
| 4 | `pca` | PCAWindowEmbedder | **Global PCA** on flattened windows (L×N) fit on training data; per-window embedding: **PCA coordinates** (k), **within-window SVD** (explained variance ratio, log singular values, time-score mean/std). Default for covariance pipeline. |

**Currently used in configs:**
- **Covariance:** `pca` (default) or `corr_eig` (ablation).
- **Volatility:** `vol_stats` (recommended; similarity = similar past vol regime).
