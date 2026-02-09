# 26W_M279

## Similarity-Based Forecasting Pipeline

This module implements a **similarity-based forecasting framework** for financial time series, designed to be **modular, extensible, and free of lookahead bias**.

The core idea is to:

* compute **similarity across historical market states** using embeddings of raw return windows, and
* forecast **future risk objects** (e.g. volatility, correlation, covariance) by aggregating outcomes from similar past states.

Crucially, **the similarity representation is decoupled from the forecast target**, allowing rich feature engineering without constraining what is being forecast.

---

### High-level workflow

For each anchor time ( t ):

1. **Raw input window**

   * Past returns:
     [
     X_t = R_{t-L+1:t} \in \mathbb{R}^{L \times N}
     ]

2. **Embedding (feature engineering)**

   * A `WindowEmbedder` maps the raw window to a fixed-dimensional vector:
     [
     e_t = f(X_t) \in \mathbb{R}^{D}
     ]
   * All feature engineering (e.g. correlation structure, volatility statistics, eigen-spectra) lives **inside the embedder**.

3. **Similarity search**

   * Nearest neighbors are retrieved via KNN in embedding space across historical windows.

4. **Target construction**

   * A `TargetObject` computes the forecast target using *future* returns only:
     [
     Y_t = g(R_{t+1:t+H})
     ]
   * Targets may be volatility vectors, correlation matrices, covariance matrices, etc.

5. **Aggregation**

   * Neighbor targets are combined using distance-based weights and a target-aware aggregator (e.g. Euclidean mean, log-Euclidean SPD mean).

---

### Key design principles

* **State ≠ Target**
  The similarity representation does *not* need to match the forecast object.

  * Example: similarity on correlation eigenstructure, forecast future volatility.
  * Example: similarity on vol + tail features, forecast correlation.

* **Feature engineering is embedder-driven**
  The pipeline operates directly on raw return windows; all transformations into features or regimes occur inside embedders.

* **Fixed-dimensional similarity space**
  Each time window maps to a fixed (D)-dimensional embedding, enabling efficient KNN search even with a changing asset universe.

* **Lookahead-safe by construction**

  * Similarity uses only past data ([t-L+1, t])
  * Targets use only future data ([t+1, t+H])
  * Asset filtering and NA handling are performed as-of the lookback window

---

### Module overview

```
similarity_forecast/
├── embeddings.py        # Window → embedding (feature engineering)
├── similarity.py        # KNN and optional two-stage similarity search
├── target_objects.py    # Forecast targets (vol, corr, cov, etc.)
├── aggregation.py       # Weighting + aggregation rules
├── pipeline.py          # End-to-end orchestration
├── config.py            # Configuration helpers
└── utils.py             # Shared numerical utilities
```

---

### Example usage

```python
model = SimilarityForecaster(
    embedder=CorrEigenEmbedder(k=20),     # similarity on correlation structure
    target_object=VolTarget(log=True),    # forecast future volatility
    weighting=RBFWeighting(tau=5.0),
    aggregator=EuclideanMean(),
    lookback=60,
    horizon=20,
)

model.fit(returns_df)
y_hat = model.predict_at_anchor(anchor_pos=1000, k=50)
```

---

### Extensibility

This design naturally supports:

* composite embeddings (concatenated feature blocks)
* regime-aware similarity (filter neighbors by regime labels)
* approximate nearest neighbors (ANN) backends
* factor-based or low-rank targets
* asset-level or cross-sectional extensions
