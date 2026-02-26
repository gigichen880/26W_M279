# run_regime_similarity.py
import pandas as pd

from similarity_forecast.data_validation import print_data_quality_report
from similarity_forecast.embeddings import CorrEigenEmbedder
from similarity_forecast.target_objects import CovarianceTarget
from similarity_forecast.core import LogEuclideanSPDMean
from similarity_forecast.regimes import RegimeModel
from similarity_forecast.pipeline import RegimeAwareSimilarityForecaster

# returns_df: index=dates, columns=tickers, values=daily returns (3655 x 100)
returns_df = pd.read_parquet("data/processed/returns_universe_100.parquet")

# Data quality report (NAs are normal for real financial data)
print_data_quality_report(returns_df)

# Optional: filter out stocks with too many NAs (e.g. keep <30% NAs)
FILTER_HIGH_NA_STOCKS = False
if FILTER_HIGH_NA_STOCKS:
    na_by_stock = returns_df.isna().sum(axis=0) / len(returns_df)
    good_stocks = na_by_stock[na_by_stock < 0.3].index
    print(f"\nFiltering to {len(good_stocks)} stocks with <30% NAs (from {len(returns_df.columns)})")
    returns_df = returns_df[good_stocks]

embedder = CorrEigenEmbedder(k=32)
target = CovarianceTarget(ddof=1)
aggregator = LogEuclideanSPDMean(eps_spd=1e-8)

regime_model = RegimeModel(n_regimes=4, random_state=0)

model = RegimeAwareSimilarityForecaster(
    embedder=embedder,
    target_object=target,
    aggregator=aggregator,
    lookback=60,
    horizon=20,
    regime_model=regime_model,
    tau=1.0,
)

model.fit(returns_df)

# predict at some anchor position in sample space
yhat, dbg = model.predict_at_anchor(anchor_pos=500, k=50, return_debug=True)
print("yhat shape:", yhat.shape)
print("alpha:", dbg["alpha"])