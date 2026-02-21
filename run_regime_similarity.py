# run_regime_similarity.py
import pandas as pd

from similarity_forecast.embeddings import CorrEigenEmbedder
from similarity_forecast.target_objects import CovarianceTarget
from similarity_forecast.aggregation import LogEuclideanSPDMean
from similarity_forecast.regimes import RegimeModel
from similarity_forecast.pipeline import RegimeAwareSimilarityForecaster

# returns_df: index=dates, columns=tickers, values=daily returns
returns_df = pd.read_parquet("returns.parquet")  # <-- replace

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