"""
similarity_forecast

Regime-aware similarity-based volatility / covariance forecasting.

Public API includes:
  - Core math + SPD utilities
  - KNN similarity search
  - Weighting schemes
  - Aggregators (Euclidean / Log-Euclidean SPD)
  - Embedders
  - Target objects
  - Regime model (GMM + filtering)
  - Regime-aware similarity forecaster
"""

# =========================
# Core (linalg + knn + weighting + aggregation)
# =========================

from .core import (
    # linalg / SPD
    project_to_spd,
    logm_spd,
    expm_sym,
    cov_from_returns,
    corr_from_cov,
    validate_window,

    # knn
    ExactKNN,

    # weighting
    Weighting,
    InverseDistanceWeighting,
    RankWeighting,

    # aggregators
    Aggregator,
    EuclideanMean,
    LogEuclideanSPDMean,
)

# =========================
# Embeddings
# =========================

from .embeddings import (
    WindowEmbedder,
    CorrEigenEmbedder,
    VolStatsEmbedder,
)

# =========================
# Target objects
# =========================

from .target_objects import (
    TargetObject,
    CovarianceTarget,
    CorrelationTarget,
    VolTarget,
)

# =========================
# Regime model
# =========================

from .regimes import (
    RegimeModel,
)

from .regime_weighting import (
    RegimeAwareWeights,
)

# =========================
# Pipelines
# =========================

from .pipeline import (
    RegimeAwareSimilarityForecaster,
)

# (optional) keep baseline if you still use it
try:
    from .pipeline import SimilarityForecaster
except Exception:
    SimilarityForecaster = None

# Data validation
from .data_validation import data_quality_report, print_data_quality_report

__all__ = [
    # core
    "project_to_spd",
    "logm_spd",
    "expm_sym",
    "cov_from_returns",
    "corr_from_cov",
    "validate_window",
    "ExactKNN",
    "Weighting",
    "InverseDistanceWeighting",
    "RankWeighting",
    "Aggregator",
    "EuclideanMean",
    "LogEuclideanSPDMean",

    # embeddings
    "WindowEmbedder",
    "CorrEigenEmbedder",
    "VolStatsEmbedder",

    # targets
    "TargetObject",
    "CovarianceTarget",
    "CorrelationTarget",
    "VolTarget",

    # regimes
    "RegimeModel",
    "RegimeAwareWeights",

    # pipelines
    "RegimeAwareSimilarityForecaster",
    "SimilarityForecaster",
    "data_quality_report",
    "print_data_quality_report",
]