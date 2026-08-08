"""Modular redesign of similarity / optional-regime covariance forecasting.

Architecture (falsifiable at each arrow):
  compact state → similarity-only → optional regime → C_η → decision eval

Protocol:
  2008–2012 methodology development
  2013–2016 model selection (already-defined candidates only)
  2017–2021 locked test (HPs frozen; parameters expand walk-forward)
"""

from .conditioner import relative_eigen_floor, ConditioningConfig
from .metrics import evaluate_forecast

__all__ = [
    "relative_eigen_floor",
    "ConditioningConfig",
    "evaluate_forecast",
]
