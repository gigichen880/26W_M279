# similarity_forecast/target_objects.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from .core import cov_from_returns, corr_from_cov, validate_window, project_to_spd

class TargetObject(Protocol):
    """
    Defines the forecast target computed from FUTURE returns,
    and a postprocess to ensure validity.
    """
    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]: ...



@dataclass(frozen=True)
class CovarianceTarget:
    ddof: int = 1
    eps_spd: float = 1e-8
    max_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        if not validate_window(
            future_returns,
            max_na_pct=self.max_na_pct,
            min_stocks_pct=self.min_stocks_with_data_pct,
        ):
            raise ValueError("CovarianceTarget.target: future window failed validation.")
        cov = cov_from_returns(future_returns, ddof=self.ddof)
        return project_to_spd(cov, eps=self.eps_spd)

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        return project_to_spd(y_hat, eps=self.eps_spd)

@dataclass(frozen=True)
class CorrelationTarget:
    ddof: int = 1
    eps_spd: float = 1e-8
    eps_diag: float = 1e-12

    # NA policy knobs (self-contained)
    max_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        # ---- Hard gate (consistent policy) ----
        if not validate_window(
            future_returns,
            max_na_pct=self.max_na_pct,
            min_stocks_pct=self.min_stocks_with_data_pct,
        ):
            raise ValueError("CorrelationTarget.target: future window failed validation.")

        # ---- Consistent cov (impute + SPD) ----
        Sigma = cov_from_returns(future_returns, ddof=self.ddof)

        # ---- Correlation + SPD projection ----
        C = corr_from_cov(Sigma, eps=self.eps_diag)
        return project_to_spd(C, eps=self.eps_spd)

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        # Ensure SPD first
        A = project_to_spd(y_hat, eps=self.eps_spd)

        # Normalize to correlation-like (diag=1), then SPD-project again
        d = np.sqrt(np.maximum(np.diag(A), self.eps_diag))
        A = (A / d[None, :]) / d[:, None]
        return project_to_spd(A, eps=self.eps_spd)


@dataclass(frozen=True)
class PrecisionTarget:
    """
    Target = precision matrix (Sigma^{-1}) of future returns.
    GMVP weights are w ∝ Sigma^{-1} 1, so forecasting in precision space can better
    align with the portfolio objective and reduce error amplification from inverting a forecast.
    """
    ddof: int = 1
    eps_spd: float = 1e-8
    max_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        if not validate_window(
            future_returns,
            max_na_pct=self.max_na_pct,
            min_stocks_pct=self.min_stocks_with_data_pct,
        ):
            raise ValueError("PrecisionTarget.target: future window failed validation.")
        cov = cov_from_returns(future_returns, ddof=self.ddof)
        Sigma = project_to_spd(cov, eps=self.eps_spd)
        # Precision = Sigma^{-1}; add small ridge so inverse is stable
        n = Sigma.shape[0]
        try:
            P = np.linalg.solve(Sigma + self.eps_spd * np.eye(n), np.eye(n))
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(Sigma)
        P = (P + P.T) / 2.0
        return project_to_spd(P, eps=self.eps_spd)

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        return project_to_spd(y_hat, eps=self.eps_spd)


@dataclass(frozen=True)
class VolTarget:
    """
    Realized volatility target: from future returns window, compute sample covariance,
    then sqrt(diag(Sigma)) = realized vol per asset over the horizon, optionally in log space.
    """
    ddof: int = 1
    eps: float = 1e-12
    log: bool = True

    # NA policy knobs (self-contained)
    max_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        # ---- Hard gate (consistent policy) ----
        if not validate_window(
            future_returns,
            max_na_pct=self.max_na_pct,
            min_stocks_pct=self.min_stocks_with_data_pct,
        ):
            raise ValueError("VolTarget.target: future window failed validation.")

        # ---- Consistent cov (impute + SPD) ----
        Sigma = cov_from_returns(future_returns, ddof=self.ddof)

        # ---- Vol from diag ----
        v = np.sqrt(np.maximum(np.diag(Sigma), self.eps))
        return np.log(v) if self.log else v

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        # vol target is vector, no SPD postprocessing needed
        return y_hat