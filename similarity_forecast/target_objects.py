from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from .core import cov_from_returns, corr_from_cov, project_to_spd


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
    min_periods_ratio: float = 0.5

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T = future_returns.shape[0]
        na_pct = np.isnan(future_returns).sum() / future_returns.size
        if na_pct > 0.5:
            warnings.warn(
                f"Future window has {na_pct:.1%} NAs, covariance may be unstable",
                UserWarning,
                stacklevel=2,
            )
        min_periods = max(2, int(T * self.min_periods_ratio))
        cov = cov_from_returns(future_returns, ddof=self.ddof, min_periods=min_periods)
        return project_to_spd(cov, eps=self.eps_spd)

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        return project_to_spd(y_hat, eps=self.eps_spd)


@dataclass(frozen=True)
class CorrelationTarget:
    ddof: int = 1
    eps_spd: float = 1e-8
    eps_diag: float = 1e-12
    min_periods_ratio: float = 0.5

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T = future_returns.shape[0]
        min_periods = max(2, int(T * self.min_periods_ratio))
        Sigma = cov_from_returns(future_returns, ddof=self.ddof, min_periods=min_periods)
        C = corr_from_cov(Sigma, eps=self.eps_diag)
        return project_to_spd(C, eps=self.eps_spd)

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        A = project_to_spd(y_hat, eps=self.eps_spd)
        d = np.sqrt(np.maximum(np.diag(A), self.eps_diag))
        A = (A / d[None, :]) / d[:, None]
        return project_to_spd(A, eps=self.eps_spd)


@dataclass(frozen=True)
class VolTarget:
    ddof: int = 1
    eps: float = 1e-12
    log: bool = True

    def target(self, future_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T = future_returns.shape[0]
        min_periods = max(2, T // 2)
        Sigma = cov_from_returns(future_returns, ddof=self.ddof, min_periods=min_periods)
        v = np.sqrt(np.maximum(np.diag(Sigma), self.eps))
        return np.log(v) if self.log else v

    def postprocess(self, y_hat: NDArray[np.floating]) -> NDArray[np.floating]:
        return y_hat
