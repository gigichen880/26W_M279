"""Split-aware walk-forward ExperimentRunner.

Frozen after 2016: architecture / HPs / selection rule / C_η / protocol.
Not frozen: model parameters — expanding refit through OOS.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from similarity_forecast.core import validate_window
from similarity_forecast.redesign.baselines import BASELINE_FNS, persistence_cov
from similarity_forecast.redesign.conditioner import ConditioningConfig
from similarity_forecast.redesign.embeddings import build_embedder
from similarity_forecast.redesign.metrics import evaluate_forecast
from similarity_forecast.redesign.regimes import build_regime_model
from similarity_forecast.redesign.similarity import (
    SimilarityForecaster,
    SimilarityLibrary,
    build_covariance_target,
)


@dataclass
class ProtocolConfig:
    lookback: int = 50
    horizon: int = 20
    neighbor_gap: int = 10
    stride: int = 5
    refit_every_days: int = 20
    burn_in: int = 252
    devel_end: str = "2012-12-31"
    select_end: str = "2016-12-31"
    test_start: str = "2017-01-01"
    random_state: int = 0


@dataclass
class ExperimentConfig:
    tag: str = "redesign_run"
    parquet_path: str = "data/processed/returns_universe_100.parquet"
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    embedder_name: str = "market_state"
    pca_k: int = 8
    k_neighbors: int = 20
    metric: str = "euclidean"
    aggregation: str = "log_euclidean"
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    baselines: tuple = ("rolling", "ewma", "ledoit_wolf", "oas", "shrink")
    # evaluation window: "select" | "test" | "devel" | "all_post_burnin"
    eval_split: str = "select"
    outdir: str = "results/redesign"
    # Regime integration: none | overlap (D2 current-state p) | overlap_pred (D5 predictive occupancy)
    regime_mode: str = "none"
    regime_name: str = "fcm"
    regime_n_states: int = 3
    method_label: str = "similarity"  # column name for the proposed forecast


def _split_mask(dates: pd.DatetimeIndex, proto: ProtocolConfig, split: str) -> np.ndarray:
    d = pd.to_datetime(dates)
    if split == "devel":
        return (d >= "2008-01-01") & (d <= proto.devel_end)
    if split == "select":
        return (d > proto.devel_end) & (d <= proto.select_end)
    if split == "test":
        return d >= proto.test_start
    if split == "all_post_burnin":
        return d >= "2008-01-01"
    raise ValueError(split)


def load_returns(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.astype(float)


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    proto = cfg.protocol
    df = load_returns(cfg.parquet_path)
    R = df.to_numpy(dtype=float)
    dates = df.index
    T, N = R.shape
    L, H, g = proto.lookback, proto.horizon, proto.neighbor_gap

    # Candidate anchors (raw indices)
    anchors = list(range(L, T - H, proto.stride))
    # burn-in by trading days from start
    if proto.burn_in > 0 and anchors:
        min_a = anchors[0] + proto.burn_in
        anchors = [a for a in anchors if a >= min_a]

    eval_mask = _split_mask(dates, proto, cfg.eval_split)
    eval_anchors = [a for a in anchors if bool(eval_mask[a])]

    forecaster = SimilarityForecaster(
        metric=cfg.metric,  # type: ignore
        k_neighbors=cfg.k_neighbors,
        aggregation=cfg.aggregation,  # type: ignore
    )
    cond = cfg.conditioning

    rows = []
    embedder = None
    library: Optional[SimilarityLibrary] = None
    regime_model = None
    last_refit_date = None

    def need_refit(a: int) -> bool:
        nonlocal last_refit_date
        if embedder is None or library is None:
            return True
        d = dates[a]
        if last_refit_date is None:
            return True
        return (d - last_refit_date).days >= proto.refit_every_days

    def refit(a: int) -> None:
        nonlocal embedder, library, regime_model, last_refit_date
        train = R[: a + 1]
        emb = build_embedder(cfg.embedder_name, lookback=L, pca_k=cfg.pca_k)
        if hasattr(emb, "fit"):
            try:
                emb.fit(train, lookback=L)  # type: ignore
            except TypeError:
                emb.fit(train)  # type: ignore
        # Daily-ish library (stride) with predictive regime features when requested
        lib_anchors, embeds, targets = [], [], []
        for t in range(L, a - H - g + 1, max(1, proto.stride // 2) or 1):
            past = train[t - L : t]
            fut = train[t + 1 : t + H + 1]
            if past.shape[0] < L or fut.shape[0] < H:
                continue
            if not validate_window(past) or not validate_window(fut):
                continue
            try:
                z = emb.embed(past)
            except Exception:
                continue
            lib_anchors.append(t)
            embeds.append(z)
            targets.append(build_covariance_target(fut))
        if not lib_anchors:
            raise RuntimeError(f"Empty library at anchor {a}")
        Z_lib = np.vstack(embeds)
        regime_p = None
        regime_model = None
        if cfg.regime_mode in ("overlap", "overlap_pred"):
            regime_model = build_regime_model(
                cfg.regime_name, n_states=cfg.regime_n_states, random_state=proto.random_state
            )
            regime_model.fit(Z_lib)
            P = regime_model.predict_proba(Z_lib)
            if cfg.regime_mode == "overlap_pred" and hasattr(regime_model, "predictive_occupancy"):
                # Predictive-information symmetry: occupancy from each anchor's own filtration
                regime_p = np.vstack(
                    [regime_model.predictive_occupancy(P[i], H) for i in range(P.shape[0])]
                )
            else:
                regime_p = P
        embedder = emb
        library = SimilarityLibrary(
            anchors=np.asarray(lib_anchors, dtype=np.int64),
            embeds=Z_lib,
            targets=targets,
            regime_p=regime_p,
        )
        last_refit_date = dates[a]

    for a in eval_anchors:
        if need_refit(a):
            try:
                refit(a)
            except Exception:
                continue
        assert embedder is not None and library is not None
        past = R[a - L : a]
        fut = R[a + 1 : a + H + 1]
        if not validate_window(past) or not validate_window(fut):
            continue
        try:
            z = embedder.embed(past)
            p_query = None
            if cfg.regime_mode != "none" and regime_model is not None:
                p0 = regime_model.predict_proba(z.reshape(1, -1))[0]
                if cfg.regime_mode == "overlap_pred" and hasattr(regime_model, "predictive_occupancy"):
                    p_query = regime_model.predictive_occupancy(p0, H)
                else:
                    p_query = p0
            S_hat, _ = forecaster.predict(
                z,
                library,
                raw_anchor=a,
                horizon=H,
                neighbor_gap=g,
                regime_p_query=p_query,
                regime_mode=cfg.regime_mode if cfg.regime_mode != "none" else "none",  # type: ignore
            )
        except Exception:
            continue
        S_real = build_covariance_target(fut)
        prev_fut = R[a - H + 1 : a + 1] if a - H + 1 >= 0 else None

        methods = {cfg.method_label: S_hat}
        for bname in cfg.baselines:
            fn = BASELINE_FNS.get(bname)
            if fn is None:
                continue
            try:
                methods[bname] = fn(past)
            except Exception:
                continue
        if prev_fut is not None and prev_fut.shape[0] == H:
            try:
                methods["persistence"] = persistence_cov(build_covariance_target(prev_fut))
            except Exception:
                pass

        for mname, Sh in methods.items():
            ev = evaluate_forecast(Sh, S_real, fut, cond)
            rows.append(
                {
                    "date": dates[a],
                    "raw_anchor": a,
                    "method": mname,
                    "split": cfg.eval_split,
                    "embedder": cfg.embedder_name,
                    "pca_k": cfg.pca_k,
                    "k_neighbors": cfg.k_neighbors,
                    "aggregation": cfg.aggregation,
                    "eta": cond.eta,
                    "raw_frob": ev.raw_frob,
                    "raw_logeuc": ev.raw_logeuc,
                    "raw_corr_frob": ev.raw_corr_frob,
                    "cond_stein": ev.cond_stein,
                    "cond_kl": ev.cond_kl,
                    "nll": ev.nll,
                    "gmvp_var": ev.gmvp_var,
                    "var_regret": ev.var_regret,
                    "pred_var": ev.pred_var,
                    "leverage_l1": ev.leverage_l1,
                    "max_abs_w": ev.max_abs_w,
                    "hhi": ev.hhi,
                }
            )

    out = pd.DataFrame(rows)
    outdir = Path(cfg.outdir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "panel.csv", index=False)
    if len(out):
        summary = (
            out.groupby("method")[
                ["gmvp_var", "var_regret", "cond_stein", "cond_kl", "raw_frob", "nll", "leverage_l1"]
            ]
            .mean()
            .sort_values("var_regret")
        )
        summary.to_csv(outdir / "summary.csv")
        (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return out
