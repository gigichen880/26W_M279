from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def resolve_backtest_path(tag: str) -> Path:
    """
    Resolve backtest file path for a given tag.

    Supports both layouts:
      - New (preferred): results/<tag>/backtest.parquet (or .csv)
      - Legacy:          results/<tag>_backtest.parquet (or .csv)
    """
    tag = str(tag)
    candidates = [
        RESULTS_DIR / tag / "backtest.parquet",
        RESULTS_DIR / tag / "backtest.csv",
        RESULTS_DIR / f"{tag}_backtest.parquet",
        RESULTS_DIR / f"{tag}_backtest.csv",
    ]
    # Legacy alias: historical "regime_similarity_*" corresponds to covariance tag
    if tag == "regime_covariance":
        candidates.extend(
            [
                RESULTS_DIR / "regime_similarity_backtest.parquet",
                RESULTS_DIR / "regime_similarity_backtest.csv",
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    # default to preferred path (even if missing) for clearer errors upstream
    return candidates[0]


def resolve_figs_dir(tag: str) -> Path:
    """
    Canonical figure root for a tag.

    Preferred: results/<tag>/figs/
    Legacy:    results/figs_<tag>/
    """
    tag = str(tag)
    preferred = RESULTS_DIR / tag / "figs"
    legacy = RESULTS_DIR / f"figs_{tag}"
    # Legacy alias: historical "figs_regime_similarity" corresponds to covariance tag
    if tag == "regime_covariance":
        legacy_alias = RESULTS_DIR / "figs_regime_similarity"
        if legacy_alias.exists() and not legacy.exists():
            legacy = legacy_alias
    return preferred if preferred.exists() else legacy


def resolve_report_path(tag: str, *, target: str | None = None) -> Path:
    """
    Resolve report path.

    New:    results/<tag>/report.csv
    Legacy: results/<tag>_report.csv (except volatility historically used regime_volatility_report.csv)
    """
    tag = str(tag)
    preferred = RESULTS_DIR / tag / "report.csv"
    if preferred.exists():
        return preferred
    # legacy special-case
    if target == "volatility":
        legacy = RESULTS_DIR / "regime_volatility_report.csv"
        if legacy.exists():
            return legacy
    legacy = RESULTS_DIR / f"{tag}_report.csv"
    if tag == "regime_covariance":
        legacy_alias = RESULTS_DIR / "regime_similarity_report.csv"
        if legacy_alias.exists() and not legacy.exists():
            legacy = legacy_alias
    return legacy

