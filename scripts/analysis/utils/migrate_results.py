from __future__ import annotations

import shutil
from pathlib import Path

from scripts.analysis.utils.paths import RESULTS_DIR, resolve_backtest_path, resolve_figs_dir, resolve_report_path


def _copy_if_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copytree_merge(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        if out.exists() and out.stat().st_size > 0:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def ensure_canonical_results(tag: str, *, target: str | None = None) -> None:
    """
    Ensure canonical layout exists:
      results/<tag>/backtest.{parquet,csv}
      results/<tag>/report.csv
      results/<tag>/figs/...

    Copies from legacy locations if needed (does not delete legacy).
    """
    tag = str(tag)
    tag_dir = RESULTS_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    # backtests
    src_bt = resolve_backtest_path(tag)
    _copy_if_missing(src_bt, tag_dir / ("backtest.parquet" if src_bt.suffix == ".parquet" else "backtest.csv"))
    # If only parquet exists, also try to copy csv legacy
    if (tag_dir / "backtest.csv").exists() is False:
        src_csv = RESULTS_DIR / f"{tag}_backtest.csv"
        if tag == "regime_covariance":
            src_csv = RESULTS_DIR / "regime_similarity_backtest.csv" if (RESULTS_DIR / "regime_similarity_backtest.csv").exists() else src_csv
        _copy_if_missing(src_csv, tag_dir / "backtest.csv")
    if (tag_dir / "backtest.parquet").exists() is False:
        src_pq = RESULTS_DIR / f"{tag}_backtest.parquet"
        if tag == "regime_covariance":
            src_pq = RESULTS_DIR / "regime_similarity_backtest.parquet" if (RESULTS_DIR / "regime_similarity_backtest.parquet").exists() else src_pq
        _copy_if_missing(src_pq, tag_dir / "backtest.parquet")

    # report
    src_report = resolve_report_path(tag, target=target)
    _copy_if_missing(src_report, tag_dir / "report.csv")

    # figs: prefer copying from legacy figs_* if canonical figs is missing/empty
    dst_figs = tag_dir / "figs"
    dst_figs.mkdir(parents=True, exist_ok=True)
    dst_has_files = any(p.is_file() for p in dst_figs.rglob("*"))

    legacy_figs = RESULTS_DIR / f"figs_{tag}"
    if tag == "regime_covariance":
        legacy_figs = RESULTS_DIR / "figs_regime_similarity" if (RESULTS_DIR / "figs_regime_similarity").exists() else legacy_figs
    if tag == "regime_volatility" and (RESULTS_DIR / "figs_regime_volatility").exists():
        legacy_figs = RESULTS_DIR / "figs_regime_volatility"

    # If dst is empty, merge from legacy
    if not dst_has_files:
        _copytree_merge(legacy_figs, dst_figs)

