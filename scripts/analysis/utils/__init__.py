"""Shared helpers for analysis scripts (paths, IO, etc.)."""

from scripts.analysis.utils.paths import (
    REPO_ROOT,
    RESULTS_DIR,
    resolve_backtest_path,
    resolve_figs_dir,
    resolve_report_path,
)

__all__ = [
    "REPO_ROOT",
    "RESULTS_DIR",
    "resolve_backtest_path",
    "resolve_figs_dir",
    "resolve_report_path",
]
