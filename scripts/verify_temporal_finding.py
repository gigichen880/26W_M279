"""
Verify temporal win-rate finding for Model vs Mix (GMVP Sharpe).

This script is intentionally dependency-light (stdlib only) so it can run in
minimal environments. It loads a per-date backtest CSV and computes:
  - win rate in fixed periods
  - rolling 63-day win rate (fraction of days where Model Sharpe > Mix Sharpe)

Expected per-date inputs (any one of the supported formats):
  1) Wide, backtest-like:
       - `date`
       - `model_gmvp_sharpe`
       - `mix_gmvp_sharpe`
  2) Long:
       - `date`
       - method column containing "model" and "mix"
       - gmvp/sharpe column (e.g. contains "gmvp" and "sharpe")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class SharpeRow:
    d: date
    model: Optional[float]
    mix: Optional[float]

    @property
    def win(self) -> Optional[bool]:
        if self.model is None or self.mix is None:
            return None
        return self.model > self.mix


def _parse_date(s: str) -> date:
    s = s.strip()
    # Most inputs are YYYY-MM-DD
    return date.fromisoformat(s)


def _parse_float(s: str) -> Optional[float]:
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_colname(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _find_first_matching_columns(header: list[str], wanted_substrings: Iterable[str]) -> list[str]:
    norm_header = [_normalize_colname(c) for c in header]
    wanted = [w.lower() for w in wanted_substrings]
    matches: list[str] = []
    for orig, norm in zip(header, norm_header):
        ok = all(w in norm for w in wanted)
        if ok:
            matches.append(orig)
    return matches


def _load_backtest_like_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[SharpeRow]]:
    """
    Load a per-date CSV where Model and Mix GMVP Sharpe are in distinct columns.
    """
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        header = list(reader.fieldnames)

        # Print: "column names and first 5 rows" (requirement #2)
        first_rows: list[dict[str, str]] = []
        for _ in range(5):
            try:
                row = next(reader)
            except StopIteration:
                break
            first_rows.append(dict(row))

        # Reset read for full parse
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])

        # Required columns
        date_cols = [c for c in header if _normalize_colname(c) == "date" or _normalize_colname(c).endswith("_date")]
        if not date_cols:
            raise ValueError(f"Missing 'date' column in {path}")
        date_col = date_cols[0]

        # Typical in this repo (observed in results/regime_covariance/backtest.csv)
        model_candidates = [c for c in header if _normalize_colname(c).startswith("model_") and "gmvp" in _normalize_colname(c) and "sharpe" in _normalize_colname(c)]
        mix_candidates = [c for c in header if _normalize_colname(c).startswith("mix_") and "gmvp" in _normalize_colname(c) and "sharpe" in _normalize_colname(c)]

        if not model_candidates or not mix_candidates:
            # Less specific fallback: any column mentioning model/mix + gmvp + sharpe
            model_candidates = _find_first_matching_columns(header, ["model", "gmvp", "sharpe"])
            mix_candidates = _find_first_matching_columns(header, ["mix", "gmvp", "sharpe"])

        if not model_candidates or not mix_candidates:
            raise ValueError(f"Could not find model/mix GMVP Sharpe columns in {path}")

        model_col = model_candidates[0]
        mix_col = mix_candidates[0]

        out: list[SharpeRow] = []
        for row in reader:
            if not row:
                continue
            d_raw = row.get(date_col, "")
            if d_raw is None or str(d_raw).strip() == "":
                continue
            try:
                d = _parse_date(str(d_raw))
            except ValueError:
                # Skip unparseable dates
                continue

            model_val = _parse_float(str(row.get(model_col, "") if row.get(model_col, None) is not None else ""))
            mix_val = _parse_float(str(row.get(mix_col, "") if row.get(mix_col, None) is not None else ""))

            out.append(SharpeRow(d=d, model=model_val, mix=mix_val))

    return header, first_rows, out


def _load_long_format_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[SharpeRow]]:
    """
    Load a per-date CSV where method/model is in one column and GMVP Sharpe in another.
    """
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        header = list(reader.fieldnames)

        first_rows: list[dict[str, str]] = []
        for _ in range(5):
            try:
                row = next(reader)
            except StopIteration:
                break
            first_rows.append(dict(row))

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])

        date_cols = [c for c in header if _normalize_colname(c) == "date" or _normalize_colname(c).endswith("_date")]
        if not date_cols:
            raise ValueError(f"Missing 'date' column in {path}")
        date_col = date_cols[0]

        method_cols = [c for c in header if "method" in _normalize_colname(c) or _normalize_colname(c) in {"model", "strategy"}]
        if not method_cols:
            raise ValueError(f"Missing method column in {path}")
        method_col = method_cols[0]

        sharpe_cols = [c for c in header if "gmvp" in _normalize_colname(c) and "sharpe" in _normalize_colname(c)]
        if not sharpe_cols:
            # fallback: contains "sharpe" and "gmvp" anywhere
            sharpe_cols = _find_first_matching_columns(header, ["gmvp", "sharpe"])
        if not sharpe_cols:
            raise ValueError(f"Missing GMVP Sharpe column in {path}")
        sharpe_col = sharpe_cols[0]

        rows_by_date: dict[date, dict[str, Optional[float]]] = {}

        for row in reader:
            if not row:
                continue
            d_raw = row.get(date_col, "")
            if d_raw is None or str(d_raw).strip() == "":
                continue
            try:
                d = _parse_date(str(d_raw))
            except ValueError:
                continue

            method = str(row.get(method_col, "")).strip().lower()
            sharpe = _parse_float(str(row.get(sharpe_col, "") if row.get(sharpe_col, None) is not None else ""))

            if "model" in method:
                rows_by_date.setdefault(d, {})
                rows_by_date[d]["model"] = sharpe
            elif "mix" in method:
                rows_by_date.setdefault(d, {})
                rows_by_date[d]["mix"] = sharpe

        out: list[SharpeRow] = []
        for d, vals in rows_by_date.items():
            out.append(SharpeRow(d=d, model=vals.get("model"), mix=vals.get("mix")))

    return header, first_rows, out


def try_load_per_date_results() -> tuple[Path, list[str], list[dict[str, str]], list[SharpeRow]]:
    """
    Try multiple likely files and supported formats.
    """
    candidates: list[Path] = []

    # Preferred "backtest-like" sources (observed in this project)
    candidates.extend(
        [
            RESULTS_DIR / "regime_covariance" / "backtest.csv",
            RESULTS_DIR / "regime_volatility" / "backtest.csv",
        ]
    )

    # Additional "comprehensive" files (may or may not be per-date)
    candidates.extend(
        [
            RESULTS_DIR / "comprehensive_baseline_comparison.csv",
            RESULTS_DIR / "regime_similarity" / "comprehensive_baseline_comparison.csv",
            RESULTS_DIR / "regime_covariance" / "comprehensive_baseline_comparison.csv",
        ]
    )

    last_err: Optional[Exception] = None
    for path in candidates:
        if not path.exists():
            continue
        for loader in (_load_backtest_like_csv, _load_long_format_csv):
            try:
                header, first_rows, rows = loader(path)
                return path, header, first_rows, rows
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not load per-date results from candidates. Last error: {last_err!r}")


def _compute_period_win_rate(rows: list[SharpeRow], start: date, end: date) -> tuple[int, int, float]:
    filt = [r for r in rows if start <= r.d <= end and r.win is not None]
    total = len(filt)
    wins = sum(1 for r in filt if r.win is True)
    win_rate = (wins / total) if total > 0 else float("nan")
    return wins, total, win_rate


def _compute_rolling_63d_mean(rows: list[SharpeRow], cutoff: date, before: bool) -> float:
    """
    Compute mean rolling win rate for dates < cutoff (before=True) or >= cutoff.

    Rolling win rate at date t is:
        (# of days in last 63 calendar days where model>mix) / (total # of days with non-missing sharpe in that window)
    """
    # Sort by date for windowing
    rows_sorted = sorted(rows, key=lambda r: r.d)

    # Sliding window over dates
    left = 0
    true_in_window = 0
    total_in_window = 0

    win_rates_for_mean: list[float] = []
    day_window = timedelta(days=62)  # inclusive => last 63 days total

    for right, r in enumerate(rows_sorted):
        # Add current point
        if r.win is not None:
            total_in_window += 1
            if r.win:
                true_in_window += 1

        # Move left pointer to keep window within [t-62, t]
        while left <= right and rows_sorted[left].d < (r.d - day_window):
            wl = rows_sorted[left]
            if wl.win is not None:
                total_in_window -= 1
                if wl.win:
                    true_in_window -= 1
            left += 1

        # Rolling value for this date
        if r.d >= cutoff:
            cond = not before
        else:
            cond = before

        if cond and total_in_window > 0:
            win_rates_for_mean.append(true_in_window / total_in_window)

    if not win_rates_for_mean:
        return float("nan")
    return sum(win_rates_for_mean) / len(win_rates_for_mean)


def _pct(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x * 100:.2f}%"


def main() -> None:
    loaded_path: Optional[Path] = None
    header: list[str] = []
    first_rows: list[dict[str, str]] = []
    rows: list[SharpeRow] = []

    loaded_path, header, first_rows, rows = try_load_per_date_results()

    print(f"LOADED FILE: {loaded_path}")
    print("COLUMN NAMES:")
    print(", ".join(header))
    print("FIRST 5 ROWS (raw):")
    for i, r in enumerate(first_rows[:5]):
        # Print only key fields if present; else print all.
        # Keep it simple to make it easy to visually validate format.
        date_val = r.get("date", r.get("Date", next(iter(r.values()), "")))
        model_key = next((k for k in r.keys() if _normalize_colname(k).startswith("model_") and "gmvp" in _normalize_colname(k) and "sharpe" in _normalize_colname(k)), None)
        mix_key = next((k for k in r.keys() if _normalize_colname(k).startswith("mix_") and "gmvp" in _normalize_colname(k) and "sharpe" in _normalize_colname(k)), None)
        if model_key and mix_key:
            print(f"  Row {i+1}: date={date_val}, {model_key}={r.get(model_key)}, {mix_key}={r.get(mix_key)}")
        else:
            print(f"  Row {i+1}: {r}")

    # Ensure we're sorted and within expected span
    if not rows:
        raise RuntimeError("No per-date rows were parsed.")

    # Periods
    p1_start = date(2013, 1, 1)
    p1_end = date(2015, 12, 31)
    p2_start = date(2016, 1, 1)
    p2_end = date(2021, 12, 31)
    overall_start = date(2013, 1, 1)
    overall_end = date(2021, 12, 31)

    w1, t1, r1 = _compute_period_win_rate(rows, p1_start, p1_end)
    w2, t2, r2 = _compute_period_win_rate(rows, p2_start, p2_end)
    w3, t3, r3 = _compute_period_win_rate(rows, overall_start, overall_end)

    print("\nWIN RATE (Model GMVP Sharpe > Mix GMVP Sharpe):")
    print(f"  Period 2013-01-01 to 2015-12-31: {w1}/{t1} wins, win rate={_pct(r1)}")
    print(f"  Period 2016-01-01 to 2021-12-31: {w2}/{t2} wins, win rate={_pct(r2)}")
    print(f"  Overall 2013-01-01 to 2021-12-31: {w3}/{t3} wins, win rate={_pct(r3)}")

    # Rolling 63-day win rate mean (rolling fraction)
    cutoff_2016 = date(2016, 1, 1)
    mean_before_2016 = _compute_rolling_63d_mean(rows, cutoff=cutoff_2016, before=True)
    mean_from_2016 = _compute_rolling_63d_mean(rows, cutoff=cutoff_2016, before=False)

    print("\nROLLING 63-DAY WIN RATE (mean of rolling fractions):")
    print(f"  Mean win rate before 2016-01-01: {_pct(mean_before_2016)}")
    print(f"  Mean win rate from 2016-01-01 onwards: {_pct(mean_from_2016)}")

    # Simple text-based summary (requirement #5)
    claimed_p1 = 0.30
    claimed_p2 = 0.65

    # Tolerance in percentage points since finding is "approx"
    tol = 0.10
    p1_supported = (r1 == r1) and abs(r1 - claimed_p1) <= tol
    p2_supported = (r2 == r2) and abs(r2 - claimed_p2) <= tol

    if p1_supported and p2_supported:
        verdict = "SUPPORTED"
    elif p1_supported or p2_supported:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    print("\nFINDING VERIFICATION:")
    print("Model vs Mix (GMVP Sharpe) win rate:")
    print(f"2013-2015: {_pct(r1)} (claimed: ~30%)")
    print(f"2016-2021: {_pct(r2)} (claimed: ~65%)")
    print(f"Finding is {verdict}")


if __name__ == "__main__":
    main()

