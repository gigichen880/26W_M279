#!/usr/bin/env python3
"""
Extract daily close-to-close returns from minutely .7z archives, validate quality,
merge with pvCLCL where appropriate, and produce final universe for covariance forecasting.
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing
import os
import shutil
import tempfile
import zipfile
from datetime import date as date_type, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    HAS_PY7ZR = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_data_by_stocks(given: Path | str | None = None) -> Path:
    """Data dir: given path, env DATA_BY_STOCKS, data/raw/data_by_stocks, or ~/Downloads/data_by_stocks."""
    if given:
        p = Path(given).expanduser().resolve()
        if p.exists():
            return p
    if os.environ.get("DATA_BY_STOCKS"):
        p = Path(os.environ["DATA_BY_STOCKS"]).expanduser().resolve()
        if p.exists():
            return p
    for candidate in [REPO_ROOT / "data" / "raw" / "data_by_stocks", Path.home() / "Downloads" / "data_by_stocks"]:
        if candidate.exists():
            return candidate.resolve()
    return Path(given or os.environ.get("DATA_BY_STOCKS") or str(REPO_ROOT / "data" / "raw" / "data_by_stocks")).expanduser().resolve()
OUT_DIR = REPO_ROOT / "results" / "eda" / "reports"
PARQUET_ZIP = REPO_ROOT / "data" / "raw" / "pvCLCL_matrix.parquet.zip"
CSV_ZIP = REPO_ROOT / "data" / "raw" / "pvCLCL_matrix.csv.zip"
RECENT_START = 2015
RECENT_END = 2021
MEGA_CAPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "FB", "TSLA", "NFLX", "JPM", "BAC", "JNJ", "PFE", "WFC"]
TRADING_DAYS_PER_YEAR = 252

# -----------------------------------------------------------------------------
# Helpers: close column detection, load pvCLCL
# -----------------------------------------------------------------------------
def _date_from_path(extracted: Path, tmpdir: Path) -> date_type | None:
    """Try to get a trading date from file path (stem, parent name, relative path parts). Only return if in [2000, today]."""
    today = date_type.today()
    lo, hi = date_type(2000, 1, 1), today
    candidates = []
    candidates.append(extracted.stem.split("_")[-1])
    candidates.extend(extracted.stem.split("_"))
    try:
        candidates.append(extracted.parent.name)
        rel = extracted.relative_to(tmpdir)
        candidates.append(rel.parent.name)
        candidates.extend(rel.parts)
    except Exception:
        pass
    for s in candidates:
        if not s or len(s) < 8:
            continue
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                continue
            d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
            if lo <= d <= hi:
                return d
        except Exception:
            continue
    return None


def _date_from_timestamp(val: Any) -> date_type | None:
    """Parse numeric timestamp (s/ms/us/ns) or string; return date in [2000, today] or None."""
    try:
        if pd.isna(val):
            return None
        today = date_type.today()
        if isinstance(val, (int, float)):
            v = float(val)
            for unit in ("s", "ms", "us", "ns"):
                try:
                    dt = pd.to_datetime(v, unit=unit, errors="coerce")
                    if pd.isna(dt):
                        continue
                    d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
                    if date_type(2000, 1, 1) <= d <= today:
                        return d
                except Exception:
                    continue
            return None
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
        return d if date_type(2000, 1, 1) <= d <= today else None
    except Exception:
        return None


def _find_close_column(df: pd.DataFrame) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for name in ["close", "last", "cl", "adj close", "adj_close"]:
        if name in cols_lower:
            return cols_lower[name]
    for c in df.columns:
        if "close" in str(c).lower() or "last" in str(c).lower():
            return c
    return None


def _close_from_bid_ask(row: pd.Series, columns: pd.Index) -> float | None:
    """When there is no 'close' column, use mid (bid+ask)/2 from quote columns (e.g. bid_1, ask_1)."""
    cols_stripped = {str(c).strip().lower(): c for c in columns}
    ask_col = cols_stripped.get("ask_1") or next((c for c in columns if "ask" in str(c).lower() and "size" not in str(c).lower()), None)
    bid_col = cols_stripped.get("bid_1") or next((c for c in columns if "bid" in str(c).lower() and "size" not in str(c).lower()), None)
    if ask_col is None or bid_col is None:
        return None
    try:
        a, b = float(row[ask_col]), float(row[bid_col])
        if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
            return None
        return (a + b) / 2.0
    except (TypeError, ValueError):
        return None


def _find_timestamp_column(df: pd.DataFrame) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for name in ["timestamp", "date", "datetime", "time", "t"]:
        if name in cols_lower:
            return cols_lower[name]
    if df.index.name and "date" in str(df.index.name).lower():
        return None
    return df.columns[0] if len(df.columns) else None


def load_pvclcl_matrix() -> pd.DataFrame:
    for path, is_parquet in [(PARQUET_ZIP, True), (CSV_ZIP, False)]:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic[:2] == b"PK":
                with zipfile.ZipFile(path, "r") as z:
                    names = [n for n in z.namelist() if n.endswith(".parquet" if is_parquet else ".csv")]
                    if not names:
                        continue
                    with z.open(names[0]) as f:
                        data = f.read()
                if is_parquet:
                    df = pd.read_parquet(BytesIO(data))
                else:
                    df = pd.read_csv(BytesIO(data), index_col=0)
            elif magic[:2] == b"\x1f\x8b" and not is_parquet:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    df = pd.read_csv(f, index_col=0)
            else:
                continue
            df.index = df.index.astype(str).str.strip()
            df.columns = pd.to_datetime(df.columns, errors="coerce")
            return df
        except Exception as e:
            print(f"  Load {path} failed: {e}")
    raise FileNotFoundError("No pvCLCL matrix found.")


# -----------------------------------------------------------------------------
# STEP 1: Inventory and inspection
# -----------------------------------------------------------------------------
def step1_inventory_and_inspection(data_dir: Path) -> list[Path]:
    print("\n" + "=" * 60)
    print("STEP 1: Inventory and Inspection")
    print("=" * 60)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data_by_stocks/ not found: {data_dir}\n"
            "Set DATA_BY_STOCKS env or run: python minutely_extraction_pipeline.py --data-dir /path/to/data_by_stocks"
        )
    files = sorted(data_dir.glob("*.7z"))
    print(f"Count of .7z files: {len(files)}")
    print("First 20 filenames:")
    for p in files[:20]:
        print(f"  {p.name}")
    samples = ["AAPL", "MSFT", "JPM"]
    for ticker in samples:
        cand = [f for f in files if ticker.upper() in f.name.upper() and f.name.upper().startswith(ticker.upper()[:2])]
        if not cand:
            cand = [f for f in files if ticker.upper() in f.name.upper()]
        if not cand:
            print(f"\n  Sample {ticker}: no matching .7z found, skipping.")
            continue
        path = cand[0]
        print(f"\n  Sample {ticker}: {path.name}")
        if not HAS_PY7ZR:
            print("  py7zr not installed; cannot extract. pip install py7zr")
            continue
        try:
            with py7zr.SevenZipFile(path, "r") as z:
                names = z.getnames()
                csvs = [n for n in names if str(n).lower().endswith(".csv")]
                print(f"  Daily CSV files inside: {len(csvs)} (first 5: {csvs[:5]})")
                if csvs:
                    tmpdir = tempfile.mkdtemp()
                    try:
                        z.extract(targets=[csvs[0]], path=tmpdir)
                        extracted = Path(tmpdir) / csvs[0]
                        if not extracted.exists():
                            extracted = list(Path(tmpdir).rglob("*.csv"))[0] if list(Path(tmpdir).rglob("*.csv")) else None
                        if extracted and extracted.exists():
                            sample_df = pd.read_csv(extracted)
                            if not sample_df.empty:
                                print("  Columns:", list(sample_df.columns))
                                print("  First 5 rows:")
                                print(sample_df.head().to_string())
                                print("  dtypes:", sample_df.dtypes.to_string())
                                close_col = _find_close_column(sample_df)
                                ts_col = _find_timestamp_column(sample_df)
                                print(f"  Identified: timestamp~{ts_col}, close~{close_col}")
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            print(f"  Error: {e}")
    return files


# -----------------------------------------------------------------------------
# STEP 2: Extract daily close prices from all .7z
# -----------------------------------------------------------------------------
def _extract_daily_closes_one_stock(path: Path, ticker: str) -> pd.Series | None:
    """Return Series index=date, value=close for last minute of each day."""
    if not HAS_PY7ZR:
        return None
    try:
        with py7zr.SevenZipFile(path, "r") as z:
            names = z.getnames()
            csvs = [n for n in names if n.lower().endswith(".csv")]
    except Exception:
        return None
    dates_closes = []
    for csv_name in csvs:
        try:
            with py7zr.SevenZipFile(path, "r") as z:
                cf = z.getmember(csv_name) if hasattr(z, 'getmember') else None
                if cf is None:
                    all_bytes = z.readall()
                    for k, v in (all_bytes or {}).items():
                        if k == csv_name or csv_name in str(k):
                            df = pd.read_csv(BytesIO(v) if hasattr(v, 'read') else BytesIO(v))
                            break
                    else:
                        continue
                else:
                    df = pd.read_csv(BytesIO(z.read(csv_name)) if hasattr(z, 'read') else BytesIO(z.read([csv_name])[csv_name]))
            if df is None or df.empty:
                continue
            close_col = _find_close_column(df)
            if close_col is None:
                continue
            ts_col = _find_timestamp_column(df)
            if ts_col:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.dropna(subset=[ts_col])
            if df.empty:
                continue
            last_row = df.iloc[-1]
            close_val = last_row[close_col]
            if pd.isna(close_val) or close_val <= 0:
                continue
            if ts_col:
                dt = df.iloc[-1][ts_col]
            else:
                dt = df.index[-1]
            if hasattr(dt, "date"):
                d = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
            else:
                d = pd.Timestamp(dt).date()
            dates_closes.append((d, float(close_val)))
        except Exception:
            continue
    if not dates_closes:
        return None
    s = pd.Series({d: c for d, c in dates_closes})
    s.name = ticker
    return s


def step2_extract_daily_closes(files: list[Path]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: Extract Daily Close Prices")
    print("=" * 60)
    if not HAS_PY7ZR:
        raise RuntimeError("pip install py7zr required")
    all_series = []
    failed = []
    for i, path in enumerate(files):
        ticker = path.stem
        if "_" in ticker or "-" in ticker:
            ticker = ticker.replace("_", " ").split()[0] if ticker.replace("_", " ").split() else ticker
        s = _extract_daily_closes_one_stock(path, ticker)
        if s is not None and len(s) > 0:
            all_series.append(s)
        else:
            failed.append(path.name)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)} stocks...")
    if not all_series:
        raise ValueError("No daily closes extracted from any .7z file.")
    closes = pd.concat(all_series, axis=1, join="outer")
    closes = closes.reindex(sorted(closes.columns), axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "minutely_daily_closes.parquet"
    closes.to_parquet(out_path)
    print(f"Saved {out_path}  shape={closes.shape}  failed={len(failed)}")
    return closes


def _ticker_from_7z_path(path: Path) -> str:
    """e.g. _data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60.7z -> AAPL."""
    stem = path.stem
    if "__" in stem:
        return stem.split("__")[-1].split("_")[0].strip()
    for part in stem.split("_"):
        if part.isalpha() and 2 <= len(part) <= 5:
            return part
    return stem.split("_")[0].split("-")[0].strip()


def _get_closes_one_7z(path: Path) -> tuple[str, pd.Series] | None:
    """Extract (ticker, Series of date->close). Last row of each CSV = close."""
    if not HAS_PY7ZR:
        return None
    ticker = _ticker_from_7z_path(path)
    tmpdir = tempfile.mkdtemp()
    try:
        with py7zr.SevenZipFile(path, "r") as z:
            names = z.getnames()
            csvs = [n for n in names if str(n).lower().endswith(".csv")]
            if not csvs:
                return None
            z.extract(targets=csvs, path=tmpdir)
        dates_closes = []
        for extracted in sorted(Path(tmpdir).rglob("*.csv")):
            try:
                df = pd.read_csv(extracted)
            except Exception:
                continue
            if df.empty:
                continue
            last = df.iloc[-1]
            close_col = _find_close_column(df)
            if close_col is not None:
                close_val = last[close_col]
            else:
                close_val = _close_from_bid_ask(last, df.columns)
            if close_val is None or (isinstance(close_val, (int, float)) and (pd.isna(close_val) or close_val <= 0)):
                continue
            if not isinstance(close_val, (int, float)):
                close_val = float(close_val)
            # Prefer date from path (filename/parent/relative path); then time column. Reject future dates.
            d = _date_from_path(extracted, Path(tmpdir))
            if d is None:
                for c in df.columns:
                    if "date" in str(c).lower() or "time" in str(c).lower():
                        d = _date_from_timestamp(last[c])
                        if d is not None:
                            break
            if d is not None and d > date_type.today():
                d = None
            if d is not None:
                dates_closes.append((d, float(close_val)))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not dates_closes:
        return None
    return (ticker, pd.Series(dict(dates_closes)))


def _worker_extract(path_str: str):
    """Worker: returns (path_str, (ticker, Series)) or (path_str, None)."""
    try:
        return (path_str, _get_closes_one_7z(Path(path_str)))
    except Exception:
        return (path_str, None)


def step2_extract_daily_closes_v2(files: list[Path], workers: int = 1) -> pd.DataFrame:
    """Extract daily close from each .7z; use workers>1 for parallel extraction."""
    print("\n" + "=" * 60)
    print("STEP 2: Extract Daily Close Prices")
    print("=" * 60)
    if not HAS_PY7ZR:
        raise RuntimeError("pip install py7zr required")
    all_series = []
    failed = []
    path_strs = [str(p.resolve()) for p in files]
    if workers < 2:
        for i, path in enumerate(files):
            res = _get_closes_one_7z(path)
            if res:
                ticker, s = res
                s.name = ticker
                all_series.append(s)
            else:
                failed.append(path.name)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(files)} stocks...")
    else:
        print(f"Using {workers} workers...")
        with multiprocessing.Pool(workers) as pool:
            for i, (pstr, res) in enumerate(pool.imap_unordered(_worker_extract, path_strs, chunksize=1)):
                if res:
                    ticker, s = res
                    s.name = ticker
                    all_series.append(s)
                else:
                    failed.append(Path(pstr).name)
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(files)} stocks...")
    if not all_series:
        raise ValueError("No daily closes extracted.")
    closes = pd.concat(all_series, axis=1, join="outer")
    closes = closes.reindex(sorted(closes.columns), axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "minutely_daily_closes.parquet"
    closes.to_parquet(out_path)
    print(f"Saved {out_path}  shape={closes.shape}  failed={len(failed)}")
    return closes


# -----------------------------------------------------------------------------
# STEP 3: Daily returns
# -----------------------------------------------------------------------------
def step3_compute_returns(closes: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3: Compute Daily Log Returns")
    print("=" * 60)
    closes = closes.sort_index(axis=0)
    ret = np.log(closes / closes.shift(1))
    out_path = OUT_DIR / "minutely_daily_returns.parquet"
    ret.to_parquet(out_path)
    print(f"Saved {out_path}  shape={ret.shape}")
    return ret


# -----------------------------------------------------------------------------
# STEP 4: Data quality report
# -----------------------------------------------------------------------------
def step4_quality_report(returns: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    print("\n" + "=" * 60)
    print("STEP 4: Data Quality Report")
    print("=" * 60)
    lines = []
    exclude = set()
    # Normalize index to DatetimeIndex so .loc and year access are consistent
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    dates = returns.index
    recent_mask = (dates.year >= RECENT_START) & (dates.year <= RECENT_END)
    total_days = len(dates)
    recent_days = recent_mask.sum()
    per_stock = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) == 0:
            per_stock.append({"ticker": ticker, "first": None, "last": None, "days": 0, "avail_pct": 0, "recent_avail": 0, "gap_30": True, "outlier": True})
            exclude.add(ticker)
            continue
        first = r.index.min()
        last = r.index.max()
        days = len(r)
        avail_pct = days / total_days * 100 if total_days else 0
        recent_avail = r.reindex(returns.index[recent_mask]).notna().sum() / recent_days * 100 if recent_days else 0
        # Gap > 30 consecutive trading days
        idx = pd.to_datetime(r.index).sort_values()
        diffs = idx.to_series().diff().dt.days
        gap_30 = (diffs > 30).any() if len(diffs) > 1 else False
        # Extreme return
        outlier = (r > 1.0).any() or (r < -0.9).any()
        per_stock.append({"ticker": ticker, "first": first, "last": last, "days": days, "avail_pct": avail_pct, "recent_avail": recent_avail, "gap_30": gap_30, "outlier": outlier})
        if avail_pct < 50:
            exclude.add(ticker)
        if recent_avail < 80:
            exclude.add(ticker)
        if gap_30:
            exclude.add(ticker)
        if outlier:
            exclude.add(ticker)
    qdf = pd.DataFrame(per_stock)
    report_path = OUT_DIR / "minutely_data_quality_report.txt"
    with open(report_path, "w") as f:
        f.write("MINUTELY DATA QUALITY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Stocks with <50% availability: {sum(qdf['avail_pct'] < 50)}\n")
        f.write(f"Stocks with <80% recent (2015-2021): {sum(qdf['recent_avail'] < 80)}\n")
        f.write(f"Stocks with gap >30 days: {sum(qdf['gap_30'])}\n")
        f.write(f"Stocks with extreme return: {sum(qdf['outlier'])}\n")
        f.write(f"EXCLUDE list (any of above): {len(exclude)} tickers\n")
        f.write("\nPer-stock summary (first 50):\n")
        f.write(qdf.head(50).to_string())
    print(f"Saved {report_path}  exclude={len(exclude)} tickers")
    return qdf, exclude


# -----------------------------------------------------------------------------
# STEP 5: Filter to top 100-150
# -----------------------------------------------------------------------------
def step5_filter_top_stocks(returns: pd.DataFrame, exclude: set[str], quality_df: pd.DataFrame) -> list[str]:
    print("\n" + "=" * 60)
    print("STEP 5: Filter to Top 100-150 Stocks")
    print("=" * 60)
    good = [t for t in returns.columns if t not in exclude]
    q = quality_df.set_index("ticker")
    good = [t for t in good if t in q.index and q.loc[t, "recent_avail"] >= 95 and q.loc[t, "avail_pct"] >= 70 and not q.loc[t, "outlier"]]
    mega_normalized = {m.upper().replace(".", ""): m for m in MEGA_CAPS}
    force_include = []
    for t in good:
        tn = t.upper().replace(".", "")
        if tn in mega_normalized or t in MEGA_CAPS or t.upper() in [m.upper() for m in MEGA_CAPS]:
            force_include.append(t)
    ranked = sorted(good, key=lambda t: q.loc[t, "recent_avail"], reverse=True)
    seen = set(force_include)
    out = list(force_include)
    for t in ranked:
        if t not in seen and len(out) < 150:
            out.append(t)
            seen.add(t)
    out = out[:150]
    print(f"Selected {len(out)} stocks (mega-caps force-included: {len(force_include)})")
    return out


# -----------------------------------------------------------------------------
# STEP 6: Compare with pvCLCL
# -----------------------------------------------------------------------------
def step6_compare_pvclcl(minutely_returns: pd.DataFrame, tickers_sample: list[str]) -> float | None:
    print("\n" + "=" * 60)
    print("STEP 6: Compare with pvCLCL Data")
    print("=" * 60)
    try:
        pv = load_pvclcl_matrix()
    except FileNotFoundError:
        print("pvCLCL matrix not found; skipping comparison.")
        return None
    pv.index = pv.index.astype(str).str.strip().str.upper()
    common = [t for t in tickers_sample if t.upper() in pv.index or t in pv.index]
    if not common:
        common = [t for t in pv.index if t in minutely_returns.columns]
    if not common:
        print("No overlapping ticker found for comparison.")
        return None
    sample_ticker = common[0]
    pv_ticker = sample_ticker if sample_ticker in pv.index else next((p for p in pv.index if p.upper() == sample_ticker.upper()), None)
    if pv_ticker is None:
        return None
    mr = minutely_returns[sample_ticker].dropna()
    pv_ser = pv.loc[pv_ticker]
    pv_ser = pv_ser.dropna()
    common_idx = mr.index.intersection(pv_ser.index)
    if len(common_idx) < 10:
        print(f"Overlap too small for {sample_ticker}")
        return None
    c = np.corrcoef(mr.reindex(common_idx).dropna(), pv_ser.reindex(common_idx).dropna())[0, 1]
    print(f"Correlation between minutely-derived and pvCLCL returns for {sample_ticker}: {c:.4f}")
    if c > 0.95:
        print("Data sources are compatible.")
    elif c < 0.90:
        print("Flag potential issue.")
    return float(c)


# -----------------------------------------------------------------------------
# STEP 7: Create final merged dataset
# -----------------------------------------------------------------------------
def step7_merge_final(minutely_returns: pd.DataFrame, selected_tickers: list[str]) -> pd.DataFrame:
    """Merge: prefer minutely for tickers we have; fill from pvCLCL where missing. Output: rows=tickers, columns=dates."""
    print("\n" + "=" * 60)
    print("STEP 7: Create Final Dataset (merge minutely + pvCLCL)")
    print("=" * 60)
    # minutely_returns: index=dates, columns=tickers
    mr = minutely_returns.reindex(columns=[t for t in selected_tickers if t in minutely_returns.columns])
    all_dates = mr.index.tolist()
    def _to_date(x):
        try:
            if pd.isna(x):
                return None
            if hasattr(x, "date") and callable(getattr(x, "date")):
                return x.date()
            return pd.Timestamp(x).date()
        except Exception:
            return None

    try:
        pv = load_pvclcl_matrix()
        pv.index = pv.index.astype(str).str.strip()
        pv_dates = pd.to_datetime(pv.columns)
        mr_dates = [d for d in (_to_date(x) for x in mr.index) if d is not None]
        pv_dates_list = [d for d in (_to_date(x) for x in pv_dates.tolist()) if d is not None]
        all_dates = sorted(set(mr_dates) | set(pv_dates_list))
    except FileNotFoundError:
        pv = pd.DataFrame()
    # Build tickers x dates: for each selected ticker, use minutely if available else pvCLCL
    pv_upper = {str(t).upper().replace(".", ""): t for t in pv.index} if not pv.empty else {}
    rows = {}
    for t in selected_tickers:
        if t in mr.columns and mr[t].notna().sum() >= 10:
            rows[t] = mr[t].reindex(all_dates)
        elif pv_upper:
            key = t.upper().replace(".", "")
            if key in pv_upper:
                pv_ticker = pv_upper[key]
                rows[t] = pd.Series(pv.loc[pv_ticker].values, index=pv.columns).reindex(all_dates)
            else:
                rows[t] = pd.Series(index=all_dates, dtype=float)
        else:
            rows[t] = mr[t].reindex(all_dates) if t in mr.columns else pd.Series(index=all_dates, dtype=float)
    # rows: dict ticker -> Series(index=dates); DataFrame(rows) -> columns=tickers, index=dates
    final = pd.DataFrame(rows).T
    final.index.name = "ticker"
    final.columns = pd.to_datetime(final.columns)
    final = final.reindex(sorted(final.columns), axis=1)
    out_path = OUT_DIR / "daily_returns_FINAL.parquet"
    final.to_parquet(out_path)
    print(f"Saved {out_path}  shape={final.shape} (tickers x dates)")
    return final


# -----------------------------------------------------------------------------
# STEP 8: Final universe CSV
# -----------------------------------------------------------------------------
def step8_final_universe_csv(selected: list[str], n: int = 100) -> None:
    out = selected[:n]
    path = OUT_DIR / f"FINAL_UNIVERSE_{n}.csv"
    pd.DataFrame({"ticker": out}).to_csv(path, index=False)
    print(f"Saved {path}  ({len(out)} tickers)")


# -----------------------------------------------------------------------------
# STEP 9: Summary report
# -----------------------------------------------------------------------------
def step9_summary_report(
    n_files: int,
    n_success: int,
    n_failed: int,
    date_range: tuple[Any, Any],
    final_size: int,
    tickers: list[str],
    corr_sample: float | None,
) -> None:
    path = OUT_DIR / "minutely_extraction_summary.txt"
    with open(path, "w") as f:
        f.write("MINUTELY EXTRACTION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total .7z files processed: {n_files}\n")
        f.write(f"Successfully extracted stocks: {n_success}\n")
        f.write(f"Failed extractions: {n_failed}\n")
        f.write(f"Date range of final dataset: {date_range[0]} to {date_range[1]}\n")
        f.write(f"Final universe size: {final_size}\n")
        f.write("Included tickers:\n")
        for i in range(0, len(tickers), 15):
            f.write("  " + ", ".join(tickers[i : i + 15]) + "\n")
        f.write(f"\nCorrelation with pvCLCL (sample): {corr_sample}\n")
        f.write("\nMega-caps confirmed present: " + ", ".join([t for t in MEGA_CAPS if t in tickers or t.upper() in [x.upper() for x in tickers]]) + "\n")
    print(f"Saved {path}")
    print("\n" + "=" * 60)
    print("✓ Minutely data extraction complete")
    print(f"✓ Final universe: {final_size} stocks")
    print(f"✓ Date range: {date_range[0]} to {date_range[1]}")
    print("✓ Ready for covariance forecasting pipeline")


# -----------------------------------------------------------------------------
# Test-run preview
# -----------------------------------------------------------------------------
def _print_test_preview(closes: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Print top 5 dates x top 5 tickers for closes and returns (verify before full run)."""
    print("\n" + "=" * 60)
    print("TEST RUN PREVIEW (first 5 dates x first 5 tickers)")
    print("=" * 60)
    print(f"Closes shape: {closes.shape}  |  Returns shape: {returns.shape}")
    n_d, n_t = 5, 5
    dates = closes.index[:n_d] if len(closes.index) >= n_d else closes.index
    tickers = closes.columns[:n_t].tolist() if len(closes.columns) >= n_t else closes.columns.tolist()
    if len(dates) and len(tickers):
        print("\nDaily closes (close-to-close):")
        print(closes.loc[dates, tickers].to_string())
        print("\nDaily log returns:")
        print(returns.loc[dates, tickers].to_string())
    print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Extract daily returns from minutely .7z archives.")
    ap.add_argument("--data-dir", type=str, default="", help="Path to data_by_stocks (default: project or ~/Downloads/data_by_stocks)")
    ap.add_argument("--workers", type=int, default=6, help="Parallel workers for Step 2 (default: 6)")
    ap.add_argument("--limit", type=int, default=None, help="Test run: only process first N stocks (e.g. --limit 10)")
    args = ap.parse_args()
    data_dir = _get_data_by_stocks(args.data_dir or None)
    workers = max(1, min(args.workers, 32))
    print(f"Using data dir: {data_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HAS_PY7ZR:
        print("pip install py7zr required.")
        return 1
    files = step1_inventory_and_inspection(data_dir)
    if len(files) == 0:
        print("No .7z files found.")
        return 1
    if args.limit is not None:
        files = files[: args.limit]
        print(f"TEST RUN: using first {len(files)} stocks only (--limit {args.limit})")
        # Quick diagnostic: how many CSVs in first archive?
        try:
            with py7zr.SevenZipFile(files[0], "r") as z:
                names = z.getnames()
                csvs = [n for n in names if str(n).lower().endswith(".csv")]
            print(f"First archive: {len(csvs)} CSV member(s) inside")
        except Exception as e:
            print(f"First archive check: {e}")
    try:
        closes = step2_extract_daily_closes_v2(files, workers=workers)
    except Exception as e:
        print(f"STEP 2 failed: {e}")
        return 1
    returns = step3_compute_returns(closes)
    if args.limit is not None:
        _print_test_preview(closes, returns)
    quality_df, exclude = step4_quality_report(returns)
    selected = step5_filter_top_stocks(returns, exclude, quality_df)
    corr = step6_compare_pvclcl(returns, selected)
    final_df = step7_merge_final(returns, selected)
    step8_final_universe_csv(selected, 100)
    step8_final_universe_csv(selected, 150)
    n_final = min(150, len(selected))
    dr = (final_df.columns.min(), final_df.columns.max()) if len(final_df.columns) else (None, None)
    step9_summary_report(len(files), len(closes.columns), len(files) - len(closes.columns), dr, n_final, selected[:n_final], corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
