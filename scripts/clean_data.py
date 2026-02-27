import os
import numpy as np
import pandas as pd

def _parse_dates(cols: pd.Index) -> pd.DatetimeIndex:
    # handles "YYYY-MM-DD", "YYYYMMDD", "XYYYYMMDD"
    s = cols.astype(str).str.replace(r"^X", "", regex=True)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        bad = s[dt.isna()][:5].tolist()
        raise ValueError(f"Failed to parse some date columns. Examples: {bad}")
    return pd.DatetimeIndex(dt)

def clean_returns_matrix_at_load(
    parquet_path: str,
    out_parquet_path: str | None = None,
    policy: str = "drop_date",      # "drop_date" | "winsorize_date" | "drop_ticker_values"
    q99_thresh: float = 0.5,        # 50% daily move is suspicious at universe level
    max_thresh: float = 1.0,        # >100% daily move: almost surely wrong at universe level
    clip_lo: float = -0.3,          # only used for winsorize policies
    clip_hi: float = 0.3,
    per_value_abs_thresh: float = 1.0,  # for drop_ticker_values
    min_non_nan_frac: float = 0.2,  # if <20% tickers present, consider date unreliable
    log_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Load ticker x date returns matrix from parquet, clean outlier/corrupted dates/values,
    and optionally save a cleaned parquet + a log CSV.

    Assumes:
      - index: tickers
      - columns: dates (parseable)
      - values: daily returns in DECIMAL units
    """
    df = pd.read_parquet(parquet_path).T
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Parquet did not load as a DataFrame.")

    # ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # parse and sort date columns
    dt = _parse_dates(df.columns)
    df.columns = dt
    df = df.loc[:, df.columns.sort_values()]

    # diagnostics per date
    abs_df = df.abs()
    q99 = abs_df.quantile(0.99, axis=0, interpolation="linear")
    mx = abs_df.max(axis=0, skipna=True)
    nan_frac = df.isna().mean(axis=0)
    non_nan_frac = 1.0 - nan_frac

    flagged = (q99 > q99_thresh) | (mx > max_thresh) | (non_nan_frac < min_non_nan_frac)
    flagged_dates = df.columns[flagged.values]

    # build log
    log = pd.DataFrame({
        "date": df.columns,
        "q99_abs": q99.values,
        "max_abs": mx.values,
        "nan_frac": nan_frac.values,
        "non_nan_frac": non_nan_frac.values,
        "flagged": flagged.values,
    })
    log["action"] = "keep"

    if len(flagged_dates) > 0:
        if policy == "drop_date":
            df.loc[:, flagged_dates] = np.nan
            log.loc[log["flagged"], "action"] = "drop_date->NaN"

        elif policy == "winsorize_date":
            # clip all values on flagged dates
            df.loc[:, flagged_dates] = df.loc[:, flagged_dates].clip(lower=clip_lo, upper=clip_hi)
            log.loc[log["flagged"], "action"] = f"winsorize_date[{clip_lo},{clip_hi}]"

        elif policy == "drop_ticker_values":
            # only remove extreme ticker values on flagged dates
            for d in flagged_dates:
                s = df[d]
                bad = s.abs() > per_value_abs_thresh
                df.loc[bad, d] = np.nan
            log.loc[log["flagged"], "action"] = f"drop_values_abs>{per_value_abs_thresh}"

        else:
            raise ValueError(f"Unknown policy: {policy}")

    # optional global clipping (even on non-flagged days)
    # uncomment if you want:
    # df = df.clip(lower=clip_lo, upper=clip_hi)

    # save outputs
    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
        log.to_csv(log_csv_path, index=False)

    if out_parquet_path is not None:
        os.makedirs(os.path.dirname(out_parquet_path) or ".", exist_ok=True)
        df.to_parquet(out_parquet_path)

    # console summary
    n_flag = int(flagged.sum())
    print(f"[clean] loaded {parquet_path} shape={df.shape}")
    print(f"[clean] flagged_dates={n_flag}/{df.shape[1]} (policy={policy})")
    if n_flag > 0:
        worst = log.sort_values("max_abs", ascending=False).head(5)
        print("[clean] worst by max_abs:")
        print(worst[["date", "q99_abs", "max_abs", "nan_frac", "action"]].to_string(index=False))

    return df