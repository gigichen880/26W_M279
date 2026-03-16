#!/usr/bin/env python3
"""
Classify extreme returns (>50%) as ARTIFACT vs LEGITIMATE vs SUSPICIOUS.
Create three-tier cleaned datasets: strict, moderate, lenient.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO / "data" / "processed"
THRESHOLD = 0.5
IMPOSSIBLE_RETURN = 5.0  # >500% = definitely artifact

# Volatile periods (keep single-stock 50-200% in these windows)
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-06-30")
CRISIS_START = pd.Timestamp("2008-09-01")
CRISIS_END = pd.Timestamp("2009-06-30")


@dataclass
class ExtremeDay:
    date: pd.Timestamp
    ticker: str
    return_val: float
    ret_before: Optional[float]
    ret_after: Optional[float]
    ret_2after: Optional[float]
    row_idx: int
    col_idx: int
    n_stocks_extreme_same_day: int  # how many stocks had |r|>0.5 on this day


def load_extreme_days(returns_df: pd.DataFrame) -> List[ExtremeDay]:
    """Find all dates with max |return| > 0.5 and build records with context."""
    R = returns_df.values
    dates = returns_df.index
    tickers = returns_df.columns.tolist()
    T, N = R.shape
    abs_ret = np.abs(R)

    extreme_days: List[ExtremeDay] = []
    seen_dates = set()

    for row_idx in range(T):
        row_abs = abs_ret[row_idx]
        max_abs = np.nanmax(row_abs)
        if max_abs <= THRESHOLD or np.isnan(max_abs):
            continue
        d = dates[row_idx]
        if d in seen_dates:
            continue
        seen_dates.add(d)

        # Ticker with max return on this day (skip if row is all NaN)
        try:
            col_idx = int(np.nanargmax(row_abs))
        except ValueError:
            continue
        ticker = tickers[col_idx]
        ret_val = float(R[row_idx, col_idx])
        rb = float(R[row_idx - 1, col_idx]) if row_idx > 0 else np.nan
        ra = float(R[row_idx + 1, col_idx]) if row_idx < T - 1 else np.nan
        r2a = float(R[row_idx + 2, col_idx]) if row_idx < T - 2 else np.nan

        n_extreme = int(np.sum(abs_ret[row_idx] > THRESHOLD))

        extreme_days.append(
            ExtremeDay(
                date=pd.Timestamp(d),
                ticker=ticker,
                return_val=ret_val,
                ret_before=None if (row_idx == 0 or np.isnan(rb)) else rb,
                ret_after=None if (row_idx >= T - 1 or np.isnan(ra)) else ra,
                ret_2after=None if (row_idx >= T - 2 or np.isnan(r2a)) else r2a,
                row_idx=row_idx,
                col_idx=col_idx,
                n_stocks_extreme_same_day=n_extreme,
            )
        )

    return extreme_days


def is_reverse_split_pattern(e: ExtremeDay, all_extreme: List[ExtremeDay]) -> bool:
    """Large positive followed by large negative (or vice versa) within 2 days, same ticker."""
    if e.ret_after is None:
        return False
    # Same ticker next day: opposite sign and huge
    if abs(e.return_val) > 2 and abs(e.ret_after) > 2:
        if (e.return_val > 0 and e.ret_after < 0) or (e.return_val < 0 and e.ret_after > 0):
            return True
    if e.ret_2after is not None and abs(e.return_val) > 2 and abs(e.ret_2after) > 2:
        if (e.return_val > 0 and e.ret_2after < 0) or (e.return_val < 0 and e.ret_2after > 0):
            return True
    return False


def in_volatile_period(ts: pd.Timestamp) -> bool:
    return (COVID_START <= ts <= COVID_END) or (CRISIS_START <= ts <= CRISIS_END)


def classify_one(e: ExtremeDay, all_extreme: List[ExtremeDay]) -> str:
    """
    Classify as ARTIFACT, LEGITIMATE, or SUSPICIOUS.
    """
    abs_ret = abs(e.return_val)

    # DEFINITELY BAD
    if abs_ret > IMPOSSIBLE_RETURN:
        return "ARTIFACT"
    if is_reverse_split_pattern(e, all_extreme):
        return "ARTIFACT"
    if e.n_stocks_extreme_same_day >= 3 and abs_ret > 2:
        return "ARTIFACT"

    # POSSIBLY LEGITIMATE
    if 0.5 <= abs_ret <= 2.0 and e.n_stocks_extreme_same_day == 1 and not is_reverse_split_pattern(e, all_extreme):
        if in_volatile_period(e.date):
            return "LEGITIMATE"
        if abs_ret <= 1.0:  # 50-100% single stock, no split pattern
            return "LEGITIMATE"

    # SUSPICIOUS: 200-500%, or multiple stocks, or not in volatile period
    if 2.0 < abs_ret <= IMPOSSIBLE_RETURN:
        return "SUSPICIOUS"
    if e.n_stocks_extreme_same_day >= 2:
        return "SUSPICIOUS"
    if 1.0 < abs_ret <= 2.0 and not in_volatile_period(e.date):
        return "SUSPICIOUS"

    return "LEGITIMATE"


def main():
    returns_df = pd.read_parquet(PROCESSED / "returns_universe_100.parquet")
    if returns_df.shape[1] > returns_df.shape[0]:
        returns_df = returns_df.T

    extreme_days = load_extreme_days(returns_df)
    classified = [(e, classify_one(e, extreme_days)) for e in extreme_days]

    n_artifact = sum(1 for _, c in classified if c == "ARTIFACT")
    n_legit = sum(1 for _, c in classified if c == "LEGITIMATE")
    n_susp = sum(1 for _, c in classified if c == "SUSPICIOUS")

    print("=" * 60)
    print("STEP 1 & 2: Extreme days and classification")
    print("=" * 60)
    print(f"Found {len(extreme_days)} days with max |return| > 50%")
    print(f"\nClassification of {len(extreme_days)} extreme return days:")
    print(f"  - Definitely artifacts: {n_artifact}")
    print(f"  - Possibly legitimate: {n_legit}")
    print(f"  - Suspicious/unclear: {n_susp}")

    # STEP 4: Patterns and examples
    artifacts = [(e, c) for e, c in classified if c == "ARTIFACT"]
    legit = [(e, c) for e, c in classified if c == "LEGITIMATE"]
    susp = [(e, c) for e, c in classified if c == "SUSPICIOUS"]

    n_split_pairs = sum(1 for e, _ in artifacts if is_reverse_split_pattern(e, extreme_days))
    n_impossible = sum(1 for e, _ in artifacts if abs(e.return_val) > IMPOSSIBLE_RETURN)
    n_multi = sum(1 for e, _ in artifacts if e.n_stocks_extreme_same_day >= 2)

    print("\n" + "=" * 60)
    print("STEP 4: Analyze patterns")
    print("=" * 60)
    print("Artifact patterns:")
    print(f"  - Consecutive +/- pairs (splits): {n_split_pairs}")
    print(f"  - Returns >500%: {n_impossible}")
    print(f"  - Multiple stocks same day: {n_multi}")
    print("\nTop 5 ARTIFACT examples:")
    for e, _ in sorted(artifacts, key=lambda x: -abs(x[0].return_val))[:5]:
        print(f"  {e.date.date()} {e.ticker} return={e.return_val*100:.1f}% before={e.ret_before} after={e.ret_after} n_stocks={e.n_stocks_extreme_same_day}")
    print("\nTop 5 LEGITIMATE examples:")
    for e, _ in sorted(legit, key=lambda x: -abs(x[0].return_val))[:5]:
        print(f"  {e.date.date()} {e.ticker} return={e.return_val*100:.1f}% n_stocks={e.n_stocks_extreme_same_day} volatile_period={in_volatile_period(e.date)}")
    print("\nTop 5 SUSPICIOUS examples:")
    for e, _ in sorted(susp, key=lambda x: -abs(x[0].return_val))[:5]:
        print(f"  {e.date.date()} {e.ticker} return={e.return_val*100:.1f}% n_stocks={e.n_stocks_extreme_same_day}")

    # STEP 5: Suspicious detail (top 10)
    print("\n" + "=" * 60)
    print("STEP 5: Suspicious cases (top 10)")
    print("=" * 60)
    for e, _ in sorted(susp, key=lambda x: -abs(x[0].return_val))[:10]:
        vol = "Yes" if in_volatile_period(e.date) else "No"
        print(f"  {e.date.date()} {e.ticker} return={e.return_val*100:.1f}% | before={e.ret_before} after={e.ret_after} | n_stocks_extreme={e.n_stocks_extreme_same_day} | volatile_period={vol}")

    # STEP 6: Three-tier cleaning
    dates = returns_df.index
    R = returns_df.values
    abs_ret = np.abs(R)
    max_abs_per_date = np.nanmax(abs_ret, axis=1)

    # Days to remove per tier (by date)
    artifact_dates = {e.date for e, c in classified if c == "ARTIFACT"}
    suspicious_dates = {e.date for e, c in classified if c == "SUSPICIOUS"}

    strict_remove = artifact_dates
    moderate_remove = artifact_dates | suspicious_dates
    lenient_remove = set()
    for e, c in classified:
        if c == "ARTIFACT":
            lenient_remove.add(e.date)
        elif abs(e.return_val) > IMPOSSIBLE_RETURN:
            lenient_remove.add(e.date)
    # Lenient: only >500% and clear split pairs (artifact set is already that)
    lenient_remove = artifact_dates

    def drop_dates(df: pd.DataFrame, to_remove: set) -> pd.DataFrame:
        to_drop = [d for d in df.index if pd.Timestamp(d) in to_remove]
        return df.drop(to_drop, errors="ignore")

    clean_strict = drop_dates(returns_df.copy(), strict_remove)
    clean_moderate = drop_dates(returns_df.copy(), moderate_remove)
    clean_lenient = drop_dates(returns_df.copy(), lenient_remove)

    clean_strict.to_parquet(PROCESSED / "returns_universe_100_cleaned_strict.parquet")
    clean_moderate.to_parquet(PROCESSED / "returns_universe_100_cleaned_moderate.parquet")
    clean_lenient.to_parquet(PROCESSED / "returns_universe_100_cleaned_lenient.parquet")

    print("\n" + "=" * 60)
    print("STEP 6: Three-tier cleaning")
    print("=" * 60)
    print(f"STRICT:  remove {len(strict_remove)} days (artifacts only)")
    print(f"MODERATE: remove {len(moderate_remove)} days (artifacts + suspicious)")
    print(f"LENIENT: remove {len(lenient_remove)} days (artifacts only, same as strict here)")

    # STEP 7: Compare
    def stats(df: pd.DataFrame) -> dict:
        V = df.values
        a = np.abs(V)
        return {
            "days": len(df),
            "max_ret": np.nanmax(a) * 100,
            "q99_ret": np.nanpercentile(a, 99) * 100,
            "n_na": int(np.isnan(V).sum()),
        }

    orig = stats(returns_df)
    s = stats(clean_strict)
    m = stats(clean_moderate)
    l = stats(clean_lenient)

    print("\n" + "=" * 60)
    print("STEP 7: Compare approaches")
    print("=" * 60)
    print(f"Original:     days={orig['days']} max_ret={orig['max_ret']:.2f}% q99={orig['q99_ret']:.2f}%")
    print(f"STRICT:       days={s['days']} removed={orig['days']-s['days']} max_ret={s['max_ret']:.2f}% q99={s['q99_ret']:.2f}%")
    print(f"MODERATE:     days={m['days']} removed={orig['days']-m['days']} max_ret={m['max_ret']:.2f}% q99={m['q99_ret']:.2f}%")
    print(f"LENIENT:      days={l['days']} removed={orig['days']-l['days']} max_ret={l['max_ret']:.2f}% q99={l['q99_ret']:.2f}%")

    removed_mod = orig["days"] - m["days"]
    pct_mod = 100 * removed_mod / orig["days"]
    print(f"\nRecommended approach: MODERATE")
    print(f"Reasoning: Removes clear artifacts and suspicious cases; keeps single-stock volatility in volatile periods. Balances data quality and retention.")
    print(f"Days removed: {removed_mod} ({pct_mod:.2f}% of data)")
    print(f"Max return after cleaning: {m['max_ret']:.2f}%")

    # STEP 8: Validate 3 kept examples (50-100% returns we keep)
    kept_examples = [e for e, c in classified if c == "LEGITIMATE" and 0.5 <= abs(e.return_val) <= 1.0][:3]
    print("\n" + "=" * 60)
    print("STEP 8: Validate kept 50-100% examples")
    print("=" * 60)
    for e in kept_examples:
        vol = "COVID/crisis period" if in_volatile_period(e.date) else "single-stock event"
        print(f"  {e.date.date()}, ticker {e.ticker}, return {e.return_val*100:.1f}%:")
        print(f"    - No reverse pattern in following days")
        print(f"    - {vol}")
        print(f"    - Classification: KEEP (LEGITIMATE)")

    # STEP 9: Final recommendation
    print("\n" + "=" * 60)
    print("REVISED CLEANING RECOMMENDATION")
    print("=" * 60)
    print("""
PROBLEM WITH ORIGINAL APPROACH:
- Dropped 73 days using 50% threshold
- Too aggressive: removed potentially legitimate volatility
- Example: Biotech stock up 80% on FDA approval = legitimate, not error

BETTER APPROACH:
Use MODERATE cleaning:
- Remove only clear artifacts (>500% or split patterns) and suspicious cases
- Keep legitimate single-day volatility (50-200% in volatile periods or single-stock)
- Removes {} days instead of 73
- Retains more real data, removes only clear errors

EVIDENCE:
- {} of 73 days show clear split/artifact patterns (must remove)
- {} of 73 days show single-stock volatility (possibly real, kept in lenient)
- {} suspicious days removed in moderate
- Keeping legitimate preserves market reality (tail events happen!)

FILES CREATED:
- returns_universe_100_cleaned_moderate.parquet (RECOMMENDED)
- returns_universe_100_cleaned_strict.parquet
- returns_universe_100_cleaned_lenient.parquet

UPDATE run_regime_covariance.py to use:
  returns_universe_100_cleaned_moderate.parquet
""".format(
        len(moderate_remove),
        n_artifact,
        n_legit,
        n_susp,
    ))


if __name__ == "__main__":
    main()
