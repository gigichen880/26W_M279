#!/usr/bin/env python3
"""
Apply CORRECT quality filters; force-include mega-caps FIRST, then fill to 150 by score.
Uses quality_metrics_all_515.csv. Saves FINAL_UNIVERSE_*_FINAL.csv and metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUALITY_CSV = REPO_ROOT / "results" / "eda" / "reports" / "quality_metrics_all_515.csv"
OUT_DIR = REPO_ROOT / "data" / "universes"

# STEP 1: Mega-caps (MUST be included, placed first)
MEGA_CAPS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA",
    "META", "FB", "TSLA", "NFLX", "JPM", "BAC", "WFC",
    "JNJ", "PFE", "UNH", "WMT", "HD", "V", "MA",
]
ETF_TICKERS = {"SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR"}

# STEP 2: Filter thresholds
MIN_RECENT_AVAIL_PCT = 85.0
MIN_AVAIL_PCT = 70.0
MAX_LONGEST_GAP_DAYS = 365
MAX_EXTREME_DAYS = 50
MEGA_OVERRIDE_MIN_RECENT = 80.0  # Still force-include mega-cap if recent_avail >= 80%


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not QUALITY_CSV.exists():
        raise FileNotFoundError(f"Not found: {QUALITY_CSV}")
    q = pd.read_csv(QUALITY_CSV)

    # STEP 2: Apply filters → passed_stocks
    hard_recent = q["recent_availability_pct"] >= MIN_RECENT_AVAIL_PCT
    hard_avail = q["availability_pct"] >= MIN_AVAIL_PCT
    hard_gap = q["longest_gap"] <= MAX_LONGEST_GAP_DAYS
    soft_extreme = q["extreme_days"] <= MAX_EXTREME_DAYS
    passed = q[hard_recent & hard_avail & hard_gap & soft_extreme]
    passed_stocks = passed["ticker"].tolist()

    # STEP 3: Force-include mega-caps FIRST
    final_universe = []
    ticker_to_row = q.set_index("ticker")
    for mega in MEGA_CAPS:
        if mega in passed_stocks:
            final_universe.append(mega)
            print(f"✓ Force-included {mega}")
        elif mega in q["ticker"].values:
            row = ticker_to_row.loc[mega]
            recent = row["recent_availability_pct"]
            if recent >= MEGA_OVERRIDE_MIN_RECENT:
                final_universe.append(mega)
                print(f"✓ Force-included {mega} (overrode filters: recent={recent:.1f}%)")
            else:
                print(f"✗ Could not include {mega}: recent_avail={recent:.1f}%")
        else:
            print(f"✗ Could not include {mega}: not in quality data")

    # STEP 4: Exclude ETFs from remaining pool
    remaining = [s for s in passed_stocks if s not in final_universe]
    remaining = [s for s in remaining if s.upper() not in {e.upper() for e in ETF_TICKERS}]

    # STEP 5: Rank remaining by quality score
    q_rem = q[q["ticker"].isin(remaining)].copy()
    q_rem["score"] = 0.5 * q_rem["recent_availability_pct"] + 0.5 * q_rem["availability_pct"]
    q_rem = q_rem.sort_values("score", ascending=False)
    remaining_ranked = q_rem["ticker"].tolist()

    # STEP 6: Fill to 150 total; then 100-stock version
    slots_150 = 150 - len(final_universe)
    universe_150 = final_universe + remaining_ranked[:slots_150]

    slots_100 = 100 - len(final_universe)
    if slots_100 <= 0:
        universe_100 = final_universe[:100]
    else:
        universe_100 = final_universe + remaining_ranked[:slots_100]

    # STEP 7: Validate
    q150 = q[q["ticker"].isin(universe_150)]
    q100 = q[q["ticker"].isin(universe_100)]
    mega_in_150 = [t for t in MEGA_CAPS if t in universe_150]
    mega_in_100 = [t for t in MEGA_CAPS if t in universe_100]

    print()
    print("Final Universe (150 stocks):")
    print(f"  Mega-caps included: {len(mega_in_150)} - {mega_in_150}")
    print(f"  Other high-quality stocks: {150 - len(mega_in_150)}")
    print(f"  Mean recent availability: {q150['recent_availability_pct'].mean():.1f}%")
    print()
    print("Final Universe (100 stocks):")
    print(f"  Mega-caps included: {len(mega_in_100)} - {mega_in_100}")
    print(f"  Other high-quality stocks: {100 - len(mega_in_100)}")
    print(f"  Mean recent availability: {q100['recent_availability_pct'].mean():.1f}%")
    print()

    # STEP 8: Save
    pd.DataFrame({"ticker": universe_150}).to_csv(OUT_DIR / "FINAL_UNIVERSE_150_FINAL.csv", index=False)
    pd.DataFrame({"ticker": universe_100}).to_csv(OUT_DIR / "FINAL_UNIVERSE_100_FINAL.csv", index=False)
    print("Saved FINAL_UNIVERSE_150_FINAL.csv")
    print("Saved FINAL_UNIVERSE_100_FINAL.csv")

    meta_path = OUT_DIR / "FINAL_UNIVERSE_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("Final universe selection (FINAL)\n")
        f.write("=" * 60 + "\n")
        f.write("1. Mega-caps (20) are force-included first.\n")
        f.write("   If a mega-cap failed filters but has recent_availability_pct >= 80%,\n")
        f.write("   it is still included (override).\n")
        f.write("2. Hard filters: recent_avail >= 85%, avail >= 70%, gap <= 365d, extreme_days <= 50.\n")
        f.write("3. ETFs excluded from pool: SPY, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, IYR.\n")
        f.write("4. Remaining slots filled by composite score (0.5*recent_avail + 0.5*overall_avail).\n")
        f.write("5. FINAL_UNIVERSE_150_FINAL.csv = mega-caps (in order) + top 150 - len(mega) by score.\n")
        f.write("6. FINAL_UNIVERSE_100_FINAL.csv = same logic for 100 stocks.\n")
    print(f"Saved {meta_path}")
    print()

    print("First 30 tickers:")
    first30 = universe_150[:30]
    for i in range(0, len(first30), 10):
        print("  " + ", ".join(first30[i : i + 10]))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
