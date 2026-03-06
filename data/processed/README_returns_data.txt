RETURNS DATA FOR FORECASTING PIPELINE
======================================

Recommended (cleaned): data/processed/returns_universe_100_cleaned.parquet
  - Same as below but with 73 days removed where any stock had >50% daily return (splits/data errors).
  - See data/DATA_QUALITY_ISSUES.md.

Raw: data/processed/returns_universe_100.parquet

Structure:
- 3,655 dates (rows) x 100 tickers (columns) — pipeline expects index=dates, columns=tickers
- Date range: 2007-06-27 to 2021-12-31
- Tickers: See data/universes/FINAL_UNIVERSE_100_FINAL.csv
- Missing data: ~1.6%. Mean availability: ~98.4%. GOOGL/FB have lower coverage (IPO/later start).

Missing Data (NAs):
- NAs are NORMAL and expected
- They occur because:
  * Some stocks started after 2007 (e.g., FB in 2012)
  * Different stocks have different trading histories
  * Some had gaps due to halts or delisting
- Mean availability: ~98.4%
- All stocks have ≥70% coverage in recent period (2015-2021)

The pipeline should handle NAs by:
1. Skipping windows with too many NAs
2. Computing covariances only on available data
3. Using pairwise-complete observations

Usage:
```python
import pandas as pd
returns = pd.read_parquet('data/processed/returns_universe_100.parquet')
# returns is a DataFrame: index=dates, columns=tickers (3655 x 100)
```

If you see only 8 tickers instead of 100:
- You're using the wrong file (daily_returns_FINAL_100.parquet)
- Use returns_universe_100.parquet instead
