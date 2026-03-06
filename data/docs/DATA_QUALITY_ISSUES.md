# Data Quality Issues and Fixes

## Issue 1: Extreme Returns (>50% daily)

**Discovered:** Feb 27, 2026 by Bree
**Severity:** High (affects 73 days out of 3655)

**Root Cause:**
Minutely data extraction did not adjust for:
- Stock splits (e.g., 2-for-1 splits appear as -50% returns)
- Reverse splits (e.g., 1-for-10 appears as +900% returns)
- Possible data corruption on thin trading days (holidays)

**Examples:**
- 2011-11-25: AMAT shows 1079.9% return (max_abs=10.80)
- 2011-11-28: AMAT shows -1078.0% return (max_abs=10.78)
- 2018-12-24: NWL shows 1018.4% return (max_abs=10.18)
- 2018-12-26: NWL shows -1018.2% return (max_abs=10.18)
- 2012-07-05: NDAQ shows -1000.7% return (max_abs=10.01)

**Affected Days:** 73 days
**Affected Tickers (top 5):** PVH, OKE, HRL, APD, SJM

**Fix Applied:**
Conservative approach: Removed 73 days entirely.
Alternative: Aggressive (set extreme values to NaN) also available.

**Impact:**
- Training data: 3582 days remain out of 3655 (98.00% retained)
- Pipeline: Should now produce stable covariance estimates
- Results: More robust to data quality issues

**Verification:**
After cleaning (conservative):
- Max |return| across all days: 48.96%
- Q99 |return|: 9.10%

---

## Revised cleaning: Three-tier classification (recommended)

**Problem with original approach:** Dropping all 73 days with >50% returns was too aggressive; some are legitimate volatility (e.g. single-stock 80% move during COVID).

**Classification of 73 extreme days:**
- **ARTIFACT (38 days):** >500% returns, reverse-split patterns (+1000% then -1000%), or multiple stocks same day → remove.
- **LEGITIMATE (25 days):** Single-stock 50–200% moves, no reverse pattern, often in volatile periods (COVID, 2008–09) → keep.
- **SUSPICIOUS (10 days):** 200–500% or unclear → remove in moderate tier.

**Three cleaned datasets:**

| Version   | Days removed | Days retained | Max return after | Use case                    |
|----------|--------------|---------------|------------------|-----------------------------|
| **strict**  | 38           | 3617          | 276%             | Artifacts only              |
| **moderate** | 48           | 3607          | **137%**         | **Recommended** (artifacts + suspicious) |
| **lenient** | 38           | 3617          | 276%             | Same as strict              |
| original_73 | 73         | 3582          | 49%              | Old “drop all >50%”         |

**Recommendation:** Use `returns_universe_100_cleaned_moderate.parquet` for day-level cleaning, or **preferred:** `returns_universe_100_cleaned_cellwise.parquet` for cell-level cleaning (only ~270 bad cells set to NaN, 0 days dropped, 99.9% data retained). See `data/docs/CELLWISE_CLEANING_REPORT.txt`. Run `python scripts/data_validation/cellwise_clean_returns.py` to regenerate.
