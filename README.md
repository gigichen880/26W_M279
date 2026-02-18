# Similarity-Based Covariance Forecasting for US Equities

## Project Overview

This project implements similarity-based covariance forecasting for US equities using daily/minutely return data. We build a clean universe of high-quality stocks, extract returns, and (in the pipeline phase) apply SPD geometry, regime detection, and GMVP backtesting against baselines (HAR, DCC-GARCH, Ledoit-Wolf).

## Team

- [Your name]
- [Teammate name]

## Project Structure

```
26W_M279/
├── data/               # Dataset storage (raw, processed, universes)
├── scripts/            # Data processing scripts
│   ├── data_extraction/
│   └── universe_selection/
├── notebooks/          # Jupyter notebooks for EDA
├── results/            # Analysis outputs (figures, reports)
├── archive/            # Old attempts (gitignored)
└── pipeline/           # Main forecasting pipeline (coming soon)
```

## Data

- **Source**: NYSE + NASDAQ daily returns (2007–2021); minutely data from archives for quality filtering.
- **Universe**: 100 stocks including major tech, financials, healthcare.
- **Availability**: ~98.5% mean data coverage for the final universe.

### Data Files

- `data/processed/returns_universe_100.parquet`: Final 100-stock returns matrix (gitignored; large).
- `data/processed/minutely_daily_returns.parquet`: Daily returns from minutely data (515 stocks; gitignored).
- `data/universes/FINAL_UNIVERSE_100_FINAL.csv`: List of selected 100 tickers (tracked).
- `data/universes/FINAL_UNIVERSE_metadata.txt`: Selection criteria and stats (tracked).

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Data files are gitignored due to size. Contact the team for access (e.g. Dropbox or shared drive).

## Usage

### Data Processing (already run)

```bash
# Build pvCLCL matrix from raw CSVs (data-dir = data/raw or unzipped source)
python scripts/data_extraction/build_pvclcl_matrix.py --data-dir data/raw --out-parquet data/processed/pvCLCL_matrix.parquet

# Extract daily returns from minutely .7z archives (writes to results/eda/reports)
python scripts/data_extraction/minutely_extraction_pipeline.py

# Select final universe (reads quality from results/eda/reports, writes to data/universes/)
python scripts/universe_selection/select_final_universe.py
```

### EDA

```bash
jupyter notebook notebooks/data_prep_eda.ipynb
```

### Main Pipeline (coming soon)

Similarity search, covariance forecasting, and backtesting will live under `pipeline/`.

## Key Results (EDA Phase)

- Extracted 515 stocks from minutely data.
- Selected 100 high-quality stocks (~98.5% availability).
- Includes 18 mega-caps: AAPL, MSFT, GOOGL, AMZN, NVDA, META/FB, NFLX, JPM, BAC, WFC, JNJ, PFE, UNH, WMT, HD, V, MA, NKE.

## Next Steps

1. Implement similarity-based forecasting pipeline.
2. SPD geometry (log-Euclidean representations).
3. Regime detection.
4. GMVP backtesting.
5. Comparison vs baselines (HAR, DCC-GARCH, Ledoit-Wolf).

## References

[Add key papers from your proposal.]

---

Last updated: February 2026
