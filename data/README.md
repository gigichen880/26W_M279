# Data Directory Structure

## Raw Data (gitignored)

Large source files are not tracked in git:

- `raw/pvCLCL_matrix.parquet.zip`: NYSE daily returns (e.g. 2000–2021).
- `raw/pvCLCL_matrix.csv.zip`: Same data in CSV form.
- `raw/data_by_stocks/`: Minutely price data archives (`.7z` files).

## Processed Data

- `processed/minutely_daily_returns.parquet`: Daily returns extracted from minutely data (515 stocks, 2007–2021).
- `processed/minutely_daily_closes.parquet`: Daily closes from minutely data.
- `processed/returns_universe_100.parquet`: Final 100-stock universe returns matrix.
- Other `processed/*.parquet` outputs from the extraction pipeline.

## Universes (tracked in git)

Small CSV/text files defining stock selection:

- `universes/FINAL_UNIVERSE_100_FINAL.csv`: Final 100 stocks.
- `universes/FINAL_UNIVERSE_150_FINAL.csv`: Alternative 150-stock universe.
- `universes/FINAL_UNIVERSE_metadata.txt`: Selection criteria and statistics.

## Accessing Data

Because of size, raw and processed data are not in the repo. To get them:

1. Ask team members for a shared link (e.g. Dropbox) to raw data.
2. Or use already-built processed files if provided.
