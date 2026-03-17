## Results layout (current + recommended)

### Current (kept for backwards-compatibility)

- **Backtests / reports / configs** (flat in `results/`):
  - `results/<tag>_backtest.{parquet,csv}`
  - `results/<tag>_report.csv` (note: volatility uses `regime_volatility_report.csv`; covariance uses `regime_covariance_report.csv`)
  - `results/<tag>_config_used.yaml`
- **Figures**:
  - `results/figs_<tag>/raw_temporal/`
  - `results/figs_<tag>/statistical_comparison/`
  - `results/figs_<tag>/regime/`

This is what the scripts default to today.

### Recommended (next cleanup step)

Move each experiment into a single folder:

- `results/<tag>/backtest.{parquet,csv}`
- `results/<tag>/report.csv`
- `results/<tag>/config_used.yaml`
- `results/<tag>/figs/raw_temporal/`
- `results/<tag>/figs/statistical_comparison/`
- `results/<tag>/figs/regime/`

If you want this, we can implement it **without breaking existing paths** by:
1) Writing new outputs into `results/<tag>/...`, and
2) Keeping a small “index” file in `results/` or auto-resolving both layouts in scripts.

