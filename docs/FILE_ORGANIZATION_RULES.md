# File Organization Rules

## Directory Structure

```
26W_M279/
├── data/
│   ├── raw/              # Original source data (gitignored)
│   ├── processed/        # Clean, ready-to-use datasets
│   ├── universes/        # Universe selection CSVs (small, tracked)
│   └── docs/             # Data documentation
├── scripts/
│   ├── data_extraction/  # Download/extract raw data
│   ├── universe_selection/ # Filter and select stocks
│   ├── data_validation/  # Quality checks and diagnostics
│   └── analysis/        # Main analysis scripts
├── similarity_forecast/  # Main package (pipeline)
├── notebooks/            # Jupyter notebooks
├── results/
│   ├── eda/
│   │   ├── figures/      # PNG/PDF plots
│   │   └── reports/      # CSV/TXT summaries
│   └── latex_tables/     # LaTeX tables for Overleaf
├── tests/                # Unit tests
├── docs/                 # General documentation
└── archive/              # Old versions (gitignored)
    ├── old_scripts/
    └── old_data/
```

## File Naming Conventions

### Data Files
- **Raw data:** `{source}_{description}.{ext}`
  - Example: `pvCLCL_matrix.parquet`, `crsp_20000101_20241231.csv`
- **Processed data:** `{description}_{version}.parquet`
  - Example: `returns_universe_100_cleaned.parquet`
- **Universe files:** `FINAL_UNIVERSE_{size}_{version}.csv`
  - Example: `FINAL_UNIVERSE_100_FINAL.csv`

### Scripts
- **Descriptive names:** `{verb}_{object}.py`
  - Example: `extract_minutely_data.py`, `investigate_extreme_returns.py`
- **NOT:** `script1.py`, `test.py`, `temp.py`

### Figures
- **Descriptive names:** `{analysis_type}_{details}.png`
  - Example: `returns_distribution.png`, `availability_heatmap.png`
- **NOT:** `figure1.png`, `plot.png`

### Documentation
- **ALL CAPS for important docs:** `README.md`, `DATA_QUALITY_ISSUES.md`
- **lowercase for specific docs:** `migration_plan.md`, `setup_instructions.md`

## Where Files Should Go

### NEVER put in repo root:
- ❌ .py scripts (→ scripts/)
- ❌ .parquet files (→ data/processed/)
- ❌ .png files (→ results/)
- ❌ .csv files (→ data/)

### ONLY in repo root:
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore
- ✅ run_*.py (main entry points like run_regime_similarity.py)

### Data files:
- **Source data** → `data/raw/`
- **Cleaned data** → `data/processed/`
- **Small CSVs** → `data/universes/` or appropriate subfolder

### Scripts:
- **Extract/download** → `scripts/data_extraction/`
- **Select universe** → `scripts/universe_selection/`
- **Validate data** → `scripts/data_validation/`
- **Analysis** → `scripts/analysis/`
- **Tests** → `tests/`

### Results:
- **Figures** → `results/eda/figures/` (EDA); regime backtest figures → `results/figs_regime_similarity/`
- **Reports** → `results/eda/reports/`; backtest report CSV and config snapshot → `results/` (e.g. `regime_similarity_report.csv`, `regime_similarity_config_used.yaml`)
- **LaTeX** → `results/latex_tables/`

### Documentation:
- **Data docs** → `data/docs/`
- **General docs** → `docs/`
- **Code docs** → `similarity_forecast/` (near code)

## When Creating New Files

**BEFORE creating a file, ask:**
1. What type of file is this? (data/script/figure/doc)
2. Where does this type go? (check structure above)
3. Does a similar file already exist? (avoid duplicates)

**ALWAYS:**
- Use descriptive names
- Put files in correct folders
- Update .gitignore if needed
- Document in appropriate README

**NEVER:**
- Create files in repo root (except run_*.py)
- Use generic names (temp.py, test.csv)
- Duplicate existing files
- Leave old versions in working directories

---

## Cursor / AI prompt reminder

When asking Cursor (or any AI) to create or move files, include:

**FILE ORGANIZATION:** Before creating ANY new file, use the correct location:
- Data → `data/processed/` or `data/raw/`
- Scripts → `scripts/{data_extraction,universe_selection,data_validation,analysis}/`
- Figures → `results/eda/figures/`
- Reports → `results/eda/reports/`
- LaTeX → `results/latex_tables/`
- Docs → `data/docs/` or `docs/`

**NEVER** create files in repo root except: `run_*.py`, `README.md`, `requirements.txt`, `.gitignore`.
