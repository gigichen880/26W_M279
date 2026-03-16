# Regime Analysis — Verification, Visualization, and Summary

## STEP 1: Verify regime data

Regime columns are saved by `run_backtest.py` when the pipeline uses `return_regime=True`:

- **regime_assigned** (int): Hard regime = argmax(α_t)
- **regime_prob_0 … regime_prob_3**: Filtered probabilities α_t(k)
- **regime_raw_0 … regime_raw_3**: Raw GMM probabilities π_t(k)

**Quick check (from repo root):**
```bash
python scripts/analysis/verify_regime_data.py
```
Or with explicit path:
```bash
python scripts/analysis/verify_regime_data.py results/regime_covariance_backtest.parquet
```

You should see: regime-related columns listed, total evaluation dates, date range, regime distribution, no missing assignments, and probability sums ≈ 1.0.

---

## STEP 2: Run regime visualization

Generate all regime figures and characterizations:

```bash
python scripts/analysis/visualize_regimes.py
```

Optional arguments:
- `--backtest results/regime_covariance_backtest.parquet`
- `--outdir results/figs_regime_covariance`
- `--K 4`

This produces:

- **results/figs_regime_covariance/regime_timeline.png** — Hard regime over time with crisis shading
- **results/figs_regime_covariance/regime_probs_stacked.png** — Stacked α_t over time
- **results/figs_regime_covariance/regime_filtering_effect.png** — Raw π vs filtered α by regime
- **results/regime_characterization.csv** — Per-regime stats (n_days, pct_time, mean_fro, mean_gmvp_sharpe, etc.)
- **results/regime_names_mapping.json** — Suggested names (Regime 0 → "…", etc.)

The script also prints: REGIME CHARACTERIZATION, KEY OBSERVATIONS (most common regime, best/worst Sharpe, highest error/turnover), and suggested regime names.

---

## STEP 3: Inspect regime characterization

After running `visualize_regimes.py`, open **results/regime_characterization.csv**. The script will have printed:

- **REGIME CHARACTERIZATION SUMMARY** — Full table
- **KEY OBSERVATIONS** — Most common regime, best/worst GMVP Sharpe, highest forecast error (Frobenius), highest turnover

Use these to tie regime indices to market conditions and to label regimes in the report.

---

## STEP 4: Visual inspection guidance

Open the generated figures and check:

1. **results/figs_regime_covariance/regime_timeline.png**
   - Do regimes line up with known stress periods? (e.g. 2008–2009, 2020 COVID, 2015–2016 selloff)
   - Which regime(s) dominate in calm periods (e.g. 2017, 2019)?

2. **results/figs_regime_covariance/regime_probs_stacked.png**
   - Are transitions smooth or abrupt?
   - Do regimes persist (probabilities stay high for a while)?
   - Any rapid switching?

3. **results/figs_regime_covariance/regime_filtering_effect.png**
   - Is filtered α smoother than raw π?
   - Does the Markov filter reduce noise?

---

## STEP 5: Interpretable regime names

**results/regime_names_mapping.json** contains suggested names from heuristics (e.g. Calm Bull, Crisis/High Stress, High Uncertainty, Volatile/Choppy, Normal/Transition). Adjust names after looking at:

- **regime_characterization.csv** (which regime has best/worst Sharpe, highest error/turnover)
- **regime_timeline.png** (when each regime appears)
- **regime_probs_stacked.png** (how regimes evolve)

Then either edit the JSON by hand or use the mapping in your report text and figure captions.

---

## STEP 6: Summary

After verification and visualization:

**REGIME ANALYSIS COMPLETE**

- Regime assignments are in the backtest results (regime_assigned, regime_prob_*, regime_raw_*).
- Timeline, stacked probabilities, and filtering-effect figures are in **results/figs_regime_covariance/**.
- Regime characterization and name mapping are in **results/regime_characterization.csv** and **results/regime_names_mapping.json**.

**Next steps:**

1. Review **regime_characterization.csv** and the KEY OBSERVATIONS to interpret each regime.
2. Assign final interpretable names using the timeline and stacked plots; update **regime_names_mapping.json** if desired.
3. Use these names and figures in the final report (regime section and discussion).
4. Optionally run performance-by-regime analysis (e.g. Sharpe by regime) using the backtest CSV and **regime_assigned**.
