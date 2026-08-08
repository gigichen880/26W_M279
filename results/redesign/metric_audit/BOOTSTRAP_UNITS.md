# Bootstrap units audit

## What is resampled?
The incremental bootstrap in `scripts/redesign/run_incremental_bootstrap.py` operates on
**paired sequences of evaluation anchors** (one observation per stride-5 forecast date).

It does **not** resample a daily portfolio-return primitive.

## Meaning of block length 20
`block=20` means **20 consecutive anchors** ≈ 20 × 5 = **100 trading days**.

That is a conservative dependence assumption relative to the 20-day holding horizon.

## Manuscript wording
Prefer: "moving-block bootstrap over evaluation anchors; block length B anchors (≈ 5B trading days)."

## Sensitivity
See `bootstrap_block_sensitivity.csv` for B ∈ {4,8,12,20} (≈20–100 trading days).
