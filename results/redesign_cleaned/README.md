# Cleaned-data rerun of the redesign protocol (2026-08-08)

Full rerun of the redesign campaign (commit d0ef8c4 code) with one change:
the data panel is `returns_universe_100_cleaned_cellwise.parquet` (cellwise
screen: ~270 corporate-action artifact cells set to missing, 52 legitimate
|r|>0.5 stress moves retained) instead of the uncleaned
`returns_universe_100.parquet` used by the original `results/redesign/` run.

Protocol order was preserved: stage A/C/D selection on 2013-2016 first,
freeze decision written (`RERUN_FREEZE_DECISION.md`) before any 2017-2021
evaluation, then the one-shot frozen test. Same frozen config re-selected
(SIM = whitened PCA-8, k=20, log-Euclidean, no regimes; regime increments
0/5 supported).

Run on scai4 (`~/rerun_cleaned`), log in `_provenance/campaign.log`.
Headline: SIM best median AND mean Frobenius, worst Stein/NLL/decision
regret; full-panel bootstrap contrasts exclude zero (no filter needed).
Selection-window change: Ledoit-Wolf now significantly beats SIM on regret
(p=0.004, was covers-zero on uncleaned data).
