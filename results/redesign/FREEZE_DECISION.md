# Freeze decision (end of selection window 2013–2016)

Documented **before** unlocking 2017–2021.

## Selection rule (pre-registered)

1. Primary: mean conditioned GMVP variance regret  
2. Secondary: realized GMVP variance  
3. Tie-break: no regime ≻ regime; lower \(D\); fewer components; D2-style nesting ≻ expert mixtures  

## Selection evidence

| Method | Var regret | Notes |
|---|---:|---|
| D5_pca8_hmm3 | 0.001137 | Point-estimate best |
| **D0_pca8** | **0.001161** | Similarity-only |
| D2_pca8_hmm3 | 0.002375 | Regime overlap **hurts** |
| Ledoit–Wolf | 0.001500 | Best classical |
| D0/D2 market | ~0.0049 | Worse |

Incremental bootstrap (see `stage_d/incremental_bootstrap.csv`): if D5−D0 CI covers 0, D5 does **not** earn its arrow under the simplicity tie-break.

## Frozen architecture for OOS

- Embedder: `pca_only`, \(D=8\), whitened  
- Neighbors: \(k=20\), Euclidean, adaptive kernel  
- Aggregation: log-Euclidean  
- Regime: **none** (D0)  
- Conditioner: \(C_\eta\) with \(\eta=0.01\) on forecast **and** realized target for SPD metrics; regret quadratic form on raw \(\Sigma^{\mathrm{real}}\)  
- No persistence blend, no leverage gate  

**Parameters continue expanding/refitting** through 2017–2021.

Optional descriptive OOS companion (not selected): D5_pca8_hmm3 for transparency only.
