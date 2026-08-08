# Stage B shortlist (selection window 2013–2016 only)

Primary metric: mean **conditioned GMVP variance regret**.

| Embedder | Similarity regret | vs LW | Notes |
|---|---:|---:|---|
| **pca_only k=8** | **0.001161** | beats LW (0.001500) | Shortlist #1 for forecasting |
| market_state 6D | 0.004889 | worse than LW | Best FCM memberships; shortlist #2 for regime layer |
| spectral | 0.004204 | worse | Drop for now |
| hybrid | 0.007182 | worst | Drop |

Geometry/regime readiness (collapse decomp): market_state has query mean max‑\(p\)≈0.64; PCA-only FCM stays near-uniform — so for Stage C/D use **market_state** (and optionally GMM/HMM on pca8 without requiring FCM).

**Frozen for next stages (already-defined):**
1. `pca_only` with \(D=8\) — similarity core candidate
2. `market_state` — regime-capable representation

No new representations after this point without returning to the 2008–2012 development protocol.
