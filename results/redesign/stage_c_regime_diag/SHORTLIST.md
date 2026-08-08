# Stage C accepted models (query 2013–2016)

Acceptance rule (probabilistic only): reject near-uniform (mean max‑p < 0.35 or norm entropy > 0.95) or unjustified near-determinism (mean max‑p > 0.98 and norm entropy < 0.05). Hard k-means is a control.

## Accepted for Stage D

| Rep | Model | Query mean max‑p | Query norm H | Role |
|---|---|---:|---:|---|
| market_state | fcm_k3 | 0.823 | 0.495 | Soft regimes on market state |
| market_state | fcm_k4 | 0.640 | 0.628 | Soft regimes (closer to old K=4) |
| pca_only_8 | gmm_diag_k3 | 0.983 | 0.056 | Borderline sharp — use cautiously |
| pca_only_8 | hmm_diag_k3 | 0.955 | 0.111 | True HMM for D2/D5 |
| pca_only_8 | gmm_transition_k3 | 0.983 | 0.056 | Explicitly **not** an HMM |

## Rejected

- market_state + GMM/HMM: near-deterministic
- pca_only_8 + FCM: near-uniform (matches collapse decomp)
