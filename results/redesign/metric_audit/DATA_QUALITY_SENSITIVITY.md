# Data-quality sensitivity (frozen models, no retuning)

Deterministic rule on the unlocked OOS panel (`oos_robustness_panel.csv`, \(\eta=0.01\)).

## Future-window rule (primary)
Exclude anchors whose **future \(H=20\)** window contains any \(|r|>0.5\).
Retains **194/247** anchors (78.5%).

| Method | Frob med | Frob mean | Stein | \(R^{\mathrm{cond}}\) |
|---|---:|---:|---:|---:|
| shrink | 0.036 | 6.42 | 394 | **0.000077** |
| persistence | 0.016 | 7.43 | 4487 | 0.000079 |
| D0 | **0.014** | **0.024** | 86710 | 0.000110 (last) |

**Qualitative:** D0 still wins median Frobenius; D0 mean Frobenius collapses to the \(10^{-2}\) scale; D0 remains worst on Stein and last on \(R^{\mathrm{cond}}\).

## Past+future rule (secondary)
Exclude if max \(|r|\) in lookback \(L=50\) **or** future \(H=20\) exceeds 0.5.
Retains **95/247**. Mean Frobenius equalizes (~0.01–0.02) for all methods; D0 **loses** median-Frobenius lead (shrink/LW ahead); D0 still worst Stein and last \(R^{\mathrm{cond}}\).

Interpretation: mean-Frobenius contamination is largely extreme-date driven; the inverse/decision failure of D0 is not.
