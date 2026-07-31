# Related-work notes: statistical vs economic evaluation of covariance forecasts

**Track B / Bree (B3).** Ready-to-paste BibTeX + 1–2 sentences each for the disconnect framing. Devansh merges into Related Work (D4).

**What our version adds (one sentence for the section close):** We stress-test the statistical-vs-economic gap in a single controlled walk-forward harness against modern shrinkage baselines (Ledoit–Wolf, OAS, EWMA) and document that rank orderings flip across Frobenius / Stein–KL / GMVP — with eigenvalue flooring as an explicit mechanism that can neutralize likelihood losses without helping portfolio ranking.

---

## BibTeX

```bibtex
@article{engle2006testing,
  title   = {Testing and Valuing Dynamic Correlations for Asset Allocation},
  author  = {Engle, Robert and Colacito, Riccardo},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {24},
  number  = {2},
  pages   = {238--253},
  year    = {2006},
  doi     = {10.1198/073500106000000017}
}

@article{fleming2001economic,
  title   = {The Economic Value of Volatility Timing},
  author  = {Fleming, Jeff and Kirby, Chris and Ostdiek, Barbara},
  journal = {The Journal of Finance},
  volume  = {56},
  number  = {1},
  pages   = {329--352},
  year    = {2001},
  doi     = {10.1111/0022-1082.00327}
}

@article{fleming2003economic,
  title   = {The Economic Value of Volatility Timing Using ``Realized'' Volatility},
  author  = {Fleming, Jeff and Kirby, Chris and Ostdiek, Barbara},
  journal = {Journal of Financial Economics},
  volume  = {67},
  number  = {3},
  pages   = {473--509},
  year    = {2003},
  doi     = {10.1016/S0304-405X(02)00259-3}
}

@article{demiguel2009optimal,
  title   = {Optimal Versus Naive Diversification: How Inefficient is the {1/N} Portfolio Strategy?},
  author  = {DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman},
  journal = {The Review of Financial Studies},
  volume  = {22},
  number  = {5},
  pages   = {1915--1953},
  year    = {2009},
  doi     = {10.1093/rfs/hhm075}
}

@article{patton2011volatility,
  title   = {Volatility Forecast Comparison Using Imperfect Volatility Proxies},
  author  = {Patton, Andrew J.},
  journal = {Journal of Econometrics},
  volume  = {160},
  number  = {1},
  pages   = {246--256},
  year    = {2011},
  doi     = {10.1016/j.jeconom.2010.03.034}
}
```

---

## Paste-ready blurbs (1–2 sentences each)

**Engle & Colacito (2006).**
Evaluate covariance / correlation forecasts with an *economic* loss: minimum-variance portfolios and Diebold–Mariano-style tests on realized portfolio volatility, rather than matrix norms alone. They show that correctly specified covariances minimize realized portfolio volatility for any required-return vector, motivating joint statistical and portfolio evaluation.

**Fleming, Kirby & Ostdiek (2001, 2003).**
Demonstrate that volatility / covariance timing can have material economic value for short-horizon mean–variance investors even when statistical explanatory power looks weak, and that realized-volatility estimators can raise that value further. The contrast between modest statistical fit and nonzero economic value is the historical precedent for our disconnect finding — with the reverse polarity in our harness (statistical losses separate methods; portfolios do not).

**DeMiguel, Garlappi & Uppal (2009).**
Show that optimized portfolios using estimated moments often fail to beat naive \(1/N\) out of sample, underscoring that estimation error can erase the economic gains implied by in-sample covariance fit. We position our GMVP parity result in that tradition: sophisticated covariance estimators need not improve portfolio outcomes once evaluated honestly OOS.

**Patton (2011).**
Shows that volatility forecast rankings are not robust to the choice of loss function when the volatility proxy is imperfect; different losses can reorder methods. Our covariance results extend that caution to the matrix setting: Frobenius, Stein/KL, and portfolio variance induce different rankings of the same forecasters.

---

## Suggested Related Work paragraph (optional assemble)

Forecast evaluation for risk models has long distinguished statistical loss from economic value. \citet{engle2006testing} value dynamic correlations through minimum-variance portfolio outcomes; \citet{fleming2001economic,fleming2003economic} show volatility timing can pay even when statistical fit is limited. At the same time, \citet{demiguel2009optimal} document that estimated optimal portfolios often fail to beat \(1/N\) out of sample, and \citet{patton2011volatility} shows that imperfect proxies make loss-function choice first-order for rankings. We contribute a controlled walk-forward comparison of a regime-aware similarity forecaster against modern shrinkage baselines in which statistical losses separate methods by orders of magnitude while portfolio outcomes do not, and we identify eigenvalue flooring as one mechanism that can neutralize likelihood losses without repairing GMVP rankings.
