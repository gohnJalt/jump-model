# §8.6 10-day VaR/ES (MSGJ1-t Monte Carlo)

Sum of the next 10 sessions' log returns, lower tail. 100,000 paths per origin. Tests: Newey–West HAC, 18 lags. Hit-rate p is two-sided. DM > 0 means MSGJ1-t has the lower FZ0 loss.

## Validation 2013–2019 (blind): 1746 overlapping windows (2013-01-02 → 2019-12-17)

| Target | α | Model | Hit rate | p (HAC) | mean loss beyond VaR | mean ES |
|---|---|---|---|---|---|---|
| E | 1.0% | ms | 1.15% | 0.792 | -13.47% | -14.87% |
| E | 1.0% | sqrt10 | 1.15% | 0.779 | -13.22% | -16.18% |
| E | 1.0% | gjr | 1.60% | 0.304 | -12.24% | -14.21% |
| E | 2.5% | ms | 3.84% | 0.137 | -10.27% | -12.07% |
| E | 2.5% | sqrt10 | 4.07% | 0.095 | -10.14% | -12.65% |
| E | 2.5% | gjr | 4.81% | 0.022 | -9.72% | -11.48% |
| F | 1.0% | ms | 0.29% | 0.002 | -5.38% | -6.39% |
| F | 1.0% | sqrt10 | 0.17% | 0.000 | -5.90% | -7.43% |
| F | 1.0% | gjr | 0.29% | 0.002 | -5.38% | -6.94% |
| F | 2.5% | ms | 1.32% | 0.016 | -4.45% | -5.44% |
| F | 2.5% | sqrt10 | 0.57% | 0.000 | -4.75% | -6.16% |
| F | 2.5% | gjr | 0.92% | 0.000 | -4.22% | -5.88% |
| USD | 1.0% | ms | 1.72% | 0.227 | -19.24% | -20.45% |
| USD | 1.0% | sqrt10 | 1.66% | 0.234 | -18.90% | -22.02% |
| USD | 1.0% | gjr | 2.29% | 0.046 | -17.12% | -20.16% |
| USD | 2.5% | ms | 4.64% | 0.051 | -15.35% | -16.85% |
| USD | 2.5% | sqrt10 | 4.98% | 0.025 | -15.02% | -17.46% |
| USD | 2.5% | gjr | 6.07% | 0.004 | -14.05% | -16.37% |

| Target | FZ0 DM vs GJR-t (2.5%) | FZ0 DM vs √10 (2.5%) |
|---|---|---|
| E | 1.64 | 1.13 |
| F | 3.52 | 4.02 |
| USD | 1.64 | 1.00 |

Coverage rejections at 5% (of 6; no hits counts as a rejection): MSGJ1-t 2, √10 3, GJR-t 5.

## Holdout 2020+ (already seen): 1680 overlapping windows (2020-01-02 → 2026-09-11)

| Target | α | Model | Hit rate | p (HAC) | mean loss beyond VaR | mean ES |
|---|---|---|---|---|---|---|
| E | 1.0% | ms | 2.02% | 0.149 | -14.85% | -16.60% |
| E | 1.0% | sqrt10 | 1.73% | 0.269 | -15.25% | -18.27% |
| E | 1.0% | gjr | 2.92% | 0.025 | -13.66% | -16.28% |
| E | 2.5% | ms | 4.58% | 0.064 | -12.27% | -13.32% |
| E | 2.5% | sqrt10 | 4.52% | 0.074 | -12.25% | -14.25% |
| E | 2.5% | gjr | 5.12% | 0.033 | -11.84% | -12.93% |
| F | 1.0% | ms | 0.89% | 0.826 | -12.64% | -7.05% |
| F | 1.0% | sqrt10 | 0.48% | 0.129 | -10.07% | -7.42% |
| F | 1.0% | gjr | 0.60% | 0.259 | -7.18% | -5.61% |
| F | 2.5% | ms | 1.79% | 0.324 | -10.04% | -5.43% |
| F | 2.5% | sqrt10 | 1.55% | 0.130 | -9.66% | -5.55% |
| F | 2.5% | gjr | 1.79% | 0.311 | -7.88% | -4.53% |
| USD | 1.0% | ms | 1.49% | 0.440 | -20.25% | -22.88% |
| USD | 1.0% | sqrt10 | 1.55% | 0.402 | -19.96% | -24.63% |
| USD | 1.0% | gjr | 2.02% | 0.176 | -18.86% | -22.17% |
| USD | 2.5% | ms | 3.21% | 0.436 | -16.57% | -18.40% |
| USD | 2.5% | sqrt10 | 4.46% | 0.063 | -14.87% | -18.94% |
| USD | 2.5% | gjr | 4.52% | 0.066 | -15.27% | -17.57% |

| Target | FZ0 DM vs GJR-t (2.5%) | FZ0 DM vs √10 (2.5%) |
|---|---|---|
| E | 1.82 | -0.62 |
| F | -7.81 | -5.00 |
| USD | 1.14 | 1.84 |

Coverage rejections at 5% (of 6; no hits counts as a rejection): MSGJ1-t 0, √10 0, GJR-t 2.

## Pre-registered checks (§8.6)

| # | Check | Result | Pass |
|---|---|---|---|
| T1 | validation coverage, ≤ 1 of 6 rejections | 2 | NO |
| T2 | validation FZ0 DM vs GJR-t > −1.96 on all targets | E 1.64, F 3.52, USD 1.64 | yes |
| H | holdout: rejections ≤ GJR-t's (0 vs 2) and T2 | E 1.82, F -7.81, USD 1.14 | NO |

**Fails: 10-day VaR/ES stays out of the daily report.**
