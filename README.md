# XU100 regime detection: a jump-diffusion risk monitor

A daily risk monitor for Borsa Istanbul's XU100 index. After each BIST close it produces:

- **1-day VaR and ES** for XU100 in TRY, XU100 in USD and USDTRY
- **10-day VaR and ES** for XU100 in TRY and in USD (USDTRY 10-day failed its test and is not published)
- **A stress flag**, with equity-only, FX-only and co-jump probabilities for the day

It models XU100 in TRY and USDTRY **jointly**. A TRY-only monitor misreads lira crises: in Nov–Dec 2021 USDTRY rose about 80% while XU100 in TRY made new highs. Log returns satisfy r_USD = r_TRY − r_FX exactly, so the USD-investor view falls straight out of the bivariate model.

The readings are published at **https://marketpanel.breezeblocks.workers.dev** (`#/risk`).

> **[`DESIGN.md`](DESIGN.md) is the source of truth.** It holds every decision (§0), every pass/fail criterion (each fixed *before* its study was run), the open questions (§12) and the full results log (§13). This README is a summary.

## Models

The project began as a benchmark ladder: GBM → Merton jump-diffusion → Markov-switching GBM → Markov-switching jump-diffusion (MS-JD). Each step had to earn its parameters out of sample. The evidence moved it elsewhere:

| Model | Outcome |
|---|---|
| **MS-JD** (K regimes, Poisson jumps, bivariate Gaussian) | Failed the simulation gate. Its regimes were mostly standing in for GARCH volatility clustering, and GARCH-t beat it out of sample. |
| **MS-GARCH-J** (per-regime GJR variance with jumps) | BIC picks K = 1 on real data. Regimes on top of GARCH aren't identifiable from 18 years of daily returns. K = 1 ties GJR-t and has better-calibrated tails. |
| **TVTP-MSGJ2** (transitions driven by Turkey 5Y CDS) | Passes 3 of 4 simulation criteria but fails 3 of 5 real-data criteria. The two regimes don't split on CDS stress. |
| **MSGJ1-t** (K = 1, GJR + jumps, Student-t innovations) | **In production for VaR/ES.** |
| **MSGJ1** (Gaussian version) | **In production for the stress flag** (GJR-J next-day sd of XU100 in USD, 92.5th percentile, 3-session persistence). |

Benchmarks are GARCH-t, GJR-t (the reference), DCC-t, EWMA, historical simulation and a 20-day realised-vol flag (RV20).

The likelihood is exact at daily frequency: a Poisson-weighted Gaussian (or t) mixture run through a Hamilton filter in JAX. It is maximised by multi-start L-BFGS with exact gradients. Filtering is real-time only (no look-ahead), and smoothed probabilities are used only for diagnostics.

## Results

Sample: 2008-08-01 onward (the start of continuous BFIX FX fixed at the BIST close). Splits were fixed before any real-data fitting:

| Segment | Period | Role |
|---|---|---|
| Burn-in | 2008-08 → 2012-12 | first estimation window |
| Validation | 2013 → 2019 | all tuning and model selection |
| Holdout | 2020 → 2026-09-25 | run **once**, on 2026-09-28 |

**Holdout (MSGJ1): 2 of 4 pre-registered criteria pass.**

| Criterion | Result | |
|---|---|---|
| VaR coverage (≤ 2 of 12 tests reject) | 2 of 12 (GJR-t: 7 of 12) | pass |
| ES Z2 at 2.5% | fails on XU100 TRY (p = 0.008) | fail |
| Log score vs GJR-t | loses on USDTRY (DM −2.47) and USD (−3.03); thin Gaussian far tails, including the 2021-12-21 KKM day | fail |
| Stress flag (≥ 4 of 7 events, ≤ 1 false alarm/yr) | **7 of 7 events**, 0.30 false-alarm episodes/yr, median lag 2 sessions | pass |

The Student-t version (MSGJ1-t) fixes both failures and has since replaced MSGJ1 for VaR/ES. The holdout motivated that switch, so its holdout result is **not blind**, and every daily report says so.

Other results worth knowing:

- **10-day VaR/ES** fails on USDTRY (too conservative in validation, ES far too small in the holdout). It ships for XU100 only, with a live kill rule.
- **Robustness battery:** 12 of 16 checks pass. The leverage and equity–FX correlation signs hold in every refit. The jump component is **not statistically supported** (bootstrap p = 0.08), so "drop jumps" is logged as a v2 candidate.
- **Stress detection in validation:** the GJR-J volatility signal has AUC 0.90. Regime probabilities (AUC 0.57) and CDS signals (AUC 0.55–0.82, late) are ruled out.

Result tables are in [`experiments/results/*_summary.md`](experiments/results/) and [`event_scoring.md`](experiments/results/event_scoring.md).

## Repository layout

```
src/
  data.py          data pipeline: vendor CSVs / yfinance / EVDS → data/processed/returns.parquet
  model.py         likelihood, JAX Hamilton filter, Kim smoother, simulator, predictive distributions,
                   MS-GJR variance, TVTP transition paths, multi-day horizon simulation
  fit.py           Spec(K, jumps, garch, n_cov, t), constrained pack/unpack, multi-start L-BFGS
  daily.py         daily runner: writes reports/daily/<date>.md and appends reports/daily.csv
  charts.py        maintains reports/history.csv (one real-time row per session since 2013) and the charts
  site_export.py   builds the website feed (derived numbers only, never raw vendor prices)
scripts/daily.sh   scheduled run: refresh → staleness check → daily.py → site export
experiments/       one script per study (simulation gate, benchmarks, event scoring, TVTP, holdout,
                   Student-t, 10-day, robustness) and their cached results
tests/             plain-assert checks: data, model (brute-force and independent implementations), bench, daily
DESIGN.md          design, criteria and results log
```

## Setup

Python 3.13. Install the dependencies:

```bash
pip install jax numpy scipy pandas statsmodels matplotlib pyarrow python-dotenv yfinance evds
```

API keys go in a git-ignored `.env` at the repo root:

```
EVDS_API_KEY=...   # CBRT EVDS, FX fallback
FRED_API_KEY=...
```

### Data

`data/` and `reports/` are **not in the repository**. The preferred inputs are vendor exports, dropped by hand into `data/raw/` as CSVs (first column date, second column close):

- `USDTRY.csv`: USDTRY sampled at the BIST close (Bloomberg BFIX 18:00 GMT+3), not a 24-hour close
- `bbg_xu100.csv`: XU100 Index PX_LAST (optional)

Without them, `data.py` falls back to yfinance for XU100 and EVDS (then yfinance) for USDTRY. The fallback FX is not time-matched to the equity close, which creates spurious jumps (DESIGN §2, §5), so treat results built on it as indicative.

## Usage

```bash
python3 src/data.py              # build data/processed/returns.parquet (--refresh re-downloads)
python3 src/daily.py             # write today's report
scripts/daily.sh                 # the scheduled pipeline (--force rebuilds an existing report)
```

`daily.sh` only writes a report when the last session's USDTRY print is current. A stale FX file would put a fake zero FX return into the forecast. Don't run `--refresh` during BIST hours, because it saves a partial bar as the day's raw snapshot.

Tests:

```bash
python3 tests/test_data.py
python3 tests/test_model.py      # ~1 min
python3 tests/test_bench.py
python3 tests/test_daily.py      # checks the runner reproduces the evaluated holdout numbers
```

Experiments resume from the cached `*.pkl` files in `experiments/results/`. Long runs use process pools; to stop one, run `pkill -f <script>.py; pkill -f multiprocessing.spawn`.

## Research rules

- The holdout (2020+) was touched once. `experiments/holdout.py` refuses to run again.
- Every pass/fail criterion was written into `DESIGN.md` before its study ran and has never been moved afterwards. Studies motivated by holdout results are labelled "not blind".
- Data anomalies are investigated by hand and logged, never silently fixed.

## Limitations

- Jumps are not statistically supported over a jump-free GJR-t; they mainly act as a fat-tail device.
- FX variance persistence hits the 0.999 cap in many post-2020 refits, so FX variance is close to integrated.
- The equity–FX correlation fell from −0.50 (2008–19) to about 0 (2020–26), while the fitted ρ stays negative.
- Sudden political shocks (e.g. the 2016 coup attempt) give volatility-based signals no warning.
- Only 4 listed stress events fall in validation, so differences between detection signals there are not significant.

This is a research tool, not investment advice.

## License

[MIT](LICENSE). The license covers the code only. It does not cover vendor data (Bloomberg, Matriks), which is not in the repository.
