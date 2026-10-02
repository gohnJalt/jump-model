# XU100 Regime Detection with a Markov-Switching Jump-Diffusion

**Status:** Draft v0.1 · 2026-09-25
**Scope:** Daily risk monitoring for Borsa Istanbul XU100. The design comes from a statistics and economics view first; engineering follows from it.

---

## 0. Decisions so far

| # | Decision | Choice | Rationale (short) |
|---|---|---|---|
| D1 | Purpose | **Risk monitoring** | Real-time *filtered* regime probabilities, a stress flag, and VaR/ES. Smoothed probabilities are only used for diagnostics. |
| D2 | Frequency | **Daily** | Long history. Enough observations to separate jumps from diffusion. |
| D3 | Target series | **Bivariate: XU100 in TRY + USDTRY, one shared latent regime** | Separates equity stress from FX stress. The USD view follows exactly, and TRY-only is blind to FX crises (§2). |
| D4 | Model family | **Benchmark ladder with MS-JD as the core** | GBM → Merton JD → MS-GBM → MS-JD. Each step must earn its parameters out-of-sample. |
| D5 | Number of regimes K | **Chosen from the data, 2–4** | Out-of-sample predictive score plus bootstrap LR, and every regime must be economically interpretable. |
| D6 | Estimation | **MLE (EM start → quasi-Newton polish), parametric-bootstrap CIs** | Exact likelihood via the Hamilton filter. Fast enough for large tuning grids. |
| D7 | Sample | **Revised 2026-09-25: 2008-08-01 onward (continuous BFIX FX)**. Originally ~1997+. | Time-matched FX only exists from there. Splicing EVDS before that would put a measurement-quality break into the data that the model could read as a regime change (§5.3). About 12 stress episodes remain. |
| D8 | Covariates | **Validation only in v1.** TVTP is a v2 candidate. | Keeps the regimes price-identified and avoids mixed-frequency and publication-lag leakage. |
| D9 | Ground truth | **Curated event list (§7.4), evaluation only** | Never used in fitting. |
| D10 | Stack and data | Python; FRED + EVDS keys; **Bloomberg + Matriks** | Paid data is needed for time-matched FX, CDS/EMBI, and intraday jump validation. |
| D11 | Pre-float FX (1997 → 22 Feb 2001) | Deterministic indicator. **Inactive since D7 was revised** (no pre-float data in the sample). The code path stays tested. | A known, announced policy regime must not use up a latent state. |
| D12 | Production VaR/ES model | **MSGJ1-t (2026-09-28)**; the flag stays on MSGJ1 | Passed §8.5. The switch came after the holdout, so its holdout result is not blind; this is stated in every daily report. |
| D13 | Website "when stress was here before" table (2026-10-02) | Spells since 2013 with the stress percentile within ±5 of today's level (gaps ≤ 10 sessions merged); XU100 TRY/USD moves 1w/1m/3m after entry and whether the flag turned on within 21 sessions. **Uses 2020+ data**, labelled by period | Owner-approved exception to the holdout-exclusion rule for descriptive stats: the holdout was spent on 2026-09-28 and its returns are already published on the site. Descriptive only; not a test and not used for any model decision. |

Items still open are listed in **§12**. Nothing marked *Proposed* is final until you sign off.

---

## 1. Problem statement

Build a daily model that produces the following each evening after the BIST close:

1. **P(S_t = k | F_t)**: the filtered probability of each latent market regime, using information up to and including day t only.
2. **A stress flag** from those probabilities, with a tuned threshold and persistence rule.
3. **Jump posteriors**: the probability that day t contained an equity-only jump, an FX-only jump, or a co-jump.
4. **Forward risk**: 1-day VaR/ES (v1; 10-day deferred 2026-09-28 until a downstream user asks for it) for XU100 in TRY, XU100 in USD, and USDTRY, using the regime-mixture predictive distribution.

The model has to answer three questions for a risk desk:
- Are we in a stress regime?
- Is the stress coming from equities, from the lira, or from both?
- How big are the tails right now?

We work only under the physical measure P. There is no option data, so there is no risk-neutral calibration.

---

## 2. Why bivariate TRY + FX (currency choice)

| Option | Main failure |
|---|---|
| TRY nominal | **FX crises reverse its sign.** Equities hedge inflation and FX, so XU100 in TRY rallies in lira collapses. In Nov–Dec 2021 USDTRY rose about 80% peak to trough while XU100 in TRY hit new highs. A TRY-only monitor would read that as a calm bull regime. |
| USD univariate | Mixes two processes and can't attribute a jump to equities or FX. Daily FX snapshots taken at a different time from the BIST close create **spurious jumps**. |
| Real TRY (CPI-deflated) | CPI is monthly, so the daily deflator is a smooth interpolation. Volatility and jump content are unchanged; only the drift is relabelled. It also depends on disputed TÜİK CPI. Not used for modelling. |
| **Bivariate (chosen)** | About twice the parameters, and co-jump identification is harder. Affordable with ~7,000 daily observations. |

The identity r^USD_t = r^E_t − r^F_t is exact in log returns. The USD-investor distribution therefore comes straight from the bivariate model, with no second model.

---

## 3. Model specification

### 3.1 State

- **Latent regime:** S_t ∈ {1,…,K}, a continuous-time Markov chain (CTMC) with generator **Q**. Off-diagonals are q_ij ≥ 0 and rows sum to 0.
  - Expected duration of regime k is 1/|q_kk|.
  - *Proposed:* full Q for K ≤ 3, and tridiagonal (birth–death, calm ↔ elevated ↔ crisis) for K = 4 to keep it identified. Both are tested.
- **Observed log-prices:** X_t = (x^E_t, x^F_t)′, where x^E = log XU100 (TRY) and x^F = log USDTRY.

### 3.2 Dynamics

```
dX_t = μ(S_t) dt + Σ(S_t)^{1/2} dW_t + dJ^E_t e₁ + dJ^F_t e₂ + dJ^C_t
```

| Component | Intensity | Jump size | Regime-dependent? |
|---|---|---|---|
| Diffusion | — | Gaussian, cov Σ(k) (σ_E, σ_F, ρ) | **Yes** |
| Equity-only jumps J^E | λ_E(k) | N(m_E, s_E²) | Intensity yes, size **no** (baseline) |
| FX-only jumps J^F | λ_F(k) | N(m_F, s_F²) | Intensity yes, size **no** (baseline) |
| Co-jumps J^C | λ_C(k) | N₂(m_C, V_C) | Intensity yes, size **no** (baseline) |
| Drift | μ(k) | — | Tested (H6) |

**Why regime-invariant jump sizes in the baseline:** a regime can raise diffusion volatility, jump frequency, or jump size. Letting all three switch makes the likelihood nearly flat along several ridges. We fix sizes and let intensities switch, then test regime-dependent sizes as a nested alternative (H3).

**Parameter count (K = 3):** 8 per regime (μ×2, Σ×3, λ×3) × 3, plus 9 shared jump-size parameters, plus 6 in Q, for 39 in total, against about 7,000 bivariate observations.

### 3.3 Discretization (exact likelihood at daily frequency)

- **Time unit:** Δ = 1 trading session. All reported parameters are annualized ×252.
- **Within-day approximation:** the regime is constant within a session. The error is O(q·Δ), which is negligible when expected durations are much longer than a day. We verify this in simulation (§6.1).
- **Transition matrix:** P = expm(QΔ), computed once per likelihood evaluation.
- **Conditional density.** Given S_t = k, the daily increment is a Poisson-weighted Gaussian mixture over jump counts (n_E, n_F, n_C):
  - mean = μ(k)Δ + n_E m_E e₁ + n_F m_F e₂ + n_C m_C
  - cov = Σ(k)Δ + n_E s_E² e₁e₁′ + n_F s_F² e₂e₂′ + n_C V_C
  - weight = Pois(n_E; λ_E(k)Δ) · Pois(n_F; λ_F(k)Δ) · Pois(n_C; λ_C(k)Δ)
- **Truncation:** N_max is chosen **adaptively per component** as the smallest value whose omitted Poisson mass is below 1e-6 at that component's largest intensity. That is 4 at the λ ≤ 25/yr bound; the earlier figure of 3 left a tail of about 4e-6.
- **Multi-session gaps:** holidays, the Feb 2023 earthquake closure, and the pre-2000s split sessions are handled as follows. Δ_t is the number of sessions spanned. *Proposed:* test a calendar-time alternative on gap days (H8).

### 3.4 Constraints and regularization

These are statistical necessities, not convenience choices:

- **Unbounded likelihood:** Gaussian mixtures let a component variance collapse onto one observation and send the likelihood to infinity. We impose lower bounds σ_·(k) ≥ σ_min and s_· ≥ s_min, set from the empirical 1st percentile of rolling volatility.
- **Jump / diffusion aliasing:** as λ → ∞ with s → 0, jumps become diffusion. We bound λ ≤ λ_max (25/yr per component) and require s_· ≥ c · min_k σ_·(k)√Δ, so jumps are larger than a typical calm-day move. c is a tuning parameter (§8).
- **Label switching:** after each fit, regimes are ordered by equity diffusion variance plus the jump-variance contribution.
- **Reparameterization for the optimizer:** log for positive parameters, logit for bounded ones, Fisher-z for ρ, and log for q_ij.

---

## 4. Estimation

### 4.1 Filtering and likelihood
- Hamilton filter, run in log-space with scaling, written in **JAX** (`lax.scan`, jit, float64). Gradients come from autodiff, so they are exact.
- The log-likelihood is the sum of log one-step predictive densities, and it is exact given the discretization.
- The initial regime distribution is the stationary distribution of P.

### 4.2 Optimization
1. **EM (ECM variant) for starting values.**
   - E-step: smoothed joint posteriors over (regime, jump counts) using the forward–backward (Kim) smoother.
   - M-step: closed-form updates for weights, intensities and jump means. Variances need a conditional maximization step, because diffusion and jump variances add.
2. **Polish:** L-BFGS-B on the exact log-likelihood, starting from the EM solution. Optimization runs on **standardized returns (y/sd)** and the estimates are rescaled afterwards. On raw units L-BFGS stalled (measured in Phase 2).
   - *Phase 2 status:* L-BFGS only, from a data-driven start plus random perturbations. EM starting values are deferred until the §6.1 recovery study shows they are needed. Measured cost: about 15 s per start at K = 3, T = 7,300, against 10–14 min with numerical gradients.
3. **Multi-start:** at least 50 random starts per specification (tuning parameter). We report how many distinct local optima we find and the log-likelihood gap between them. A local optimum within 2 log-lik units of the global one is a warning sign.

### 4.3 Uncertainty
- **Primary:** a parametric bootstrap (B = 500). Simulate from the fitted model, refit, and read off percentile CIs. This also gives the finite-sample bias of λ and the q_ij.
- **Quick check:** a sandwich (QML) covariance from the numerical Hessian and outer product of scores. A large gap between the sandwich and Hessian-only SEs signals misspecification.
- **Regime-probability uncertainty:** we propagate bootstrap parameter draws through the filter to get bands around P(S_t | F_t).

### 4.4 Real-time protocol (no look-ahead)
- At date t, parameters come only from data up to t.
- **Expanding window, refit every R sessions.** *Proposed:* R = 21, about monthly. R is a tuning parameter.
- Between refits, parameters are frozen and only the filter runs forward.
- A rolling window is tested as an alternative (§8) because Turkey's structural breaks are large.

---

## 5. Data

### 5.1 Series

| Series | Primary source | Cross-check | Notes |
|---|---|---|---|
| XU100 close (TRY) | Bloomberg (`XU100 Index`) | yfinance `XU100.IS`, EVDS BIST series | 27 Jul 2020 rebase ÷100 → check continuity. Price index; total-return variant considered (§12). |
| USDTRY | **Bloomberg, time-matched to the BIST close** (BFIX fixings or intraday; history depth to confirm, §12 Q1) | CBRT indicative rate (EVDS, ~15:30), yfinance `TRY=X` | 2005 redenomination (1 TRY = 1,000,000 TRL) → rescale. |
| Intraday XU100 and USDTRY (validation only) | Matriks (BIST intraday), Bloomberg | Each other | Nonparametric jump tests (§7.1). |
| **Validation covariates** | | | Codes confirmed at implementation |
| CBRT policy / funding rate, reserves, CPI | EVDS | — | Monthly/weekly; publication lags recorded |
| VIX, broad USD index, US 2Y/10Y, Fed funds | FRED | — | |
| Turkey 5Y CDS | Bloomberg | — | Key stress covariate |
| EMBI Turkey spread | Bloomberg | — | If available |

### 5.2 Alignment and cleaning
- **Calendar:** BIST trading sessions. FX returns are measured over exactly the same interval as equity returns.
- **Synchronicity:** FX must be sampled at the BIST close. The close time has changed over history (session structure, closing auction), so we keep a table of close times by date.
  - If time-matched FX isn't available for part of the sample, we flag those dates and run a sensitivity check.
  - An asynchronous FX snapshot shows up as spurious negative autocorrelation in r^F and inflated co-jump intensity. We test for both.
- **Level breaks:** scan for |r| > 8% on days that have no matching event in §7.4. Every such day is investigated by hand and logged in `data/ANOMALIES.md`, never silently fixed.
- **Circuit breakers and price limits:** BIST's stock-level price limits and index-level halts (e.g. March 2020, March 2025) can **spread a jump over several days**. That looks like a burst of high diffusion volatility, not a jump. We tag these days so the jump posteriors on them are read with care.
- **Short-selling bans** (2020, 2023, 2025) are tagged as a validation covariate.
- **Reproducibility:** raw pulls are stored as immutable parquet with the fetch timestamp and source. Processing is a pure function of raw data plus config.

### 5.3 Known structural issues in the sample (econometric, not engineering)

| Period | Issue | Proposed handling |
|---|---|---|
| 1997–Feb 2001 | Crawling peg / exchange-rate-based stabilization. FX variance is administratively suppressed and the drift is pre-announced. | **Decided (D11):** a deterministic pre-float indicator D_t = 1{t < 2001-02-22} shifts the FX drift and FX diffusion variance only, with separate parameters (μ_F^peg, σ_F^peg) that are the same in every regime. Equity and jump parameters are unaffected. Sensitivity check: start in 2001-03. |
| 1997–2004 | Inflation of 30–100%, so the TRY equity drift is large | Drift is weakly identified anyway (SE of the annual drift ≈ σ/√T ≈ 6–7%/yr over the full sample). H6 tests regime-invariant drift. |
| 2021–2024 | Unorthodox policy, KKM, managed FX, reserve sales | FX variance artificially low. Left to the latent chain, with a flag in the interpretation. Stability test H7. |
| Foreign ownership fell from ~65% to ~30–40% of the float | Investor-base change shifts the equity dynamics | Rolling-window sensitivity (§8). |

---

## 6. Validating the model before real data

### 6.1 Simulation study (gate before touching real data)
1. **Parameter recovery:** simulate from MS-JD with parameters close to the expected estimates (T = 7,000, K = 2, 3). Report the bias and RMSE of every parameter, regime classification accuracy (filtered and smoothed), and jump-detection precision and recall.
2. **Misspecification probes.** Fit MS-JD to data it didn't generate:
   - GARCH(1,1)-t data: does MS-JD invent spurious regimes or jumps? It probably will. **We need to know how badly.**
   - Pure Merton JD (K = 1): does the K-selection procedure correctly pick K = 1?
   - Asynchronous-FX contamination: how much does fake co-jump intensity inflate?
3. **Discretization error:** simulate at a fine grid (1/100 session) and estimate on daily aggregates to measure the within-day-constant-regime approximation.

### 6.1a Go/no-go thresholds (fixed 2026-09-25, **before** any simulation results)
True parameters come from a calibration fit on burn-in + validation data (2008-08 → 2019-12, BFIX FX). The holdout is never touched. R replications at T = 4,544, the production sample length. *The sample and T were revised with D7; the thresholds below are unchanged, and no simulation results had been examined.*

| Criterion | Pass if |
|---|---|
| Diffusion sd, regime durations (1/\|q_kk\|) | \|relative bias\| < 15% |
| Jump intensities λ | \|relative bias\| < 30% **and** RMSE < 60% of the true value |
| Filtered regime classification accuracy | ≥ 85% |
| Jump-day precision (filtered posterior > 0.5) | ≥ 70% |
| K selection when data come from the model | Correct K ≥ 80% of the time |
| GARCH(1,1)-t data (no regimes) | Selected K ≤ 2, i.e. no more than 2 spurious regimes |

If a criterion fails, the fix is a simpler spec (e.g. dropping a jump component or pooling intensities), re-tested against the same thresholds. The thresholds themselves don't move.

### 6.1b TVTP simulation criteria (fixed 2026-09-27, **before** any TVTP results)
- **Truth:** the calibrated K = 2 MS-GARCH-J (the weak regimes that returns alone can't identify), plus TVTP.
- **Covariate:** exogenous, CDS-like AR(1) with φ = 0.995, standardized.
- **Transition effect:** δ = (+δ* calm→stress, −δ* stress→calm).
- **Scenarios:** null δ* = 0, weak δ* = 0.75, strong δ* = 1.5. R = 20 each, T = 4,544.
- **Fits:** static MSGJ2 vs TVTP MSGJ2 on the same data.

| Criterion | Pass if |
|---|---|
| Size: LR test of δ = 0 (χ²₂, 5%) under the null | Rejection rate ≤ 10% |
| Power: the same test under strong | Rejection rate ≥ 80% |
| δ recovery under strong | \|relative bias\| < 30% for each δ |
| Regime identification under strong | TVTP filtered accuracy ≥ static + 5 pp (mean) |

The weak scenario is reported only.

### 6.2 Code-correctness checks
- With K = 1 and no jumps, the likelihood equals the bivariate Gaussian log-likelihood.
- On tiny T, the filter likelihood equals brute-force summation over all regime paths.
- The MS-GBM special case matches `statsmodels` `MarkovRegression` (univariate, switching variance) on the same data, up to optimizer tolerance.

---

## 7. Evaluation framework

Evaluation runs on **filtered** output in the out-of-sample periods (§8.1). There are four tracks, and a spec has to be acceptable on all four.

### 7.1 Statistical fit
- **Out-of-sample predictive log score.** Specs are compared with Amisano–Giacomini / Diebold–Mariano tests (HAC variance).
- **PIT diagnostics:** Berkowitz LR test on the probability integral transforms. Ljung–Box on PITs and squared PITs; leftover ARCH is the main suspected failure mode (§10).
- **Jump validation:** with paid intraday data, compare model jump posteriors against nonparametric jump days (Barndorff-Nielsen–Shephard bipower test, Lee–Mykland). Report agreement rates.

### 7.2 Risk accuracy (the purpose, D1)
- VaR at 1% and 2.5%, over 1 and 10 days, for three series: XU100 TRY, XU100 USD, and USDTRY.
- 10-day forecasts come from Monte Carlo over the regime-mixture predictive distribution (≥100k paths).
- Tests:
  - Kupiec POF
  - Christoffersen independence and conditional coverage
  - Acerbi–Székely ES test
  - FZ0 joint VaR/ES loss for ranking specs
- **External benchmarks the model must beat or match:** historical simulation, RiskMetrics EWMA, GARCH(1,1)-t, GJR-GARCH-t, and a bivariate DCC-GARCH-t.
  - *If MS-JD doesn't beat GARCH-t on risk, that is a finding, not a failure to hide.*

### 7.3 Regime-detection quality
Measured against the event list (§7.4):
- **Detection lag:** sessions from event onset until the filtered P(stress) exceeds τ.
- **Hit rate**, and **false alarms per year** (alarm days more than 20 sessions from any listed event).
- **Share of time in alarm.** A monitor that is always on is useless.
- **ROC/AUC** of P(stress) against daily event labels.
- **Stability:** switches per year, mean regime duration, and how much the historical filtered path is revised after each refit.
- **Attribution accuracy:** does the equity / FX / joint decomposition match the event type column?

### 7.4 Stress-event list: **approved, frozen in `data/events.csv`**
Evaluation only. Onset dates are approximate and must be checked before the file is frozen. Type: E = equity-led, F = FX-led, EF = joint.

| Onset | Event | Type |
|---|---|---|
| 1997-10 | Asian crisis spillover | E |
| 1998-08 | Russian default contagion | EF |
| 1999-08-17 | Marmara earthquake | E |
| 2000-11 | Banking liquidity crisis | EF |
| 2001-02-19 | Political dispute → 22 Feb float | EF |
| 2001-09 | 9/11 | E |
| 2002-05 | Political crisis, early elections | EF |
| 2003-03-03 | Parliament rejects Iraq troop motion | EF |
| 2006-05 | EM sell-off | EF |
| 2007-04-27 | Presidential crisis / e-memorandum | E |
| 2008-09 | Global financial crisis | EF |
| 2011-08 | US downgrade / euro crisis | E |
| 2013-05-22 | Taper tantrum; 31 May Gezi protests | EF |
| 2013-12-17 | Corruption probe → Jan 2014 emergency hike | EF |
| 2016-07-15 | Coup attempt | EF |
| 2018-08-10 | US sanctions / lira crisis | F (EF in USD) |
| 2020-03 | Covid | EF |
| 2021-03-20 | CBRT governor dismissal | EF |
| 2021-11-23 | Lira crisis → 20 Dec KKM | F |
| 2022-02-24 | Russia invades Ukraine | E |
| 2023-02-06 | Kahramanmaraş earthquakes (BIST closed) | E |
| 2023-05 | Elections → June policy U-turn | EF |
| 2025-03-19 | İstanbul mayor detained | EF |

### 7.5 Economic interpretability
- Covariate profiles by regime: CDS, VIX, policy-rate changes, reserves. The stress regime must show high CDS.
- Lead–lag: do filtered stress probabilities lead or lag CDS moves? This is useful and doesn't require TVTP.
- Every retained regime must have a one-line economic name. A regime we can't name is evidence for a smaller K.

---

### 7.6 TVTP on real data: criteria (**fixed 2026-09-28, before any real-data TVTP result**)
Spec: TVTP-MSGJ2 with one covariate, `z_cds`. It's scored the same way as the ladder in Phase 4 (2013–2019) and in §7.3.
| Check | Pass if |
|---|---|
| Covariate matters | LR of TVTP vs static MSGJ2 on the 2008-08 → 2012-12 fit rejects at **1%** (χ²₂). The level is stricter than usual because §6.1b found the test slightly oversized. |
| Economic sign | δ(calm→stress) > 0 and δ(stress→calm) < 0: higher CDS makes entering stress more likely and leaving it less likely. The median over refits is reported. |
| Forecasts no worse | Log score vs static MSGJ2 and vs GJR-t: DM > −1.96 on all three targets. |
| Detection | §7.3 AUC ≥ 0.75 (MSGJ2 had 0.57), and event cost ≤ the v1 GJR-J usd flag's. |
| Interpretability (§7.5) | Mean CDS is higher in the stress regime (smoothed, validation only). |
If all pass, TVTP-MSGJ2 joins the v1 flag as the regime-identification output. If detection passes but forecasts don't, it's used for detection only.

### 7.7 CDS as a direct detection signal (**fixed 2026-09-28, before any results**)
- **Signals** (real-time; CDS is lagged one session as in `covariates.parquet`; the §7.3 (τ, m) grid; per-event cost, ratio 4):
  - `CDS lvl`: expanding percentile of log CDS.
  - `CDS d20`: expanding percentile of the 20-session change in log CDS.
- **Pass** if either signal meets **one** of these:
  - (a) **Standalone:** AUC ≥ 0.80 **and** event cost ≤ 5 (the GJR-J usd flag's).
  - (b) **Complementary:** OR-ed with GJR-J usd, each at its own chosen (τ, m), it catches an event GJR-J misses (2013-12-17) and adds ≤ 1 false-alarm episode.
- **Disclosure:** CDS levels on 3 event dates were seen during the 2026-09-28 data check. Changes were not looked at.

### 7.8 Choosing the v1 stress flag (**fixed 2026-09-28**)
- **Candidates:**
  - GJR-J usd (the default);
  - RV20;
  - GJR-J usd OR RV20;
  - GJR-J usd OR the passing CDS signal, only if §7.7 passes.
- **(τ, m) for an OR flag:** each component keeps its own individually chosen (τ, m). No joint grid, which would be 441 combinations fitted to 4 events.
- **Rule:**
  - A candidate replaces GJR-J usd only if its event cost is lower by **more than 4** (one missed event's worth) at ratio 4, **and** it is no worse than GJR-J at ratios 2 and 8.
  - Otherwise GJR-J usd stays. On ties the default wins, because it is model-based and comes from the same model as the VaR/ES.
- **Holdout:** the chosen flag is run once. RV20 is reported next to it as a benchmark. Nothing gets switched after seeing the holdout.
- **Disclosure:** this rule was written after seeing the GJR-J and RV20 validation results (§13). For those two it is not a blind test; the holdout is.

## 8. Tuning and experiment protocol

### 8.1 Sample splits (fixed **before** any real-data fitting)

| Segment | Period | Role | Crises inside |
|---|---|---|---|
| Burn-in | 2008-08 → 2012-12 | First estimation window only | 2008, 2011 |
| **Validation** | 2013 → 2019-12 | All tuning and model selection | 2013 ×2, 2016, 2018 |
| **Holdout** | 2020 → present | **Touched once**, for the final chosen spec | 2020, 2021 ×2, 2022, 2023 ×2, 2025 |

*Redrawn 2026-09-25 after D7 was revised. The holdout had never been examined; descriptive stats always excluded it.*

The holdout is run once and logged. If we change the spec after seeing holdout results, the report has to say so explicitly.

### 8.2 Hyperparameter / specification grid

| Parameter | Values | Notes |
|---|---|---|
| K | 1, 2, 3, 4 | Selected by predictive score + bootstrap LR (H1) |
| Jump structure | none / E+F / E+F+C | Nested |
| Jump sizes | regime-invariant / regime-dependent | H3 |
| Drift | regime-invariant / regime-dependent | H6 |
| Q structure | full / birth–death | K ≥ 3 |
| Window | expanding / rolling 5y, 8y, 12y | Structural-break robustness |
| Refit frequency R | 5, 21, 63 sessions | Trade-off: responsiveness vs revision noise |
| Jump-size floor c | 1.5, 2, 3 | Identification of jumps vs diffusion |
| Alarm threshold τ | 0.5 → 0.9 | Tuned on a loss function (§12, Q3) |
| Alarm persistence m | 1, 2, 3 consecutive sessions | Reduces false alarms |

### 8.3 Discipline
- Every run is logged: config hash, data snapshot hash, seed, metrics, and wall time. Logs are append-only.
- **Multiple-testing awareness:** we report the number of configs tried and the full metric distribution, not only the best. The winning spec's advantage is judged against that distribution (reality-check / SPA-style bootstrap over the grid).
- Prefer the simplest spec whose predictive score is statistically indistinguishable from the best. That is the one-standard-error rule applied to log scores.

### 8.4 Holdout protocol (**fixed 2026-09-28**; one run, only with the user's go-ahead)
- **Final spec (v1):** MSGJ1, the bivariate GJR-J with K = 1, used for both the 1-day VaR/ES and the stress flag (GJR-J usd, τ = 0.925, m = 3; the percentile history is expanding from the burn-in).
  - Real-time protocol as in Phase 4: expanding window, a refit every 21 sessions, and the first holdout fit on all data through 2019-12-31.
- **Benchmarks run alongside:** GJR-t (the VaR reference), DCC-t (bivariate), and RV20 (the flag benchmark, §7.8). Nothing gets switched after the run.
- **Reported in full:** log score and DM vs GJR-t; PIT/Berkowitz; VaR/ES at 1% and 2.5% on E, F and USD (Kupiec, Christoffersen, Z2, FZ0 DM); the §7.3 detection table for the 7 holdout events (lags, false-alarm episodes per year, share of days in alarm).
- **Pass criteria:**

| Check | Pass if |
|---|---|
| VaR coverage | Of the 12 Kupiec + Christoffersen tests (2 levels × 3 targets × 2 tests), **at most 2** reject at 5%. Even for a correct model, the chance that at least one rejects is about 46%. |
| ES | Z2 does not reject at 1% on any target, at 2.5%. |
| Forecasts vs the reference | Log score DM vs GJR-t > −1.96 on all three targets. |
| Flag | Catches **≥ 4 of 7** events, with **≤ 1 false-alarm episode a year**. |

- **Known risk, stated beforehand:** in the 2021–24 managed-FX period, FX variance was administratively low (§5.3, §10.4). USDTRY VaR may fail coverage there because it's too wide, and the lira crises may then look like jumps out of calm. Coverage is reported for 2021–24 and for the rest of the holdout separately. The pass criteria use the whole holdout.
- **If it fails:** the report says so. v1 ships with the failure documented, and the spec is not tuned on holdout data.


### 8.5 v2 candidate: Student-t innovations (**fixed 2026-09-28, before any results; the holdout part is NOT blind**)
- **Spec:** MSGJ1-t is MSGJ1 with each Poisson-mixture component a bivariate Student-t(ν) instead of a Gaussian, standardized so the component covariance is unchanged. There's one shared ν in (2.1, 200). This is a t-scale mixture, not the exact convolution of t diffusion and Gaussian jumps. It is a valid density that nests MSGJ1 as ν → ∞. Linear combinations stay t(ν) per component, so the predictive for E, F and USD is a mixture of t's.
- **Scope:** VaR/ES only. The stress flag stays on MSGJ1 (it passed the holdout).
- **Replaces MSGJ1 for VaR/ES only if all of these hold:**

| # | Check | Pass if |
|---|---|---|
| V1 | Burn-in fit (2008-08 → 2012-12): LR of MSGJ1-t vs MSGJ1 | rejects at 1%, χ²₁ (conservative: ν = ∞ is a boundary) |
| V2 | Validation 2013–2019 log score, DM vs MSGJ1 | > 0 on all three targets |
| V3 | Validation VaR/ES | ≤ 2 of 12 Kupiec + CC rejections at 5%, and Z2 at 2.5% p ≥ 1% on every target |
| H | Holdout 2020+ (**already seen**, reported as not blind) | no §8.4 criterion worse than MSGJ1's holdout result |

- A pass here isn't the evidence a clean holdout would give. The holdout was used to *motivate* this candidate, so H can only veto, never confirm.

### 8.6 10-day VaR/ES (**fixed 2026-09-28, before any results; the holdout part is NOT blind**)
- **Spec:** MSGJ1-t (the production VaR model, D12). The target is the sum of the next 10 sessions' log returns for E, F and USD, lower tail, at 1% and 2.5%. Monte Carlo with 100k paths (§7.2). Each simulated day draws Poisson jump counts, then that component's bivariate t(ν) with the model's moments, and the GJR variances update on the simulated shocks. Params come from the real-time refit history (the daily runner's state; no new fits). For K = 1 there are no regime draws.
- **Origins:** every session t. The forecast uses data through t − 1 and covers sessions t … t + 9. Validation origins run from 2013-01-02 to the last one whose window ends by 2019-12-31. Holdout origins run from 2020-01-02 to the last one with 10 realized sessions.
- **Benchmarks:** (a) √10 × the MSGJ1-t 1-day VaR/ES (naive scaling); (b) GJR-t 10-day Monte Carlo, univariate per target, 100k paths, refit on the same 21-session blocks.
- **Overlap:** consecutive windows share 9 sessions, so hits are autocorrelated. The coverage and DM tests use Newey–West HAC with 18 lags. With about 175 independent windows in validation, the 1% tests have little power; that is stated, not fixed.
- **Ships in the daily report only if all of these hold:**

| # | Check | Pass if |
|---|---|---|
| T1 | Validation coverage: HAC t-test of hit rate = α | at most 1 of 6 (3 targets × 2 levels) rejects at 5% |
| T2 | Validation FZ0 joint VaR/ES loss at 2.5%, HAC DM vs GJR-t 10-day | DM > −1.96 on all three targets |
| H | Holdout 2020+ (**already seen**, reported as not blind) | T1 rejections ≤ GJR-t's, and T2 holds |

- **Reported, not pass/fail:** FZ0 DM vs √10 scaling, hit rates per sub-period, and the mean realized loss beyond VaR vs the forecast ES.
- **If it fails:** the result is documented in §13 and the 10-day numbers stay out of the daily report.

### 8.7 10-day VaR/ES for XU100 only (**fixed 2026-09-28, AFTER the §8.6 results were seen; nothing here is blind**)
- **Spec:** as in §8.6, for E (XU100 in TRY) and USD (XU100 in USD) only. USDTRY 10-day is not reported (§13, §8.6 failure). Nothing is refit or re-tuned.
- **This is a post-hoc selection.** Keeping the two series that passed, on the same data that showed they passed, adds no evidence. The checks below only confirm that the §8.6 tests pass on the subset. The only blind test left is live data, so the live kill rule is the part that carries weight.
- **Ships if all of these hold on the existing §8.6 run (`experiments/results/tenday.pkl`):**

| # | Check | Pass if |
|---|---|---|
| P1 | Validation coverage, HAC t-test (as §8.6) | at most 1 of 4 (2 targets × 2 levels) rejects at 5% |
| P2 | Validation FZ0 at 2.5%, HAC DM vs GJR-t 10-day | > −1.96 on E and USD |
| P3 | Holdout (already seen) | coverage rejections ≤ GJR-t's on E and USD, and FZ0 DM vs GJR-t > −1.96 on E and USD |

- **Live kill rule (prospective, the blind part):** from the first report that carries the 10-day numbers, score the non-overlapping windows that start every 10th session. Per series, over the last 25 such windows (about a year), remove the 10-day numbers from the report if **≥ 3 breaches at 2.5%** or **≥ 2 breaches at 1%**. For a correct model, each rule fires by chance with probability 2.4% and 2.6% (binomial, n = 25), about 5% per series for either one. Any breach whose loss is worse than 1.5 × the forecast 1% ES is flagged in the report for review but doesn't remove the numbers.
- **Every report labels the 10-day numbers as not blind** and links this section.
- **Live count start moved to 2026-09-28 (decided 2026-09-28, before any window was scored).** The first 10-day report (as of 2026-09-25) used a stale USDTRY print (`USDTRY.csv` ended 2026-09-24), so that forecast doesn't count.

### 8.8 Robustness and economic-intuition battery (**fixed 2026-09-28, before any results; holdout parts are NOT blind and are reported only**)
- **Model:** MSGJ1-t, the production VaR/ES model. **F19** is its fit on 2008-08-01 → 2019-12-31 (6 starts), the last fit before the holdout. The refit history is the 165 evaluated refits: 84 in validation, 81 in the holdout. Pass/fail uses 2008–2019 only; holdout numbers are shown next to them, labelled not blind. Nothing in production changes on the result. Failures go into §13 and the report's limitations.
- **Multiple testing:** about 40 checks at 1–5%, so one or two false failures are expected even for a correct model. The report says so.

| # | Check | Pass if |
|---|---|---|
| A1 | Optimizer: 20 random starts on the F19 data | ≥ 10 of 20 end within 0.5 loglik of the best, and the best matches the 6-start F19 optimum within 0.5 |
| A2 | Parametric bootstrap: B = 200 series simulated from F19 (same length), each refit (3 starts, warm at F19) | for α, γ, β (both series), ρ and ν, the bootstrap median is within 0.5 × the 90% CI width of the F19 value. Jump parameters are reported only; λ_E and λ_F are expected to be weakly identified. |
| A3 | Refit stability, validation refits | persistence α + γ/2 + β < 1 and 4 < ν < 30 in every refit; at refit dates the 1% 1-day VaR revision (new vs previous params, same day, E/F/USD) has median < 5% and 95th percentile < 15% |
| A4 | 10-day Monte Carlo error: 20 seeds × 10 validation origins | relative sd of the 1% 10-day VaR < 1% for E and USD |
| B1 | Rolling 5-year and 8-year estimation windows vs expanding (validation, 1-day) | DM log score of expanding vs each > −1.96 on E, F and USD |
| B2 | Refit every 63 sessions vs every 21 (validation, 1-day) | DM log score of R = 21 vs R = 63 > −1.96 on E, F and USD |
| B3 | H2, are jumps needed: F19 vs the same model without jumps; bootstrap LR (B = 200 under the no-jump fit) and validation DM | a finding, not pass/fail. **Jumps are supported** if bootstrap p < 5% and DM (jumps vs none) > 0 on ≥ 2 of 3 targets. If not, "drop jumps" is logged as a v2 candidate and nothing is switched. |
| B4 | Sub-period coverage, validation 2013–15, 2016–17, 2018–19 | ≤ 2 of 18 Kupiec tests reject at 5%. Holdout 2020–21, 2022–23, 2024–26 reported. |
| B5 | Jump-size floor c = 1.5 and 3 (vs 2) on the F19 data | reported only (loglik, parameters, 1% VaR on the last day) |
| C1 | Leverage effect: γ_E > 0 | the A2 90% CI excludes 0, and γ_E > 0 in every validation refit |
| C2 | Lira-depreciation asymmetry: γ_F > 0 | as C1 |
| C3 | Equity–FX correlation: ρ < 0 (lira weakness coincides with equity falls) | as C1, with the sign reversed |
| C4 | Co-jump direction: m_C,E < 0 < m_C,F (stocks fall as the lira falls) | holds in F19 and in ≥ 90% of validation refits |
| C5 | Jump days: the 10 days with the highest P(any jump) in 2008-08 → 2019 (F19, in-sample) | ≥ 4 of 10 fall inside an event window [onset − 5, onset + 20) of the frozen `data/events.csv`. The windows cover about 5% of sessions, so by chance P(≥ 4) ≈ 0.1%. |
| C6 | Implied unconditional sd (diffusion + jumps) of E and F vs the 2008-08 → 2019 sample sd | within ±20% on both |
| C7 | News-impact curves and the FX share of XU100-in-USD variance over time | reported only |
| D1 | Berkowitz LR on the validation PITs | p ≥ 1% on E, F and USD |
| D2 | Ljung–Box (10 lags) on z = Φ⁻¹(PIT) and on z² | p ≥ 1% on all 6 |
| D3 | Engle ARCH-LM (5 lags) on z | p ≥ 1% on E, F and USD |
| D4 | VaR hit rate when the stress flag is on vs off | reported only (validation and holdout) |

---

## 9. Hypotheses to test

| ID | Hypothesis | Test | Complication |
|---|---|---|---|
| H1 | Choice of K (e.g. K = 2 vs 3) | Parametric-bootstrap LR + OOS log score | Nuisance parameters unidentified under the null, so the χ² asymptotics fail (Hansen 1992; Garcia 1998). Bootstrap only. |
| H2 | Jumps are needed beyond regimes (MS-GBM vs MS-JD) | Bootstrap LR, OOS score | λ = 0 sits on the boundary and the jump-size parameters are unidentified (Davies problem) → bootstrap |
| H2b | Co-jumps exist (λ_C > 0) | Same as H2 | Same as H2 |
| H3 | Jump sizes differ across regimes | LR (nested, interior) | Standard χ² is acceptable if the parameters are interior |
| H4 | Equity–FX correlation ρ is regime-dependent | LR | — |
| H5 | FX stress leads equity stress | Lead–lag of component-specific jump posteriors; Granger on filtered probabilities | Estimated-regressor problem → bootstrap |
| H6 | Drift differs across regimes | LR / OOS score | Weak power expected |
| H7 | Parameter stability across 2001, 2008 and 2021 | Split-sample LR; rolling-window parameter paths | — |
| H8 | Gap days scale in session vs calendar time | LR on the gap-day variance multiplier | — |
| H9 | Regime durations are exponential (Markov) | Compare smoothed-duration distributions to exponential | Motivates a semi-Markov model if rejected |

---

## 10. Known limitations and risks

1. **Volatility clustering within a regime.** MS-JD has constant volatility inside each regime, while daily EM returns have strong GARCH effects. The likelihood may add regimes just to mimic GARCH, which shows up as flickering and high K.
   - Detection: ARCH left in the PITs, and short durations.
   - Response: MS-GARCH and MS-SVJ are the v2 candidates, and GARCH-t is the benchmark to beat.
2. **Few crises.** We have roughly 20 stress episodes in 28 years, so stress-regime parameters and event-based metrics have wide CIs. Every detection metric is reported with a bootstrap CI.
3. **Price limits and circuit breakers** censor jumps (§5.2). Jump intensity is probably underestimated.
4. **Policy-managed FX periods** understate FX risk in the model exactly when real risk is building (2021 pre-crisis, 2023–24). This is a structural limit of any price-only model and the motivation for TVTP with CDS or reserves in v2.
5. **Filter vs smoother gap.** Filtered probabilities necessarily lag. The honest number to report is filtered detection lag, never smoothed.
6. **Regime ≠ causal state.** Regimes are statistical constructs. Their economic labels are interpretation, backed by §7.5, not identification.

---

## 11. Implementation sketch (minimal)

```
data/raw/          immutable vendor pulls (parquet + fetch metadata)
data/processed/    aligned bivariate returns + tags (gaps, halts, bans, pre-float)
data/events.csv    §7.4 list (frozen after review)
src/data.py        fetch + align + clean
src/model.py       density, filter, smoother, simulate   (JAX)
src/fit.py         EM start, L-BFGS polish, multi-start, bootstrap
src/benchmarks.py  HS, EWMA, GARCH family (via `arch`)
src/evaluate.py    log score, PIT, VaR/ES backtests, detection metrics
experiments/       one config per run + append-only results log
tests/             §6.2 correctness checks
```
**Libraries:** numpy, scipy, pandas, jax, statsmodels (cross-check), arch (GARCH benchmarks), pyarrow, fredapi, an EVDS client or plain requests, yfinance for cross-checks.

### Phases
1. **Data:** fetch, align, anomaly log, and a descriptive-statistics report covering moments, tails, the ACF of r and r², and rough counts of jump days.
2. **Model core and correctness tests** (§6.2).
3. **Simulation study** (§6.1). **This is a go/no-go gate:** if parameters aren't recoverable at T = 7,000, we simplify the spec before touching real data.
4. **Benchmark ladder and GARCH benchmarks** on the validation period.
5. **Tuning grid** (§8) on the validation period.
6. **Single holdout run** and final report.
7. **v2 candidates:** ~~TVTP (CDS, VIX, reserves)~~ (closed, §12 Q11), MS-GARCH / MS-SVJ, semi-Markov durations.

---

## 12. Open questions (need your answers)

| # | Question | Why it matters | Proposed default |
|---|---|---|---|
| Q1 | ~~History depth~~ | **Resolved 2026-09-28:** BFIX USDTRY in `data/raw/USDTRY.csv` starts **2007-03-12**, so the 2008-08-01 sample start is set by continuity, not by availability. Matriks is a live DDE feed with **no history**, so the intraday jump validation (§7.1) has no data and is dropped. | — |
| Q2 | ~~Pre-float FX~~ | **Resolved → D11** | — |
| Q3 | ~~Miss vs false-alarm cost~~ | **Resolved:** a miss is slightly costlier. Encoded as a cost ratio of **1.5 : 1** (miss : false-alarm day) for tuning τ and m, with a sensitivity band of 1.25–2. | — |
| Q4 | ~~Price or total-return XU100~~ | **Resolved 2026-09-28: price index.** Dividends add about 1 bp a day, which only shifts the drift. Tails, regimes and the flag are unaffected, and the project reports risk, not returns. | — |
| Q5 | ~~Event list~~ | **Resolved:** approved and frozen in `data/events.csv` (the `approx` column flags approximate onsets). | — |
| Q6 | ~~Operational use~~ | **Resolved 2026-09-28: end-of-day only.** Intraday would add noise, not signal. D2 stands. | — |
| Q8 | ~~Gradient engine~~ | **Resolved:** JAX (autodiff + jit), about 50x faster. | — |
| Q9 | ~~Next model class~~ | **Resolved:** MS-GARCH-J was built and tested (§13). | — |
| Q10 | ~~Source of the regime signal~~ | **Resolved:** (a) GJR-J stress flag for v1, plus (c) TVTP with covariates as the regime-identification track. (b) is ruled out by §13 event scoring. | — |
| Q3b | ~~Unit of the miss : false-alarm cost~~ | **Resolved 2026-09-28: per event.** The cost is COST_MISS × missed events + 1 × false-alarm episodes. An event counts as caught if a flag is on anywhere in [onset − 5, onset + 20) sessions. Ties go to fewer days in alarm. **Ratio fixed 2026-09-28 at 4** (one caught crisis is worth 4 false-alarm episodes), with 2 and 8 reported as a sensitivity band. The old 1.5 was set per day and doesn't carry over. | — |
| Q11 | ~~Covariate data for TVTP~~ | **Resolved 2026-09-28: TVTP track closed.** The CDS spec failed §7.6 (3 of 5), and CDS alone failed as a flag (§7.7). VIX and reserves are not built: VIX is a weaker, global signal, reserves are weekly or monthly and lagged, and the v1 flag caught 7 of 7 holdout events. `z_cds` ingest stays in `src/data.py covariates()`. | — |
| Q7 | ~~Who consumes the output~~ | **Resolved 2026-09-28: daily report + CSV.** | — |

---

## 13. Results log

### Phase 3: simulation gate (2026-09-26), **FAILED on 6 of 18 checks** (`experiments/results/phase3_summary.md`)
- **Pass:**
  - Diffusion sd and regime durations recover with bias ≤ 6%.
  - ρ and ρ_C recover.
  - BIC picks the true K 98–100% of the time.
  - The within-day regime approximation causes no discretization bias.
  - The async probe confirms that PX_LAST would have biased the estimates: calm-regime equity-jump λ +155%, ρ pulled from −0.48 to −0.35.
- **Fail:**
  - Jump-intensity RMSE is 125–142% of the truth, mainly for rare (1/yr) components.
  - rec3 filtered accuracy is 0.834.
  - rec3 equity jump precision is just under 0.70.
  - rec2 out-of-sample K choice is 78% correct.
  - **GARCH(1,1)-t data with no regimes gets K = 3–4 every time**, with jump intensities at the cap and 14–21 regime switches a year. That's the same signature as the real-data calibration.

### Phase 4: validation-period out-of-sample, 2013–2019 (2026-09-26) (`experiments/results/phase4_summary.md`)
- **Log score:** every ladder model loses to GARCH-t on all three targets (DM −1.5 to −4.2, mostly significant). Bivariate MS-JD3 loses to DCC-t (DM −3.2). GJR-t beats GARCH-t significantly.
- **VaR/ES:**
  - MS-JD2/3 coverage passes Kupiec, Christoffersen and Z2 on all targets. It's better calibrated than GARCH-t on USD at 2.5%, where GARCH-t has 3.4% hits and Z2 p = 0.001.
  - But FZ0 is never significantly better than GARCH-t, and GJR-t has the best FZ0 overall.
- **PIT:** the Berkowitz test rejects every ladder model on E (p ≈ 0), and squared-PIT autocorrelation remains. Both point to volatility clustering left unmodelled inside the regimes.
- **Ladder (H2 preview):** jumps help single-regime models (Merton beats GBM), but once there are 2–3 regimes, jumps add almost nothing (MS-JD3 ≈ MS-GBM3).
- **Reading:** Phases 3 and 4 agree. MS-JD's regimes and jumps are mostly standing in for GARCH-type volatility clustering. Next step pending a decision (§12 Q9).

### MS-GARCH-J (§12 Q9), 2026-09-27
**Real-data calibration** (2008-08 → 2019-12, `experiments/results/calibration_msgj.log`):
- BIC picks **K = 1**: −37,293 vs −37,278 for K = 2 and −37,207 for K = 3.
- K = 1 GJR-J beats MS-JD K = 3 by 87 log-lik units with 18 fewer parameters.
- K = 3 is degenerate: one regime has duration ≈ 0 and ρ = −1.
- Co-jump λ still sits at the cap and at the size floor. With Gaussian innovations, jumps act as the fat-tail device.

**Phase 3 gate, FAILED on 6 of 11 checks** (`phase3_msgj_summary.md`; rec3 dropped because the K = 3 truth is degenerate):
- **GARCH probe, BIC:** picks K = 1 in 24 of 24 runs. BIC doesn't invent regimes once GARCH is in the model.
- **GARCH probe, OOS criterion:** noisy; picks K = 1, 2, 3 in 11, 9 and 4 runs. **Fails.**
- **Merton probe:** passes on both criteria.
- **rec2 (truth = the calibrated K = 2 MS-GARCH-J) fails almost everywhere:**
  - BIC finds K = 2 only 40% of the time.
  - Filtered accuracy is 0.72.
  - Calm-regime FX sd is biased +88%, and duration bias is +43%.
  - β recovers almost exactly; α and γ in the calm regime do not.
- **Reading:** regimes on top of GJR dynamics, at the size the data suggest, **aren't identifiable** from 18 years of daily returns. The BIC choice of K = 1 on real data is reliable in the false-positive direction, but it has low power against a weak K = 2.

**Phase 4, out-of-sample 2013–2019, reference GJR-t** (`phase4_summary.md`):
- **Log score:** MSGJ1, MSGJ2 and MSGJ3 are statistically tied with GJR-t on every target (|DM| ≤ 0.64) and with DCC-t on the bivariate score. Old MS-JD3 lost with DM −2.7 to −4.4.
- **Calibration:**
  - MSGJ2 is the only model that passes Berkowitz on all three targets (p = 0.25, 0.12, 0.21).
  - MSGJ1 passes USD 1% ES (Z2 p = 0.45) where GJR-t fails (p = 0.022).
  - MSGJ2 and MSGJ3 have **significantly lower FZ0 than GJR-t on USDTRY at 2.5%** (DM 2.97 and 2.65). GJR-t is too conservative there, with 1.5% hits.
- **Regime detection against the event list (§7.3) has not been run yet.**

### §7.3 stress detection, validation 2013–2019 (2026-09-27) (`experiments/results/event_scoring.md`)
Only 4 listed events fall in this window, so the differences between signals below aren't statistically significant. The holdout (7 events) is the real test.

| Signal (real-time) | AUC | Hits at the cost-optimal (τ, m) | False-alarm episodes/yr | Share of days in alarm |
|---|---|---|---|---|
| **GJR-J next-day sd, XU100 in USD** (Q10 a) | **0.90** | 2/4 (τ = 95th pct, m = 1) | 0.0 | 1.8% |
| RV20, naive benchmark | 0.89 | 3/4 | 0.29 | 4.4% |
| MS-JD3 P(top regime) | 0.86 | 3/4 | 0.86 | 2.3% |
| GJR-J max(E, F) | 0.85 | 1/4 | 0.0 | 1.1% |
| MSGJ2 P(stress) (Q10 b) | 0.57 | 2/4 | 0.43 | 0.7% |

- MSGJ2's regime probability is close to useless for detection (AUC 0.57), which matches its Phase 3 unidentifiability. **That rules out Q10 (b).**
- The day-level cost (1.5 × missed event-days + 1 × false-alarm days) picks very high thresholds, because event-days are only ~4.6% of sessions. It gives few alarms but misses half the events. The unit of the cost ratio needs confirming (§12 Q3b).
- The 2016 coup was missed by every signal except RV20 (at 3 sessions). A sudden political shock gives volatility models no warning.

### TVTP simulation study, §6.1b (2026-09-27): **3 of 4 criteria pass** (`experiments/results/tvtp_sim_summary.md`)
- **Size:** under the null, the LR test rejects 10% of the time. That's exactly at the limit; the mean LR is 3.4 against 2.0 expected under χ²₂, so the test is slightly oversized.
- **Power:** 100% under both strong and weak.
- **Regime identification:** accuracy goes from 0.80 to 0.87 under strong (+7.1 pp) and from 0.74 to 0.79 under weak (+5.7 pp). Under the null there's no gain and no loss.
- **FAIL, δ recovery (strong):** calm→stress mean bias is +41% (mean 2.12 vs 1.5). The median is 1.74 (+16%), with a q10–q90 range of 0.75–3.71, so the distribution is heavily right-skewed with a few estimates of 4–5.6. Stress→calm is fine (−16% bias, median −1.58).
- **Reading:**
  - The covariate does identify regimes that returns alone can't, which is what Q10 (c) needed.
  - The *size* of the calm→stress effect is imprecise: stress entries are rare (short stress spells), so the rate effect rests on few transitions.
  - When reporting real-data δ, use bootstrap CIs and the median, not the point estimate alone.
  - If the LR test is the decision tool on real data, use a bootstrap LR or a stricter level (e.g. 1%), given the slight oversizing.

### TVTP-MSGJ2 on real data, §7.6 (2026-09-28): **FAILED on 3 of 5 criteria** (`experiments/results/tvtp_real_summary.md`, `event_scoring.md`)
The covariate is log Turkey 5Y CDS, lagged one session and standardized on burn-in stats. The Phase 4 run took ~40 min.

| Check | Result | |
|---|---|---|
| LR vs static MSGJ2, burn-in fit, 1% | LR 3.94, p = 0.14 | **FAIL** |
| Signs: δ(calm→stress) > 0 > δ(stress→calm) | median +1.62 and **+1.02**: higher CDS speeds up exits from stress too | **FAIL** |
| Log score DM > −1.96 | −0.62 … +0.15 vs MSGJ2 and GJR-t, all ties | pass |
| Detection: AUC ≥ 0.75 and event cost ≤ GJR-J usd (5) | AUC 0.535, 1/4 hits, cost 13, the worst signal | **FAIL** |
| Mean CDS higher in stress (smoothed) | 252 vs 250 bp | passes on the letter, but no economic difference |

- **Reading:** MSGJ2's two regimes aren't a CDS-stress split. P(stress) > 0.5 on 29% of validation days, and CDS is the same in both regimes. A transition covariate can't turn the regimes into crisis states when the returns split them some other way. The §6.1b simulation assumed the regimes *were* covariate-driven, so it tested identification, not whether that assumption holds on real data.
- **Q10 (c) with CDS alone fails.** Per §8.3 the criteria stay as they are. This spec isn't tuned any further on validation data.

**Event scoring under the per-event cost** (§12 Q3b, ratio 4) changes the ranking of the existing signals:
- RV20 catches 4/4 with 3 false-alarm episodes (cost 3). The GJR-J usd flag catches 3/4 with 1 (cost 5); it now catches the 2016 coup at lag 6.
- One event separates them, which is noise with 4 events. At ratio 2, both catch 3/4 with 0–1 false alarms. At ratio 8, both catch 4/4.

### §7.7 CDS signals and §7.8 v1 flag choice (2026-09-28) (`experiments/results/event_scoring.md`)
**§7.7: FAIL for both CDS signals.**

| Signal | AUC | Hits | FA eps | Cost | (a) standalone | (b) complementary |
|---|---|---|---|---|---|---|
| CDS lvl | 0.553 | 1/4 | 0 | 12 | fail | fail: OR misses 2013-12 as well |
| CDS d20 | 0.821 | 2/4 | 3 | 11 | fail: AUC passes, cost doesn't | fail: OR misses 2013-12, +2 FA eps |

- The CDS level catches only 2018, and 13 sessions late. It is a slow-moving, trending series. CDS d20 ranks well (AUC 0.82) but flags late (8 sessions on average), and it misses the equity-led events (Gezi, the coup) and the corruption probe.

**§7.8: GJR-J usd stays as the v1 flag.** No candidate clears the margin of more than 4:

| Candidate | Cost at ratio 2 / 4 / 8 | GJR-J usd alone |
|---|---|---|
| RV20 alone | 2 / 3 / 3 | 3 / 5 / 5 |
| GJR-J usd OR RV20 | 3 / 4 / 6 | |
| GJR-J usd OR CDS lvl | 3 / 5 / 5 | |
| GJR-J usd OR CDS d20 | 3 / 7 / 5 | |

- RV20 alone is lower at every ratio, by 1–2. Under the fixed rule that is less than one event's worth, so the model-based default stays. At the holdout, RV20 is reported next to it as the benchmark (§7.8).
- **Where v1 stands:** GJR-J usd, τ = 92.5th percentile, m = 3. Validation: 3/4 events caught (misses 2013-12), 1 false-alarm episode in 7 years, 2.3% of days in alarm.

### Holdout, §8.4 (2026-09-28, the single run, 2020-01-02 → 2026-09-25): **2 of 4 criteria pass** (`experiments/results/holdout_summary.md`, `holdout_tables.md`)

| Criterion | Result | |
|---|---|---|
| VaR coverage (≤ 2 of 12 tests reject at 5%) | 2 of 12: E 1% Kupiec (1.55% hits), F 2.5% Kupiec and CC (1.43%, too conservative). **GJR-t, the reference, would reject 7 of 12.** | **PASS** |
| ES Z2 at 2.5% (p ≥ 1% on every target) | E p = 0.008; F 0.87 and USD 0.56 | **FAIL** (E only) |
| Log score DM vs GJR-t > −1.96 | E −1.13, **F −2.47, USD −3.03** | **FAIL** |
| Flag (≥ 4 of 7 events, ≤ 1 false-alarm episode a year) | **7 of 7**, 0.30 FA episodes a year, 6.7% of days in alarm, median lag 2 sessions (the 2023 elections at 18). RV20 also catches 7 of 7, but with 0.75 FA episodes a year. | **PASS** |

- **Where the log score is lost (diagnostic, not tuning):**
  - F: the loss totals −76 log units. The 2021-12-21 KKM reversal (USDTRY −34.5% in one day) accounts for −26 of it. The rest is spread over the years, with −18 in 2026.
  - USD: the loss is spread over every year except 2025.
  - The pattern fits Gaussian diffusion plus a finite jump grid giving thinner far tails than a Student-t. That costs density on extreme days even though the quantiles (VaR coverage) are better calibrated than GJR-t's.
- **The 2021–24 managed-FX risk (§8.4)** didn't show up as FX VaR failures: F hit rates are low in both sub-periods (0.6–1.6%), so the forecasts were conservative, not breached.
- DCC-t logged numerical warnings in the optimizer. Its holdout scores contain no NaNs.
- **Per §8.4, v1 ships as specified, with these failures documented.** The holdout has now been used, so any later spec change (e.g. Student-t innovations) has no clean test sample left and must be reported as such.

### Student-t v2 candidate, §8.5 (2026-09-28): **all 4 checks pass** (`experiments/results/tdist_summary.md`)
| Check | Result |
|---|---|
| V1 burn-in LR | LR 14.7, p = 0.0001; ν = 8.5 |
| V2 validation DM vs MSGJ1 > 0 | E +0.46, F +0.90, USD +0.67: all positive, **none significant** |
| V3 validation VaR/ES | 1 of 12 rejections (F 2.5%, too conservative); Z2 p 0.53 / 0.98 / 0.44 |
| H seen holdout, no §8.4 criterion worse | yes on all three. On the holdout MSGJ1-t would clear every §8.4 criterion: coverage 2/12, Z2 E p = 0.03, DM vs GJR-t ≥ −1.70. DM vs MSGJ1 is +1.99 / +2.93 / +2.40 |

- ν is lower in the holdout (median 6.4) than in validation (10.1): the tails got fatter after 2020.
- **Reading:** on the clean data (validation) the t version is consistently but not significantly better. On the holdout it is significantly better and fixes both v1 failures. The holdout motivated this change, so that improvement is **not independent evidence**. Per §8.5, MSGJ1-t qualifies to replace MSGJ1 for VaR/ES; the flag stays on MSGJ1. Switching the daily runner awaits the user's decision.
- **Switched 2026-09-28 (D12).** `src/daily.py` now takes VaR/ES from MSGJ1-t, and the flag stays on MSGJ1. First report (as of 2026-09-25, ν = 6.3), compared with Gaussian MSGJ1: XU100 TRY 1% ES −6.97% vs −6.00% (deeper tail, as intended). USDTRY 1% VaR/ES −0.25% / −1.15% vs −0.82% / −2.54%: the t version puts less weight on the rare-jump mixture component in the long-USD tail.

### 10-day VaR/ES, §8.6 (2026-09-28): **fails, stays out of the daily report** (`experiments/results/tenday_summary.md`)
| # | Check | Result | Pass |
|---|---|---|---|
| T1 | Validation coverage, ≤ 1 of 6 HAC rejections | 2 (both USDTRY, too few hits: 0.29% at 1%, 1.32% at 2.5%) | no |
| T2 | Validation FZ0 DM vs GJR-t 10-day | E 1.64, F 3.52, USD 1.64 | yes |
| H | Holdout: rejections ≤ GJR-t's, and T2 | 0 vs 2 rejections, but F DM −7.81 (E 1.82, USD 1.14) | no |
- **Where it works:** XU100 in TRY and in USD. There, MSGJ1-t has the best coverage of the three methods and beats GJR-t on FZ0 in both samples (DM 1.1–1.8, not significant). Validation coverage rejections: MSGJ1-t 2, √10 3, GJR-t 5.
- **Where it fails: USDTRY.** In validation it is too conservative. In the holdout, its hits in the lira-strengthening tail (2020-05 → 2021-02, 2021-12, 2023-08) land far beyond the forecast ES: the mean loss beyond the 2.5% VaR is −10.0% against a mean ES of −5.4%. GJR-t is also off (−7.9% vs −4.5%) but less so. Dropping the KKM windows barely changes this (DM −7.62), so it isn't one day.
- **Per §8.6, the 10-day numbers stay out of the report.** Not tuned after the fact. The code stays: `model.horizon()` and `experiments/tenday.py`.

### XU100-only 10-day VaR/ES, §8.7 (2026-09-28): **P1–P3 pass (not blind); shipped with the live kill rule**
- P1: 0 of 4 validation coverage rejections. P2: FZ0 DM vs GJR-t E 1.64, USD 1.64. P3: holdout rejections 0 vs GJR-t's 1; DM E 1.82, USD 1.14.
- `src/daily.py` reports 10-day VaR/ES for XU100 TRY and USD (seed = origin index, so it reproduces `tenday.pkl` exactly; `tests/test_daily.py`). `live10()` scores the non-overlapping live windows from the first 10-day report (2026-09-25) and drops a series once the kill rule fires.
- First report (as of 2026-09-25): XU100 TRY 1% VaR −15.8%, ES −20.3%; XU100 USD 1% VaR −17.9%, ES −23.3%.

### Robustness battery, §8.8 (2026-09-28): **12 of 16 checks pass; jumps not supported (B3)** (`experiments/results/robustness_summary.md`)
- **Pass:** A1 (all 20 starts reach the same optimum), A2 (all core parameters recoverable; bias ≤ 0.13 CI widths), A3, A4 (E 0.996%, just inside 1%), B2, B4 (1 of 18), C1–C3 (γ_E, γ_F > 0 and ρ < 0, with 90% CIs clear of zero and the sign held in 100% of refits, holdout included), C6, D2, D3.
- **Fail:**
  - **B1:** a 5-year rolling window forecasts XU100 TRY better than the expanding window (DM −2.54). F and USD aren't significantly different. The 8-year window ties.
  - **C4:** the co-jump direction (m_C,E < 0 < m_C,F) holds in 88% of validation refits against the 90% required. It holds in F19 and in 100% of holdout refits. m_C is imprecise (CI spans zero).
  - **C5:** only 2 of the 10 largest jump-probability days fall in an `events.csv` window. Looked up afterwards, most of the others match news outside the frozen crisis list (the 2015 elections, the 2017 US visa suspension, March 2019). That lookup is post hoc and not evidence.
  - **D1:** validation Berkowitz p = 0.009 on F and 0.008 on USD (E 0.014).
- **B3, the H2 finding:** LR 12.8 for jumps vs none, bootstrap p = 0.080. Validation DM is +1.13, +0.30 and +1.32, all positive but not significant. Under the §8.8 rule, jumps are **not supported**. "Drop jumps (GJR-t bivariate)" is logged as a v2 candidate. Nothing is switched.
- **A3 caveat:** the persistence part of A3 can't fail, because `fit.py` caps persistence at 0.999. FX persistence **hits the cap in 16 holdout refits**, so FX variance is effectively integrated after 2020.
- **C6 computation fix:** the first computation used σ² + jump variance, which leaves out jump shocks feeding the GJR recursion. It gave E 1.35% and F 0.68% (a fail on F). The simulated value (10M days) is E 1.45% vs a sample 1.56%, and F 0.77% vs 0.92%. This fixes a code error; the criterion is unchanged.
- **Holdout (not blind):** FX PIT diagnostics all reject (p ≈ 0). The sample equity–FX correlation fell from −0.50 (2008–19) to −0.01 (2020–26), while the fitted ρ stays negative.
