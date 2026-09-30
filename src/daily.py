"""Daily risk report: MSGJ1-t 1-day VaR/ES for XU100 TRY (E), USDTRY (F) and XU100 USD (switched from MSGJ1 on
2026-09-28 per DESIGN §8.5), plus the MSGJ1 GJR-J usd stress flag (tau = 0.925, m = 3; §7.8, holdout-scored §13),
plus MSGJ1-t 10-day VaR/ES for XU100 TRY and USD only (§8.7, not blind; removed by the live kill rule in live10()).

Run after the close, once data/processed/returns.parquet is current (drop vendor files, then python3 src/data.py):
    python3 src/daily.py
Writes reports/daily/<date>.md and appends reports/daily.csv (one row per as-of date; a rerun replaces it).
State (data/state/<model>.pkl, one per model): the real-time refit history (block starts, params), seeded from the
evaluated validation and holdout runs and extended every R = 21 sessions exactly as evaluated.
"""
import pickle
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from phase4_bench import ALPHAS, LADDER, R, TARGETS, Mix, MixT, blocks
from phase4_bench import load as load_val
import numpy as np
import pandas as pd

from event_scoring import gjr_sd
from fit import default_sig_min, fit
import charts
from model import garch_sig, horizon, predictive
from tenday import H, N, tail

RES = ROOT / "experiments/results"
VAR_MODEL, FLAG_MODEL = "MSGJ1-t", "MSGJ1"
RUNS = {"MSGJ1": (RES / "phase4/MSGJ1.pkl", RES / "holdout/MSGJ1.pkl"),  # (validation, holdout) evaluated runs
        "MSGJ1-t": (RES / "phase4/MSGJ1-t.pkl", RES / "holdout_t/MSGJ1-t.pkl")}
STATE = {"MSGJ1": ROOT / "data/state/msgj1.pkl", "MSGJ1-t": ROOT / "data/state/msgj1-t.pkl"}
REPORTS = ROOT / "reports"
TAU, M = 0.925, 3
TEN = ("E", "USD")  # §8.7: 10-day targets (USDTRY failed §8.6)
LIVE0 = "2026-09-28"  # §8.7 live count start (the 2026-09-25 report used a stale USDTRY print)
HOLD0 = "2020-01-01"
CAVEATS = [  # DESIGN §13 holdout; keep in sync if the spec ever changes
    "VaR/ES switched to Student-t components (MSGJ1-t) on 2026-09-28, AFTER the holdout (DESIGN §8.5). On clean "
    "validation data it beat the Gaussian v1 on all three series, but not significantly; its holdout pass is not blind.",
    "USDTRY VaR at 2.5% is conservative (1.1% hits in the holdout, Kupiec p < 0.001).",
    "The flag caught 7 of 7 holdout events (median lag 2 sessions), 0.30 false-alarm episodes a year.",
    "Sudden political shocks give a volatility model no warning (the 2016 coup was caught 6 sessions late).",
]


def load():
    d = pd.read_parquet(ROOT / "data/processed/returns.parquet")
    return d[["rE", "rF"]].to_numpy(), d


def seed(y, d, model):
    """Refit history of the evaluated runs: validation (Phase 4) + holdout blocks and params."""
    _, _, i0v = load_val()
    i0h = int((d.index < HOLD0).sum())
    hold = pickle.load(open(RUNS[model][1], "rb"))
    val = pickle.load(open(RUNS[model][0], "rb"))["params"]
    starts = [b0 for b0, _ in blocks(i0v, i0h)] + [b0 for b0, _ in blocks(i0h, i0h + len(hold["dates"]))]
    assert len(starts) == len(val) + len(hold["params"])
    return {"starts": starts, "params": val + hold["params"], "sig_min": default_sig_min(y[:i0h])}


def update(y, d, model):
    """Refit on y[:b0] whenever the next session starts a new R-session block (warm start, 3 starts, as evaluated)."""
    path = STATE[model]
    st = pickle.load(open(path, "rb")) if path.exists() else seed(y, d, model)
    s = replace(LADDER[model], sig_min=st["sig_min"])
    while st["starts"][-1] + R <= len(y):
        b0 = st["starts"][-1] + R
        p, _, _ = fit(y[:b0], s, n_starts=3, init=st["params"][-1])
        st["starts"].append(b0)
        st["params"].append(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(path, "wb"))
    return st


def forecast(y, st, fl):
    """Risk for session len(y) (the next one), from data through the last close. st: VaR model state, fl: flag's."""
    n = len(y)
    k = max(i for i, b in enumerate(st["starts"]) if b <= n)  # block containing the forecast day
    p = st["params"][k]
    sig = np.asarray(garch_sig(p, np.vstack([y, np.zeros((1, 2))])))[-1:]  # row n uses y up to n-1
    out = {"var_model": VAR_MODEL if "nu" in p else FLAG_MODEL, "nu": float(p.get("nu", np.inf)),
           "sd_E": sig[0, 0, 0], "sd_F": sig[0, 0, 1]}
    for name, w in TARGETS.items():
        args = predictive(p, np.ones((1, 1)), w, sig)
        mix = MixT(*args, p["nu"]) if "nu" in p else Mix(*args)
        for a in ALPHAS:
            v = mix.var(a)
            out[f"VaR{a}_{name}"], out[f"ES{a}_{name}"] = float(v[0]), float(mix.es(a, v)[0])
    X = horizon(p, sig[0, 0], H, N, seed=n)  # seed = origin index, as in experiments/tenday.py
    for name in TEN:
        tt = tail(X @ TARGETS[name])
        for a in ALPHAS:
            out[f"VaR10_{a}_{name}"], out[f"ES10_{a}_{name}"] = float(tt[f"var{a}"]), float(tt[f"es{a}"])
    # flag: percentile of each forecast among all earlier ones (history from the burn-in), m sessions in a row
    live = [b for b in fl["starts"] if b < n]
    g = gjr_sd(y, list(zip(live, live[1:] + [n])), fl["params"][:len(live)])[:, 0]
    pct = np.array([(g[: t + 1] <= g[t]).mean() for t in range(n - 10, n)])
    run = next((i for i, v in enumerate(pct[::-1] >= TAU) if not v), len(pct))
    out.update(pct=pct[-1], sessions_above=run, flag=run >= M, sd_usd=g[-1], fit_n=st["starts"][k])  # fit on y[:fit_n]
    return out


def live10(csv, d, live0=LIVE0):
    """§8.7 live kill rule. Non-overlapping windows start at the first report with 10-day numbers on or after live0, and every H-th
    session after it; each covers the H sessions after its as-of date. Per series: completed windows, breaches
    (as-of date, alpha, loss), losses worse than 1.5 x ES 1%, and whether the rule has ever fired (it stays fired)."""
    c = csv.dropna(subset=[f"VaR10_{ALPHAS[0]}_{TEN[0]}"]) if f"VaR10_{ALPHAS[0]}_{TEN[0]}" in csv else csv[:0]
    c = c[c.index >= live0]
    ret = {"E": d["rE"].to_numpy(), "USD": (d["rE"] - d["rF"]).to_numpy()}
    out = {k: {"n": 0, "missing": 0, "w": [], "hits": [], "big": [], "killed": False} for k in TEN}
    if c.empty:
        return out
    pos = {d.index.get_loc(pd.Timestamp(i)): i for i in c.index}
    for i in range(min(pos), len(d) - H, H):  # sessions i+1 .. i+H are realized
        for k in TEN:
            o = out[k]
            if i not in pos:
                o["missing"] += 1
                continue
            row, loss = c.loc[pos[i]], ret[k][i + 1:i + 1 + H].sum()
            o["n"] += 1
            o["w"].append(tuple(loss <= row[f"VaR10_{a}_{k}"] for a in ALPHAS))
            o["hits"] += [(pos[i], a, loss) for a in ALPHAS if loss <= row[f"VaR10_{a}_{k}"]]
            if loss < 1.5 * row[f"ES10_{ALPHAS[0]}_{k}"]:
                o["big"].append((pos[i], loss))
            last = np.array(o["w"][-25:])
            o["killed"] |= bool(last[:, 0].sum() >= 2 or last[:, 1].sum() >= 3)  # 1%: >= 2, 2.5%: >= 3
    return out


def ten(f, live):
    label = {"E": "XU100 (TRY)", "USD": "XU100 (USD)"}
    L = ["## 10-day VaR / ES, sum of the next 10 sessions (log return, %; NOT blind, DESIGN §8.7)", "",
         "Selected after the §8.6 backtest was seen (USDTRY failed it and is not reported). The only blind test is "
         "live: a series is removed if, over its last 25 non-overlapping windows, it has >= 2 breaches at 1% or >= 3 "
         "at 2.5%.", "", "| Series | Level | VaR | ES |", "|---|---|---|---|"]
    for k in TEN:
        if live[k]["killed"]:
            continue
        L += [f"| {label[k]} | {a:.1%} | {100 * f[f'VaR10_{a}_{k}']:.2f} | {100 * f[f'ES10_{a}_{k}']:.2f} |" for a in ALPHAS]
    L.append("")
    for k in TEN:
        o = live[k]
        L.append(f"- {label[k]}: **REMOVED by the live kill rule.**" if o["killed"] else
                 f"- {label[k]} live score: {o['n']} completed window(s), breaches: "
                 + ", ".join(f"{sum(h[1] == a for h in o['hits'])} at {a:.1%}" for a in ALPHAS)
                 + (f", {o['missing']} window(s) with no report" if o["missing"] else "") + ".")
        L += [f"  - Review: the window from {day} lost {100 * loss:.2f}%, worse than 1.5 x ES 1%." for day, loss in o["big"]]
    return L + [""]


def report(d, f, live, figs=()):
    day = d.index[-1].date()
    last = d.iloc[-1]
    rows = [(name, a, f[f"VaR{a}_{name}"], f[f"ES{a}_{name}"]) for name in TARGETS for a in ALPHAS]
    label = {"E": "XU100 (TRY)", "F": "USDTRY (long USD)", "USD": "XU100 (USD)"}
    L = [f"# XU100 / USDTRY daily risk, as of the {day} close", "",
         f"**Stress flag: {'ON' if f['flag'] else 'off'}.** The XU100-in-USD volatility forecast is at the "
         f"{f['pct']:.1%} percentile of its history. It has been above {TAU:.1%} for {f['sessions_above']} session(s); "
         f"the flag needs {M} in a row.", "",
         "## 1-day VaR / ES for the next session (log return, %)", "",
         "Lower tail = loss on a long position, as backtested. For USDTRY that is the loss from holding USD (lira "
         "strengthening); the lira-depreciation tail was not backtested and is not reported.", "",
         "| Series | Level | VaR | ES |", "|---|---|---|---|",
         *[f"| {label[n]} | {a:.1%} | {100 * v:.2f} | {100 * e:.2f} |" for n, a, v, e in rows], "",
         *ten(f, live),
         f"Forecast daily sd: XU100 TRY {100 * f['sd_E']:.2f}%, USDTRY {100 * f['sd_F']:.2f}%, XU100 USD {100 * f['sd_usd']:.2f}%.", "",
         "## Data", "",
         f"- Last session {day}. Sources: equity {last['eq_src']}, FX {last['fx_src']}."
         + (f" **FX print is {int(last['fx_lag_days'])} day(s) stale.**" if last["fx_lag_days"] > 0 else ""),
         f"- VaR/ES: {f['var_model']} (bivariate GJR-J, K = 1, Student-t nu = {f['nu']:.1f}); flag: {FLAG_MODEL}. "
         f"Params fitted on data through {d.index[f['fit_n'] - 1].date()} (refit every {R} sessions).", "",
         *(["## Charts (data: reports/history.csv)", "", *[f"![{c[:-4]}](charts/{day}/{c})" for c in figs], ""] if figs else []),
         "## Known limitations (holdout, DESIGN §13)", "", *[f"- {c}" for c in CAVEATS]]
    return "\n".join(L)


def main():
    y, d = load()
    st, fl = update(y, d, VAR_MODEL), update(y, d, FLAG_MODEL)
    f = forecast(y, st, fl)
    (REPORTS / "daily").mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"date": d.index[-1].date(), **{k: v for k, v in f.items()}}]).set_index("date")
    csv = REPORTS / "daily.csv"
    if csv.exists():
        old = pd.read_csv(csv, index_col="date", parse_dates=False)
        row = pd.concat([old[old.index != str(row.index[0])], row.rename(index=str)])
    row.to_csv(csv)
    day = d.index[-1].date()
    figs = charts.charts(charts.history(y, d, st, fl, TAU, M), f, REPORTS / "daily" / "charts" / str(day), TAU)
    txt = report(d, f, live10(row, d), figs)
    (REPORTS / "daily" / f"{d.index[-1].date()}.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
