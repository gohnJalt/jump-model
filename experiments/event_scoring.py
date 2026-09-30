"""DESIGN §7.3 stress-detection scoring on the validation period (2013-2019). Real-time signals only.

Every signal reuses the Phase 4 refit params (results/phase4/*.pkl), so it is out-of-sample and uses data through
the close of day t, which is what an evening risk report would publish.
  GJR-J usd   MSGJ1 (K=1) next-day conditional sd of XU100-in-USD, expanding-window percentile   [§12 Q10 (a)]
  GJR-J max   max of the equity and FX conditional-sd percentiles (stress from either source)   [§12 Q10 (a)]
  MSGJ2 P     MSGJ2 filtered P(stress regime)                                                    [§12 Q10 (b)]
  MS-JD3 P    MS-JD3 filtered P(top regime)                                                      [old model]
  TVTP-MSGJ2 P  TVTP-MSGJ2 filtered P(stress regime), transitions driven by lagged log CDS          [§12 Q10 (c)]
  RV20        20-session realized sd of XU100-in-USD, expanding percentile                       [naive benchmark]
  CDS lvl     log Turkey 5Y CDS (lagged one session), expanding percentile                      [§7.7]
  CDS d20     20-session change in log CDS (lagged one session), expanding percentile           [§7.7]
OR flags (§7.8): GJR-J usd OR another signal, each at its own chosen (tau, m); no joint grid.
Flag = signal >= tau for m consecutive sessions. (tau, m) per signal minimizes the event-level cost (§12 Q3b)
COST_MISS x missed events + 1 x false-alarm episodes; ties go to the smaller share of days in alarm.
An event is caught if a flag is on anywhere in [onset - 5, onset + 20) sessions.

Run: python3 experiments/event_scoring.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from model import garch_sig, hamilton
from phase4_bench import blocks, load, load_x

WIN, PRE, NEAR = 20, 5, 40  # event window, early-detection allowance, "near an event" for false-alarm episodes
COST_MISS, BAND = 4.0, (2.0, 8.0)  # §12 Q3b (fixed 2026-09-28): one caught crisis = 4 false-alarm episodes
TAUS_PCT = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
TAUS_PROB = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def pct_expanding(x, i0):
    """Percentile of x[t] among x[0..t] (real-time), for t >= i0."""
    return np.array([(x[: t + 1] <= x[t]).mean() for t in range(i0, len(x))])


def gjr_sd(y, bl, params):
    """MSGJ1 next-day sd forecasts (usd, E, F); row t = forecast made after the close of t, t = 0..bl[-1][1]-1.
    Rows before the first block use the first refit's params (the percentile reference history)."""
    hist, parts = None, []
    for (b0, b1), p in zip(bl, params):
        # sd forecast for t+1 made after the close of t -> row t+1 of garch_sig on y[:t+1] (append a dummy row)
        sd = np.asarray(garch_sig(p, np.vstack([y[:b1], np.zeros((1, 2))])))[:, 0]  # (b1+1, 2), K=1
        r = p["rho"][0]
        usd = np.sqrt(sd[:, 0] ** 2 + sd[:, 1] ** 2 - 2 * r * sd[:, 0] * sd[:, 1])
        if hist is None:
            hist = np.column_stack([usd, sd])[1:b0 + 1]
        parts.append(np.column_stack([usd, sd])[b0 + 1:b1 + 1])
    return np.vstack([hist, *parts])


def signals(y, dates, i0):
    n = len(y)
    P = {m: pickle.load(open(ROOT / f"experiments/results/phase4/{m}.pkl", "rb"))["params"]
         for m in ("MSGJ1", "MSGJ2", "MS-JD3", "TVTP-MSGJ2")}
    x = load_x(dates)
    bl = blocks(i0, n)
    g = gjr_sd(y, bl, P["MSGJ1"])
    out = {"GJR-J usd": pct_expanding(g[:, 0], i0)}
    pE, pF = pct_expanding(g[:, 1], i0), pct_expanding(g[:, 2], i0)
    out["GJR-J max"], attrib = np.maximum(pE, pF), (pE, pF)
    for m in ("MSGJ2", "MS-JD3", "TVTP-MSGJ2"):
        pr = []
        for (b0, b1), p in zip(bl, P[m]):
            pr.append(np.asarray(hamilton(p, y[:b1], x=x[:b1] if "delta" in p else None)[1])[b0:b1, -1])  # regimes sorted by equity variance: last = stress
        out[f"{m} P"] = np.concatenate(pr)
    usd_ret = y[:, 0] - y[:, 1]
    rv = pd.Series(usd_ret).rolling(WIN).std().to_numpy().copy()
    rv[:WIN] = np.nanmean(rv)
    out["RV20"] = pct_expanding(rv, i0)
    lc = np.log(pd.read_parquet(ROOT / "data/processed/covariates.parquet").reindex(dates)["cds"].to_numpy())
    d20 = pd.Series(lc).diff(WIN).to_numpy().copy()
    d20[:WIN] = np.nanmean(d20)
    out["CDS lvl"], out["CDS d20"] = pct_expanding(lc, i0), pct_expanding(d20, i0)
    return out, attrib


def flag(s, tau, m):
    above = s >= tau
    run = np.zeros(len(s), int)
    for t in range(len(s)):
        run[t] = run[t - 1] + 1 if (above[t] and t) else int(above[t])
    return run >= m


def evaluate(f, sig, onsets, n):
    """onsets: session indices (in the scored window) of events. Returns metrics dict + per-event lags."""
    label = np.zeros(n, bool)
    for o in onsets:
        label[o:o + WIN] = True
    lags = []
    for o in onsets:
        w = np.flatnonzero(f[max(o - PRE, 0):o + WIN])
        lags.append(max(w[0] + max(o - PRE, 0) - o, 0) if len(w) else None)
    starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
    near = lambda t: any(o - NEAR <= t <= o + NEAR for o in onsets)
    fa = [t for t in starts if not near(t)]
    hits = sum(l is not None for l in lags)
    miss = len(onsets) - hits
    auc = mannwhitneyu(sig[label], sig[~label]).statistic / (label.sum() * (~label).sum())
    return {"miss": miss, "fa": len(fa), "cost": COST_MISS * miss + len(fa), "hits": hits, "lags": lags,
            "FA_eps_per_yr": len(fa) / (n / 252), "alarm_share": f.mean(), "AUC": auc}


def or_flags(S, picks, onsets):
    base, n, lines, cost = "GJR-J usd", len(S["RV20"]), [], {}
    for c in (COST_MISS, *BAND):
        tb, mb, rb = picks[base](c)
        cost[base, c] = c * rb["miss"] + rb["fa"]
        for other in ("RV20", "CDS lvl", "CDS d20"):
            to, mo, _ = picks[other](c)
            r = evaluate(flag(S[base], tb, mb) | flag(S[other], to, mo), np.maximum(S[base], S[other]), onsets, n)
            cost[other, c] = c * r["miss"] + r["fa"]
            cost[other + " alone", c] = c * picks[other](c)[2]["miss"] + picks[other](c)[2]["fa"]
            lines.append(f"ratio {c:g}: {base} | {other}: hits {r['hits']}/{len(onsets)}, FA eps {r['fa']}, cost {cost[other, c]:g}"
                         f" (alone: {base} {cost[base, c]:g}); lags {[l if l is None else int(l) for l in r['lags']]}")
    for other in ("RV20", "CDS lvl", "CDS d20", "RV20 alone"):
        rep = cost[other, COST_MISS] < cost[base, COST_MISS] - 4 and all(cost[other, c] <= cost[base, c] for c in BAND)
        lines.append(f"§7.8 rule, {other if 'alone' in other else base + ' | ' + other}: {'REPLACES' if rep else 'does not replace'} {base}")
    return lines


def main():
    y, dates, i0 = load()
    d = dates[i0:]
    ev = pd.read_csv(ROOT / "data/events.csv", parse_dates=["onset"])
    ev = ev[(ev.onset >= d[0]) & (ev.onset <= d[-1])].reset_index(drop=True)
    onsets = [int(np.searchsorted(d, o)) for o in ev.onset]
    assert COST_MISS is not None, "fix COST_MISS (DESIGN §12 Q3b) before scoring"
    S, (pE, pF) = signals(y, dates, i0)
    rows, detail, band, picks = [], [], [], {}
    for name, s in S.items():
        taus = TAUS_PROB if name.endswith(" P") else TAUS_PCT
        grid = [(tau, m, evaluate(flag(s, tau, m), s, onsets, len(s))) for tau in taus for m in (1, 2, 3)]
        pick = picks[name] = lambda c, g=grid: min(g, key=lambda r: (c * r[2]["miss"] + r[2]["fa"], r[2]["alarm_share"]))
        band.append(f"{name}: " + ", ".join(f"ratio {c:g} -> tau {t}, m {mm}, hits {r['hits']}, FA eps {r['fa']}"
                                            for c in BAND for t, mm, r in [pick(c)]))
        tau, m, r = pick(COST_MISS)
        rows.append({"signal": name, "tau": tau, "m": m, "AUC": r["AUC"], "hits": f"{r['hits']}/{len(onsets)}",
                     "mean_lag": np.mean([l for l in r["lags"] if l is not None]) if r["hits"] else np.nan,
                     "FA_eps/yr": r["FA_eps_per_yr"], "alarm_share": r["alarm_share"], "cost": r["cost"]})
        detail.append((name, tau, m, r["lags"]))
    tab = pd.DataFrame(rows).set_index("signal")
    L = [f"# §7.3 stress detection, validation {d[0].date()} .. {d[-1].date()} ({len(onsets)} events, real-time signals)",
         f"Cost = {COST_MISS} x missed events + 1 x false-alarm episodes (§12 Q3b); ties -> fewer alarm days.\n",
         tab.round(3).to_string(), "\n## Detection lag per event (sessions; '-' = missed)"]
    lag_tab = pd.DataFrame({n: [("-" if l is None else l) for l in lags] for n, _, _, lags in detail},
                           index=[f"{o.date()} {e} [{t}]" for o, e, t in zip(ev.onset, ev.event, ev.type)])
    L.append(lag_tab.to_string())
    L += ["\n## Sensitivity of the chosen (tau, m) to the cost ratio"] + band
    L += ["\n## OR flags (§7.8): GJR-J usd OR x, each part at its own (tau, m) for that ratio", *or_flags(S, picks, onsets)]
    # attribution (GJR-J max): which side crossed tau first inside the event window
    tau = tab.loc["GJR-J max", "tau"]
    att = []
    for o, t in zip(onsets, ev.type):
        w = slice(max(o - PRE, 0), o + WIN)
        e_on, f_on = (pE[w] >= tau).any(), (pF[w] >= tau).any()
        got = "EF" if e_on and f_on else "E" if e_on else "F" if f_on else "-"
        att.append(f"{t}->{got}")
    L.append(f"\nAttribution (GJR-J max, listed type -> detected source): {', '.join(att)}")
    txt = "\n".join(L)
    (ROOT / "experiments/results/event_scoring.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
