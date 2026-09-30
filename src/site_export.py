"""Website feed: reports/history.csv + reports/daily.csv -> one compact risk.json for the Market Readings site
(../bubble-methodology, site/risk.json). Derived numbers and realised log returns only; the raw BFIX file never leaves
this machine. Run by scripts/daily.sh after a successful daily.py; by hand: python3 src/site_export.py [out.json]

Units: every return, VaR and ES is in percent (log return x 100), 2 dp; probabilities 4 dp.
Alignment (as in history.csv): hist row t carries the forecast FOR session t made at the close of t-1, and the
realised return OF t, so a breach is r[t] <= VaR[t]. `next` is the forecast for the session after `asof`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import daily  # noqa: E402  (live10, constants and caveats stay defined in one place)

OUT = ROOT.parent / "bubble-methodology" / "site" / "risk.json"
SERIES = ("E", "F", "USD")
A1, A25 = daily.ALPHAS


def _pc(s):
    return [None if not np.isfinite(v) else round(100 * float(v), 2) for v in s]


def _p(s, nd=4):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in s]


def build():
    h = pd.read_csv(daily.REPORTS / "history.csv", index_col="date", parse_dates=["date"])
    c = pd.read_csv(daily.REPORTS / "daily.csv", index_col="date")
    f = c.iloc[-1]
    ev = pd.read_csv(ROOT / "data/events.csv")
    ev = ev[ev["onset"] >= f"{h.index[0]:%Y-%m-%d}"]

    d = pd.DataFrame({"rE": h["rE"], "rF": h["rF"]})  # live10 only needs returns indexed by session
    live = daily.live10(c, d)

    hist = {"date": [f"{t:%Y-%m-%d}" for t in h.index],
            "period": "".join({"validation": "v", "holdout": "h", "live": "l"}[p] for p in h["period"]),
            "pct": _p(h["pct"]), "flag": [int(bool(x)) for x in h["flag"]],
            "fx_share": _p(h["fx_share"], 3),
            **{f"pj_{k}": _p(h[f"pj_{k}"]) for k in "EFC"},
            **{f"r_{k}": _pc(h[f"r{k}"]) for k in SERIES}}
    for k in SERIES:
        hist[f"var1_{k}"], hist[f"var25_{k}"] = _pc(h[f"VaR{A1}_{k}"]), _pc(h[f"VaR{A25}_{k}"])
        hist[f"es1_{k}"] = _pc(h[f"ES{A1}_{k}"])

    nxt = {"pct": round(float(f["pct"]), 4), "sessions_above": int(f["sessions_above"]), "flag": bool(f["flag"] == True),
           "sd": {"E": round(100 * f["sd_E"], 2), "F": round(100 * f["sd_F"], 3), "USD": round(100 * f["sd_usd"], 2)},
           "var_model": f["var_model"], "flag_model": daily.FLAG_MODEL, "nu": round(float(f["nu"]), 2),
           "fit_n": int(f["fit_n"]), "one": {}, "ten": {}}
    for k in SERIES:
        nxt["one"][k] = {str(a): {"var": round(100 * f[f"VaR{a}_{k}"], 2), "es": round(100 * f[f"ES{a}_{k}"], 2)}
                         for a in daily.ALPHAS}
    for k in daily.TEN:
        if live[k]["killed"]:  # §8.7: a killed series is not reported
            continue
        nxt["ten"][k] = {str(a): {"var": round(100 * f[f"VaR10_{a}_{k}"], 2), "es": round(100 * f[f"ES10_{a}_{k}"], 2)}
                         for a in daily.ALPHAS}

    return {"built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "asof": str(c.index[-1]),
            "tau": daily.TAU, "m": daily.M, "hold0": daily.HOLD0, "live0": daily.LIVE0, "horizon10": daily.H,
            "alphas": list(daily.ALPHAS), "next": nxt,
            "live10": {k: {"n": o["n"], "missing": o["missing"], "killed": o["killed"],
                           "hits": {str(a): sum(x[1] == a for x in o["hits"]) for a in daily.ALPHAS},
                           "big": [[str(day), round(100 * loss, 2)] for day, loss in o["big"]]}
                       for k, o in live.items()},
            "caveats": daily.CAVEATS,
            "events": ev[["onset", "event", "type"]].to_dict("records"),
            "hist": hist}


def _selfcheck(o):
    n = len(o["hist"]["date"])
    assert all(len(v) == n for v in o["hist"].values()), "ragged history columns"
    assert o["hist"]["date"][-1] <= o["asof"], "history runs past the as-of session"
    assert "F" not in o["next"]["ten"], "USDTRY 10-day failed §8.6 and must never be published"
    assert all(v["var"] < 0 and v["es"] <= v["var"] for s in o["next"]["one"].values() for v in s.values())


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    o = build()
    _selfcheck(o)
    out.write_text(json.dumps(o, separators=(",", ":"), allow_nan=False))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB), as of {o['asof']}")
