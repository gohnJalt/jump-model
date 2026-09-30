"""The daily runner must reproduce the evaluated numbers: MSGJ1-t holdout VaR/ES and the MSGJ1 flag percentile. Run: python3 tests/test_daily.py"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

import daily
from event_scoring import gjr_sd, pct_expanding
from phase4_bench import ALPHAS, R, TARGETS, blocks

y, d = daily.load()
st, fl = daily.seed(y, d, "MSGJ1-t"), daily.seed(y, d, "MSGJ1")  # in memory; never touches data/state
hold = pickle.load(open(daily.RUNS["MSGJ1-t"][1], "rb"))
i0h = int((d.index < daily.HOLD0).sum())
nh = i0h + len(hold["dates"])
ten = pickle.load(open(ROOT / "experiments/results/tenday.pkl", "rb"))  # §8.6 run
for day in (i0h + R, i0h + R + 5, nh - 1):  # first day of a refit block, mid-block, last holdout day
    f = daily.forecast(y[:day], st, fl)
    assert f["var_model"] == "MSGJ1-t"
    j = np.flatnonzero(ten["t"] == day)
    for k in daily.TEN:
        for a in ALPHAS:
            for key, src in ((f"VaR10_{a}_{k}", f"var{a}"), (f"ES10_{a}_{k}", f"es{a}")):
                assert not j.size or abs(f[key] - ten[k]["ms"][src][j[0]]) < 1e-12, (day, key)
    for k in TARGETS:
        for a in ALPHAS:
            for key, src in ((f"VaR{a}_{k}", f"var{a}"), (f"ES{a}_{k}", f"es{a}")):
                want = hold["targets"][k][src][day - i0h]
                assert abs(f[key] - want) < 1e-9, (day, key, f[key], want)
_, _, i0v = daily.load_val()
g = gjr_sd(y[:nh], blocks(i0v, i0h) + blocks(i0h, nh), fl["params"])[:, 0]
assert abs(daily.forecast(y[:nh], st, fl)["pct"] - pct_expanding(g, i0h)[-1]) < 1e-12

# §8.7 live kill rule on a synthetic report history: windows every H sessions from the first 10-day row
import pandas as pd
dd = d.iloc[-200:]
rows = pd.DataFrame(index=[str(t.date()) for t in dd.index[:150]])
for k in daily.TEN:
    for a in ALPHAS:
        rows[f"VaR10_{a}_{k}"], rows[f"ES10_{a}_{k}"] = -1.0, -2.0  # never breached
rows.iloc[:3] = np.nan  # before 10-day numbers existed
lv = daily.live10(rows, dd, live0="")
assert lv["E"]["n"] == 15 and lv["E"]["missing"] == 4 and not lv["E"]["killed"] and not lv["E"]["hits"]
rows.loc[rows.index[3], "VaR10_0.01_E"] = rows.loc[rows.index[13], "VaR10_0.01_E"] = 1.0  # two 1% breaches
lv = daily.live10(rows, dd, live0="")
assert lv["E"]["killed"] and len(lv["E"]["hits"]) == 2 and not lv["USD"]["killed"]
assert daily.live10(rows, dd, live0=rows.index[4])["E"]["n"] == 15 and not daily.live10(rows, dd, live0=rows.index[4])["E"]["hits"]
# chart history: real-time VaR equals the evaluated holdout run; an incremental update equals a full rebuild
import tempfile
import charts
k = max(i for i, b in enumerate(st["starts"]) if b <= i0h + R)
b0 = st["starts"][k]
blk = charts._block(st["params"][k], y, b0, b0 + R)
for key in ("VaR0.01_E", "ES0.025_USD", "VaR0.01_F"):
    a, c = key.split("_")[0][3 if key.startswith("VaR") else 2:], key.split("_")[1]
    want = hold["targets"][c][("var" if key.startswith("VaR") else "es") + a][b0 - i0h:b0 - i0h + R]
    assert np.allclose(blk[key], want, rtol=0, atol=1e-9), key
sub = lambda s_, j: {"starts": s_["starts"][j:], "params": s_["params"][j:]}
st3, fl3 = sub(st, len(st["starts"]) - 3), sub(fl, len(fl["starts"]) - 3)
with tempfile.TemporaryDirectory() as tmp:
    p1, p2 = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
    charts.history(y[:len(y) - 30], d[:len(y) - 30], st3, fl3, daily.TAU, daily.M, p1)
    inc = charts.history(y, d, st3, fl3, daily.TAU, daily.M, p1)
    full = charts.history(y, d, st3, fl3, daily.TAU, daily.M, p2)
    num = full.select_dtypes("number").columns
    assert inc.index.equals(full.index) and np.allclose(inc[num], full[num], rtol=1e-5, equal_nan=True)
    assert (inc["flag"] == full["flag"]).all()
print("ok")
