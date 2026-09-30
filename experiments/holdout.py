"""DESIGN §8.4 holdout: the ONE run of the final v1 spec on 2020-01-01 onward. Needs the user's explicit go-ahead.

Spec: MSGJ1 (bivariate GJR-J, K=1) for 1-day VaR/ES and the stress flag (GJR-J usd, tau = 0.925, m = 3).
Benchmarks: GJR-t (VaR reference), DCC-t (bivariate), RV20 (flag; its validation (tau, m) at ratio 4).
Protocol as Phase 4: expanding window, refit every 21 sessions, first fit on all data through 2019-12-31.
Flag percentiles: expanding history from the burn-in (validation rows use the Phase 4 MSGJ1 refits).
"Christoffersen" = conditional-coverage test, the CC column Phase 4 reports.

Run: python3 experiments/holdout.py --go [workers]     Refuses without --go, and refuses once the summary exists.
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from phase4_bench import (ALPHAS, TARGETS, blocks, dm, kupiec_christoffersen, run_dcc, run_garch, run_ladder,
                          summary, z2)
from phase4_bench import load as load_val
import numpy as np
import pandas as pd

from event_scoring import WIN, evaluate, flag, gjr_sd, pct_expanding

HOLD0 = "2020-01-01"
OUT, SUMMARY = ROOT / "experiments/results/holdout", ROOT / "experiments/results/holdout_summary.md"
JOBS = ("MSGJ1", "GJR-t", "DCC-t")
FLAG, RV20 = (0.925, 3), (0.85, 3)  # §7.8 v1 flag; RV20 as chosen on validation at ratio 4 (event_scoring.md)
MANAGED = ("2021-01-01", "2024-12-31")  # §8.4: managed-FX years, coverage reported separately


def load_all():
    d = pd.read_parquet(ROOT / "data/processed/returns.parquet")
    return d[["rE", "rF"]].to_numpy(), d.index, int((d.index < HOLD0).sum())


def job(name):
    path = OUT / f"{name}.pkl"
    if not path.exists():
        y, dates, i0 = load_all()
        tg, biv, params = (run_ladder(name, y, i0) if name == "MSGJ1" else
                           run_garch(name, y, i0) if name == "GJR-t" else run_dcc(y, i0))
        pickle.dump({"name": name, "dates": dates[i0:], "targets": tg, "biv": biv, "params": params}, open(path, "wb"))
    return name


def checks():
    y, dates, i0 = load_all()
    M = {m: pickle.load(open(OUT / f"{m}.pkl", "rb")) for m in JOBS}
    L, ok = [], {}

    # VaR coverage: Kupiec + CC, 2 levels x 3 targets x 2 tests; <= 2 rejections at 5%
    rej, rows = 0, []
    managed = (M["MSGJ1"]["dates"] >= MANAGED[0]) & (M["MSGJ1"]["dates"] <= MANAGED[1])
    for m in ("MSGJ1", "GJR-t"):
        for k in TARGETS:
            r = M[m]["targets"][k]
            for a in ALPHAS:
                I = r["y"] <= r[f"var{a}"]
                uc, _, cc = kupiec_christoffersen(I, a)
                if m == "MSGJ1":
                    rej += (uc < .05) + (cc < .05)
                rows.append({"model": m, "target": k, "alpha": a, "hits": I.mean(), "Kupiec_p": uc, "CC_p": cc,
                             "hits 2021-24": I[managed].mean(), "hits rest": I[~managed].mean()})
    ok["VaR coverage (<= 2 of 12 reject)"] = rej <= 2
    L += ["## VaR coverage (hits = share of days below VaR)", pd.DataFrame(rows).round(4).to_string(index=False),
          f"MSGJ1 rejections at 5%: {rej} of 12"]

    # ES: Z2 at 2.5% does not reject at 1% on any target
    zp = {}
    for k in TARGETS:
        r, a = M["MSGJ1"]["targets"][k], 0.025
        v, e = r[f"var{a}"], r[f"es{a}"]
        zp[k] = (z2(r["sims"], v[:, None], e[:, None], a) <= z2(r["y"], v, e, a)).mean()
    ok["ES Z2 at 2.5% (p >= 1%)"] = min(zp.values()) >= 0.01
    L.append("Z2 p at 2.5%: " + ", ".join(f"{k} {p:.3f}" for k, p in zp.items()))

    # log score vs GJR-t
    d = {k: dm(M["MSGJ1"]["targets"][k]["ls"] - M["GJR-t"]["targets"][k]["ls"]) for k in TARGETS}
    ok["Log score DM vs GJR-t > -1.96"] = min(d.values()) > -1.96
    L.append("DM log score MSGJ1 vs GJR-t (+ = MSGJ1 better): " + ", ".join(f"{k} {v:.2f}" for k, v in d.items()))

    # stress flag: >= 4 of 7 events, <= 1 false-alarm episode a year
    _, _, i0v = load_val()
    P4 = pickle.load(open(ROOT / "experiments/results/phase4/MSGJ1.pkl", "rb"))["params"]
    bl = blocks(i0v, i0) + blocks(i0, len(y))
    g = gjr_sd(y, bl, P4 + M["MSGJ1"]["params"])
    assert len(g) == len(y) and len(P4) == len(blocks(i0v, i0))
    rv = pd.Series(y[:, 0] - y[:, 1]).rolling(WIN).std().to_numpy().copy()
    rv[:WIN] = np.nanmean(rv)
    S = {"GJR-J usd (v1)": (pct_expanding(g[:, 0], i0), FLAG), "RV20 (benchmark)": (pct_expanding(rv, i0), RV20)}
    hd = dates[i0:]
    ev = pd.read_csv(ROOT / "data/events.csv", parse_dates=["onset"])
    ev = ev[(ev.onset >= hd[0]) & (ev.onset <= hd[-1])].reset_index(drop=True)
    onsets = [int(np.searchsorted(hd, o)) for o in ev.onset]
    assert len(onsets) == 7, f"§8.4 was fixed for 7 holdout events, found {len(onsets)}: stop and ask the user"
    rows, lags = [], {}
    for name, (s, (tau, m)) in S.items():
        r = evaluate(flag(s, tau, m), s, onsets, len(s))
        rows.append({"signal": name, "tau": tau, "m": m, "AUC": r["AUC"], "hits": f"{r['hits']}/{len(onsets)}",
                     "FA_eps/yr": r["FA_eps_per_yr"], "alarm_share": r["alarm_share"]})
        lags[name] = ["-" if l is None else int(l) for l in r["lags"]]
        if name.startswith("GJR"):
            ok[f"Flag >= 4/{len(onsets)} events, <= 1 FA episode/yr"] = r["hits"] >= 4 and r["FA_eps_per_yr"] <= 1
    L += ["\n## Stress flag", pd.DataFrame(rows).set_index("signal").round(3).to_string(),
          pd.DataFrame(lags, index=[f"{o.date()} {e}" for o, e in zip(ev.onset, ev.event)]).to_string()]

    head = [f"# Holdout ({hd[0].date()} .. {hd[-1].date()}), DESIGN §8.4, single run", "",
            "Pass/fail: " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items()), ""]
    txt = "\n".join(head + L + ["", "Full tables: holdout_tables.md"])
    SUMMARY.write_text(txt)
    print(txt)


if __name__ == "__main__":
    if "--go" not in sys.argv:
        sys.exit("Holdout is one-shot (DESIGN §8.4). Run with --go only after the user's explicit go-ahead.")
    if SUMMARY.exists():
        sys.exit(f"{SUMMARY} exists: the holdout has already been run. Not re-running.")
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(next((a for a in sys.argv[1:] if a.isdigit()), 3))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name in ex.map(job, JOBS):
            print("done", name, flush=True)
    summary(OUT, ROOT / "experiments/results/holdout_tables.md", f"Holdout out-of-sample ({HOLD0} ..)")
    checks()
