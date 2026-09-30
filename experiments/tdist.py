"""DESIGN §8.5: MSGJ1-t (Student-t mixture components) vs MSGJ1, the v2 candidate for VaR/ES.

V1 burn-in LR, V2 validation DM vs MSGJ1, V3 validation VaR/ES, H holdout (ALREADY SEEN: can veto, never confirm).
Validation pkl: experiments/results/phase4/MSGJ1-t.pkl (Phase 4 protocol). Holdout pkl: results/holdout_t/.
The one-shot holdout outputs (results/holdout/) are only read, never rewritten.

Run: python3 experiments/tdist.py            Summary only (pkls exist): python3 experiments/tdist.py --summary
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from phase4_bench import ALPHAS, LADDER, TARGETS, dm, kupiec_christoffersen, load, run, run_ladder, z2
import numpy as np
import pandas as pd
from scipy.stats import chi2

from fit import NU_MAX, default_sig_min, fit
from model import hamilton

RES = ROOT / "experiments/results"
HT = RES / "holdout_t"


def job(which):
    if which == "val":
        return run("MSGJ1-t")
    path = HT / "MSGJ1-t.pkl"
    if not path.exists():
        d = pd.read_parquet(ROOT / "data/processed/returns.parquet")
        y, i0 = d[["rE", "rF"]].to_numpy(), int((d.index < "2020-01-01").sum())
        tg, biv, params = run_ladder("MSGJ1-t", y, i0)
        pickle.dump({"name": "MSGJ1-t", "dates": d.index[i0:], "targets": tg, "biv": biv, "params": params}, open(path, "wb"))
    return which


def crit(tg, ref):
    """§8.4-style numbers for one run: Kupiec + CC rejections of 12, Z2 p at 2.5%, DM log score vs ref per target."""
    rej, rows = 0, []
    for k in TARGETS:
        r = tg[k]
        for a in ALPHAS:
            I = r["y"] <= r[f"var{a}"]
            uc, _, cc = kupiec_christoffersen(I, a)
            rej += (uc < .05) + (cc < .05)
            rows.append(f"{k} {a:.1%}: hits {I.mean():.4f}, Kupiec {uc:.3f}, CC {cc:.3f}")
    zp = {}
    for k in TARGETS:
        r, a = tg[k], 0.025
        v, e = r[f"var{a}"], r[f"es{a}"]
        zp[k] = (z2(r["sims"], v[:, None], e[:, None], a) <= z2(r["y"], v, e, a)).mean()
    d = {k: dm(tg[k]["ls"] - ref[k]["ls"]) for k in TARGETS}
    return rej, zp, d, rows


def summary():
    y, _, i0 = load()
    P = lambda path: pickle.load(open(path, "rb"))
    vt, vg, vr = P(RES / "phase4/MSGJ1-t.pkl"), P(RES / "phase4/MSGJ1.pkl"), P(RES / "phase4/GJR-t.pkl")
    ht, hg, hr = P(HT / "MSGJ1-t.pkl"), P(RES / "holdout/MSGJ1.pkl"), P(RES / "holdout/GJR-t.pkl")

    # V1: burn-in LR; MSGJ1-t also warm-started from MSGJ1 at nu near the top, keep the better fit
    pg, pt = vg["params"][0], vt["params"][0]
    s = replace(LADDER["MSGJ1-t"], sig_min=default_sig_min(y[:i0]))
    warm, _, _ = fit(y[:i0], s, n_starts=1, init=pg | {"nu": NU_MAX - 1})
    ll_g = float(hamilton(pg, y[:i0])[0])
    ll_t, pb = max(((float(hamilton(q, y[:i0])[0]), q) for q in (pt, warm)), key=lambda r: r[0])
    LR = 2 * (ll_t - ll_g)
    ok = {"V1 LR at 1%": chi2.sf(LR, 1) < 0.01}

    rv, zv, dv, rows_v = crit(vt["targets"], vg["targets"])
    _, _, dv_ref, _ = crit(vt["targets"], vr["targets"])
    ok["V2 DM vs MSGJ1 > 0 (validation)"] = min(dv.values()) > 0
    ok["V3 VaR/ES (validation)"] = rv <= 2 and min(zv.values()) >= 0.01

    rh, zh, dh_ref, rows_h = crit(ht["targets"], hr["targets"])
    rg, zg, dg_ref, _ = crit(hg["targets"], hr["targets"])
    _, _, dh, _ = crit(ht["targets"], hg["targets"])
    no_worse = {"coverage": (rh <= 2) >= (rg <= 2), "ES": (min(zh.values()) >= .01) >= (min(zg.values()) >= .01),
                "log score": (min(dh_ref.values()) > -1.96) >= (min(dg_ref.values()) > -1.96)}
    ok["H no §8.4 criterion worse (seen holdout)"] = all(no_worse.values())

    nus = [float(p["nu"]) for p in vt["params"] + ht["params"]]
    f = lambda d: ", ".join(f"{k} {v:.2f}" for k, v in d.items())
    L = ["# MSGJ1-t vs MSGJ1, DESIGN §8.5 (holdout part NOT blind)", "",
         "Pass/fail: " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items()), "",
         f"V1 burn-in: MSGJ1 ll {ll_g:.1f}, MSGJ1-t ll {ll_t:.1f} (nu {float(pb['nu']):.1f}) -> LR {LR:.1f}, p {chi2.sf(LR, 1):.2g}",
         f"nu over refits: validation median {np.median(nus[:len(vt['params'])]):.1f}, holdout median {np.median(nus[len(vt['params']):]):.1f}"
         f" (range {min(nus):.1f}..{max(nus):.1f})", "",
         "## Validation 2013-2019",
         f"DM log score vs MSGJ1: {f(dv)};  vs GJR-t: {f(dv_ref)}",
         f"VaR rejections {rv} of 12; Z2 p at 2.5%: {f(zv)}", *rows_v, "",
         "## Holdout 2020+ (seen; veto only)",
         f"DM log score vs MSGJ1: {f(dh)};  vs GJR-t: {f(dh_ref)} (MSGJ1 vs GJR-t: {f(dg_ref)})",
         f"VaR rejections {rh} of 12 (MSGJ1: {rg}); Z2 p at 2.5%: {f(zh)} (MSGJ1: {f(zg)})", *rows_h,
         "", "No-worse-than-MSGJ1 on §8.4: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in no_worse.items())]
    txt = "\n".join(L)
    (RES / "tdist_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        HT.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as ex:
            for w in ex.map(job, ("val", "hold")):
                print("done", w, flush=True)
    summary()
