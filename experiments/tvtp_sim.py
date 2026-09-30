"""TVTP simulation study (DESIGN §6.1b): can a covariate identify regimes that returns alone can't?

Truth: calibrated K=2 MS-GARCH-J (Phase 3 showed these regimes are unidentifiable from returns) + TVTP with
delta = (+d calm->stress, -d stress->calm) on one exogenous CDS-like covariate (AR(1), phi = 0.995, standardized).
Per replication: fit static MSGJ2 and TVTP MSGJ2 (warm-started from the static fit, so LR >= 0), LR test delta = 0.
Scenarios: null d = 0, weak d = 0.75, strong d = 1.5; R = 20 each; T = 4,544.

Run: python3 experiments/tvtp_sim.py [workers]      Summary only: python3 experiments/tvtp_sim.py --summary
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
import numpy as np
from scipy.stats import chi2

from fit import Spec, default_sig_min, fit
from model import hamilton, simulate
from phase3_sim import truth

OUT = ROOT / "experiments/results/tvtp_sim"
SCEN, R, T, STARTS, PHI = {"null": 0.0, "weak": 0.75, "strong": 1.5}, 20, 4544, 3, 0.995


def covariate(seed):
    rng = np.random.default_rng(10_000 + seed)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = PHI * x[t - 1] + np.sqrt(1 - PHI**2) * rng.standard_normal()
    return ((x - x.mean()) / x.std())[:, None]


def run(job):
    scen, rep = job
    path = OUT / f"{scen}_{rep}.pkl"
    if path.exists():
        return job
    seed = 100 * list(SCEN).index(scen) + rep
    d = SCEN[scen]
    p = truth(2, msgj=True) | {"delta": np.array([[[0.0], [d]], [[-d], [0.0]]])}
    x = covariate(seed)
    y, s, _ = simulate(p, T, seed=seed, x=x)
    smin = default_sig_min(y)
    ps, lls, _ = fit(y, Spec(2, garch=True, sig_min=smin), n_starts=STARTS)
    pt, llt, _ = fit(y, Spec(2, garch=True, n_cov=1, sig_min=smin), n_starts=STARTS, x=x,
                     init=ps | {"delta": np.zeros((2, 2, 1))})
    acc = lambda q, **kw: float((np.asarray(hamilton(q, y, **kw)[1]).argmax(1) == s).mean())
    out = {"scen": scen, "rep": rep, "delta_true": d, "delta_hat": (float(pt["delta"][0, 1, 0]), float(pt["delta"][1, 0, 0])),
           "LR": 2 * (llt - lls), "acc_static": acc(ps), "acc_tvtp": acc(pt, x=x), "p_static": ps, "p_tvtp": pt}
    pickle.dump(out, open(path, "wb"))
    return job


def summary():
    rs = [pickle.load(open(f, "rb")) for f in OUT.glob("*.pkl")]
    crit = chi2.ppf(0.95, 2)
    L, gate = ["# TVTP simulation study (DESIGN §6.1b)\n"], []
    for scen, d in SCEN.items():
        g = [r for r in rs if r["scen"] == scen]
        if not g:
            continue
        rej = np.mean([r["LR"] > crit for r in g])
        dh = np.array([r["delta_hat"] for r in g])
        a_s, a_t = np.mean([r["acc_static"] for r in g]), np.mean([r["acc_tvtp"] for r in g])
        L.append(f"## {scen} (delta* = {d}, R = {len(g)})\nLR rejection rate at 5%: {rej:.2f}   mean LR {np.mean([r['LR'] for r in g]):.2f}"
                 f"\ndelta_hat calm->stress mean {dh[:, 0].mean():+.2f} (sd {dh[:, 0].std():.2f}), stress->calm mean {dh[:, 1].mean():+.2f} (sd {dh[:, 1].std():.2f})"
                 f"\nfiltered accuracy: static {a_s:.3f}  TVTP {a_t:.3f}  (gain {a_t - a_s:+.3f})\n")
        if scen == "null":
            gate.append(("size: rejection <= 10% under null", rej <= 0.10, f"{rej:.0%}"))
        if scen == "strong":
            bias = (dh.mean(0) - np.array([d, -d])) / d
            gate += [("power: rejection >= 80% under strong", rej >= 0.80, f"{rej:.0%}"),
                     ("delta |rel. bias| < 30% under strong", bool((np.abs(bias) < 0.30).all()), f"{np.round(bias, 2).tolist()}"),
                     ("accuracy gain >= +5 pp under strong", a_t - a_s >= 0.05, f"{a_t - a_s:+.3f}")]
    L.append("## Criteria (DESIGN §6.1b)")
    L += [f"{'PASS' if ok else 'FAIL'}  {n}: {v}" for n, ok, v in gate]
    txt = "\n".join(L)
    (ROOT / "experiments/results/tvtp_sim_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    jobs = [(sc, r) for r in range(R) for sc in SCEN]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for i, j in enumerate(ex.map(run, jobs), 1):
            print(f"{i}/{len(jobs)} {j}", flush=True)
    summary()
