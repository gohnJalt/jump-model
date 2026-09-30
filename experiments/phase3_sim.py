"""Phase 3 simulation study (DESIGN §6.1) — the go/no-go gate (§6.1a).

Probes (each job = one replication, saved to results/phase3/<probe>_<rep>.pkl, so reruns resume):
  rec2 / rec3  recovery: data from calibrated MS-JD K=2 / K=3; fit K-1..K+1 (K-selection) + metrics at true K
  garch        CCC-GARCH(1,1)-t data (no regimes, no jumps); fit K=1..4 -> spurious regimes/jumps?
  merton       K=1 jump-diffusion; fit K=1,2 -> does selection keep K=1?
  async        K=2 truth, FX sampled 6h after the equity close (as Bloomberg PX_LAST) -> co-jump bias
  disc         K=2 truth simulated in exact continuous time (regime can switch intraday) -> discretization bias

MS-GARCH-J family (§12 Q9): add --msgj. Fits Spec(K, garch=True), true params for rec2/rec3 come from
calibration_msgj.pkl, probes rec2/rec3/garch/merton only, results in results/phase3_msgj/. Same gate thresholds.

Run: python3 experiments/phase3_sim.py [workers] [--msgj]      Summary: ... --summary [--msgj]
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
import numpy as np
from scipy.linalg import expm

from fit import LAM_MAX, Spec, default_sig_min, fit, pack
from model import hamilton, jump_post, simulate, stationary

MSGJ = "--msgj" in sys.argv
OUT = ROOT / ("experiments/results/phase3_msgj" if MSGJ else "experiments/results/phase3")
T, SPLIT, STARTS = 4544, 0.7, 3
# MSGJ: no rec3 (the K=3 calibration is degenerate: a ~0-session regime with rho = -1, not a credible truth)
REPS = ({"rec2": 30, "garch": 24, "merton": 16} if MSGJ else
        {"rec2": 40, "rec3": 40, "garch": 24, "merton": 24, "async": 24, "disc": 24})
FITS = {"rec2": (1, 2, 3), "rec3": (2, 3, 4), "garch": (1, 2, 3) if MSGJ else (1, 2, 3, 4), "merton": (1, 2),
        "async": (2,), "disc": (2,)}
TRUE_K = {"rec2": 2, "rec3": 3, "merton": 1, "async": 2, "disc": 2}


def interior(p):
    """Pre-stated rule: pull calibrated boundary values inside, so bias is measurable (λ∈[1,20]/yr, |ρC|≤.9, sd≥1.25·floor;
    GARCH: alpha, gamma ≥ 0.01, persistence ≤ 0.99)."""
    p = {k: np.array(v, float) if k != "peg" else None for k, v in p.items()}
    if "alpha" in p:
        p["alpha"], p["gamma"] = np.maximum(p["alpha"], 0.01), np.maximum(p["gamma"], 0.01)
        scale = np.minimum(1, 0.99 / (p["alpha"] + p["gamma"] / 2 + p["beta"]))
        p["alpha"], p["gamma"], p["beta"] = p["alpha"] * scale, p["gamma"] * scale, p["beta"] * scale
    p["lam"] = np.clip(p["lam"], 1 / 252, 20 / 252)
    p["rhoC"] = float(np.clip(p["rhoC"], -0.9, 0.9))
    floor = 2.0 * p["sig"].min(0)
    p["js"], p["sC"] = np.maximum(p["js"], 1.25 * floor), np.maximum(p["sC"], 1.25 * floor)
    return p


def truth(K, msgj=None):
    """rec probes use the family's calibration; the GARCH and Merton probes always use the MS-JD K=1 calibration."""
    msgj = MSGJ if msgj is None else msgj
    cal = pickle.load(open(ROOT / f"experiments/results/calibration{'_msgj' if msgj else ''}.pkl", "rb"))
    return interior(cal["truth"][K][0])


# ---------- data generators ----------

def sim_garch(seed):
    """CCC-GARCH(1,1), standardized t(6) shocks; unconditional vol/corr matched to the K=1 calibration."""
    rng, p = np.random.default_rng(seed), truth(1, msgj=False)
    tot = np.sqrt(p["sig"][0] ** 2 + p["lam"][0, :2] * p["js"] ** 2 + p["lam"][0, 2] * p["sC"] ** 2)
    a, b, nu, rho = 0.08, 0.90, 6.0, -0.4
    L = np.linalg.cholesky([[1, rho], [rho, 1]])
    z = rng.standard_t(nu, (T, 2)) / np.sqrt(nu / (nu - 2)) @ L.T
    h, y = tot**2, np.empty((T, 2))
    for t in range(T):
        y[t] = np.sqrt(h) * z[t]
        h = tot**2 * (1 - a - b) + a * y[t] ** 2 + b * h
    return y


def _ctmc_fractions(Q, days, rng, sub=1):
    """Exact CTMC path; time share of each regime in each of days*sub equal slices."""
    K, n = len(Q), days * sub
    frac, t, k = np.zeros((n, K)), 0.0, rng.choice(K, p=np.clip(stationary(Q), 0, None) / np.clip(stationary(Q), 0, None).sum())
    while t < n:
        dt = rng.exponential(1 / -Q[k, k] / 1) * sub if K > 1 else n
        end = min(t + dt, n)
        i0, i1 = int(t), int(np.ceil(end))
        for i in range(i0, i1):  # ponytail: per-slice loop, fine at a few hundred switches
            frac[i, k] += min(end, i + 1) - max(t, i)
        t = end
        if K > 1:
            w = Q[k].clip(0)
            k = rng.choice(K, p=w / w.sum())
    return frac


def sim_ct(p, seed, sub=1):
    """Continuous-time MS-JD aggregated exactly over slices (regime may switch within a day).
    Returns per-slice increments (days*sub, 2), and the dominant regime + jump counts per slice."""
    rng = np.random.default_rng(seed)
    f = _ctmc_fractions(p["Q"], T + 1, rng, sub)
    dt = 1 / sub
    mu = f @ p["mu"] * dt
    vE, vF = f @ p["sig"][:, 0] ** 2 * dt, f @ p["sig"][:, 1] ** 2 * dt
    cEF = f @ (p["rho"] * p["sig"][:, 0] * p["sig"][:, 1]) * dt
    n = rng.poisson(f @ p["lam"] * dt)
    e = rng.standard_normal((len(f), 6))
    r = cEF / np.sqrt(vE * vF)
    y = mu + np.column_stack([np.sqrt(vE) * e[:, 0], np.sqrt(vF) * (r * e[:, 0] + np.sqrt(1 - r**2) * e[:, 1])])
    sq, rc = np.sqrt(n), p["rhoC"]
    y[:, 0] += n[:, 0] * p["jm"][0] + sq[:, 0] * p["js"][0] * e[:, 2] + n[:, 2] * p["mC"][0] + sq[:, 2] * p["sC"][0] * e[:, 4]
    y[:, 1] += (n[:, 1] * p["jm"][1] + sq[:, 1] * p["js"][1] * e[:, 3] + n[:, 2] * p["mC"][1]
                + sq[:, 2] * p["sC"][1] * (rc * e[:, 4] + np.sqrt(1 - rc**2) * e[:, 5]))
    return y, f.argmax(1), n


# ---------- one replication ----------

def metrics(p_hat, p, y, s_true, n_true):
    ll, filt, _ = hamilton(p_hat, y)
    jp = jump_post(p_hat, y, filt)
    jd = {}
    for i, c in enumerate("EFC"):
        hit, tru = jp[:, i] > 0.5, n_true[:, i] > 0
        jd[c] = ((hit & tru).sum() / hit.sum() if hit.any() else np.nan, (hit & tru).sum() / max(tru.sum(), 1))
    return {"p_hat": p_hat, "acc_filt": float((np.asarray(filt).argmax(1) == s_true).mean()), "jump_prec_rec": jd,
            "switches_per_yr": float((np.diff(np.asarray(filt).argmax(1)) != 0).sum() / (len(y) / 252))}


def select(y, Ks):
    """Per K: full-sample loglik, BIC, OOS predictive log score on the last 30% (fit on the first 70%)."""
    n70, res = int(SPLIT * len(y)), {}
    for K in Ks:
        s = Spec(K, sig_min=default_sig_min(y), garch=MSGJ)
        p, ll, _ = fit(y, s, n_starts=STARTS)
        p70, _, _ = fit(y[:n70], s, n_starts=STARTS)
        oos = float(hamilton(p70, y)[0] - hamilton(p70, y[:n70])[0])
        res[K] = {"p": p, "ll": ll, "bic": -2 * ll + len(pack(p, s)) * np.log(len(y)), "oos": oos,
                  "switches_per_yr": float((np.diff(np.asarray(hamilton(p, y)[1]).argmax(1)) != 0).sum() / (len(y) / 252)),
                  "lam_yr": p["lam"] * 252}
    return res


def run(job):
    probe, rep = job
    path = OUT / f"{probe}_{rep}.pkl"
    if path.exists():
        return job
    seed = 1000 * list(REPS).index(probe) + rep
    out = {"probe": probe, "rep": rep}
    if probe in ("rec2", "rec3", "merton"):
        p = truth(TRUE_K[probe], msgj=MSGJ and probe != "merton")
        y, s, n = simulate(p, T, seed=seed)
    elif probe == "garch":
        y = sim_garch(seed)
    elif probe == "disc":
        p = truth(2)
        y, s, n = sim_ct(p, seed)
        y, s, n = y[:T], s[:T], n[:T]
    elif probe == "async":
        p = truth(2)
        q, s4, n4 = sim_ct(p, seed, sub=4)  # 6h slices; equity = slices 0-3 of day t, FX = slices 1-3 of t + slice 0 of t+1
        q = q.reshape(T + 1, 4, 2)
        y = np.column_stack([q[:T, :, 0].sum(1), q[:T, 1:, 1].sum(1) + q[1:, 0, 1]])
        s, n = s4.reshape(T + 1, 4)[:T, -1], n4.reshape(T + 1, 4, 3)[:T].sum(1)
    out["sel"] = select(y, FITS[probe])
    if probe in TRUE_K:
        out["truth"] = p
        out |= metrics(out["sel"][TRUE_K[probe]]["p"], p, y, s, n)
    pickle.dump(out, open(path, "wb"))
    return job


# ---------- summary + gate ----------

def summary():
    R = {}
    for f in OUT.glob("*.pkl"):
        r = pickle.load(open(f, "rb"))
        R.setdefault(r["probe"], []).append(r)
    L, gate = [], []

    def rb(est, tru):
        est = np.array(est)
        return (est.mean(0) - tru) / np.abs(tru), np.sqrt(((est - tru) ** 2).mean(0)) / np.abs(tru)

    for probe in ("rec2", "rec3", "disc", "async"):
        if probe not in R:
            continue
        rs, p = R[probe], R[probe][0]["truth"]
        K = len(p["Q"])
        L.append(f"\n## {probe}  (true K={K}, R={len(rs)})")
        for name, get in [("sig", lambda q: q["sig"]), ("dur", lambda q: -1 / np.diag(q["Q"])), ("lam", lambda q: q["lam"])]:
            b, rm = rb([get(r["p_hat"]) for r in rs], get(p))
            L.append(f"{name:4s} truth {np.round(get(p) * (252 if name == 'lam' else np.sqrt(252) if name == 'sig' else 1), 3).tolist()}"
                     f"\n     rel.bias {np.round(b, 2).tolist()}\n     rel.RMSE {np.round(rm, 2).tolist()}")
            if probe in ("rec2", "rec3"):
                ok = (np.abs(b) < 0.15).all() if name != "lam" else ((np.abs(b) < 0.30) & (rm < 0.60)).all()
                gate.append((f"{probe} {name}", ok, f"max|bias| {np.abs(b).max():.2f}" + (f", max RMSE {rm.max():.2f}" if name == "lam" else "")))
        L.append(f"rho  truth {np.round(p['rho'], 2).tolist()} mean est {np.round(np.mean([r['p_hat']['rho'] for r in rs], 0), 2).tolist()}")
        for g in ("alpha", "gamma", "beta"):  # reported, not gated (not in the pre-registered §6.1a table)
            if g in p:
                b, rm = rb([r["p_hat"][g] for r in rs], p[g])
                L.append(f"{g:5s} truth {np.round(p[g], 3).tolist()} rel.bias {np.round(b, 2).tolist()} rel.RMSE {np.round(rm, 2).tolist()}")
        L.append(f"rhoC truth {p['rhoC']:.2f} mean est {np.mean([r['p_hat']['rhoC'] for r in rs]):.2f}")
        acc = np.mean([r["acc_filt"] for r in rs])
        prec = {c: np.nanmean([r["jump_prec_rec"][c][0] for r in rs]) for c in "EFC"}
        rec = {c: np.nanmean([r["jump_prec_rec"][c][1] for r in rs]) for c in "EFC"}
        L.append(f"filtered accuracy {acc:.3f}  | jump precision {({c: round(v, 2) for c, v in prec.items()})}  recall {({c: round(v, 2) for c, v in rec.items()})}")
        if probe in ("rec2", "rec3"):
            gate.append((f"{probe} filtered accuracy", acc >= 0.85, f"{acc:.3f}"))
            gate.append((f"{probe} jump precision", min(prec.values()) >= 0.70, str({c: round(v, 2) for c, v in prec.items()})))
    for probe in ("rec2", "rec3", "merton", "garch"):
        if probe not in R:
            continue
        for crit in ("bic", "oos"):
            pick = [min(r["sel"], key=lambda k: r["sel"][k]["bic"]) if crit == "bic" else max(r["sel"], key=lambda k: r["sel"][k]["oos"]) for r in R[probe]]
            dist = {k: pick.count(k) for k in sorted(set(pick))}
            L.append(f"{probe:6s} K picked by {crit}: {dist}")
            if probe in TRUE_K:
                share = pick.count(TRUE_K[probe]) / len(pick)
                gate.append((f"{probe} K-selection ({crit})", share >= 0.80, f"{share:.0%} correct"))
            else:
                gate.append((f"garch K-selection ({crit})", max(pick) <= 2, f"picked {dist}"))
    if "garch" in R:
        for K in (1, 2, 3):
            lam = np.array([r["sel"][K]["lam_yr"] for r in R["garch"]]).mean(0)
            sw = np.mean([r["sel"][K]["switches_per_yr"] for r in R["garch"]])
            L.append(f"garch fit K={K}: mean λ/yr E,F,C by regime {np.round(lam, 1).tolist()}  switches/yr {sw:.1f}")
    L.append("\n## GATE (DESIGN §6.1a)")
    L += [f"{'PASS' if ok else 'FAIL'}  {n}: {d}" for n, ok, d in gate]
    txt = "\n".join(L)
    (ROOT / f"experiments/results/phase3{'_msgj' if MSGJ else ''}_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 7
    jobs = [(pr, r) for r in range(max(REPS.values())) for pr, n in REPS.items() if r < n]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for i, j in enumerate(ex.map(run, jobs), 1):
            print(f"{i}/{len(jobs)} {j}", flush=True)
    summary()
