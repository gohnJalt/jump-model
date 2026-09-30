"""§8.6 10-day VaR/ES: MSGJ1-t Monte Carlo vs sqrt(10) scaling and GJR-t 10-day Monte Carlo, on validation (blind)
and the holdout (already seen, veto only). Params: the evaluated refit history (daily.seed); no MSGJ1-t refits.

Run: python3 experiments/tenday.py [workers]      Summary only: python3 experiments/tenday.py --summary
Caches forecasts in experiments/results/tenday.pkl; writes experiments/results/tenday_summary.md.
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
from scipy.stats import norm

from phase4_bench import ALPHAS, R, TARGETS, fz0

H, N, L = 10, 100_000, 18  # horizon, paths, Newey-West lags (windows overlap by H - 1)
OUT = ROOT / "experiments/results/tenday.pkl"
DEST = ROOT / "experiments/results/tenday_summary.md"


def tail(s):
    """VaR/ES per alpha from simulated sums s (N,)."""
    out = {}
    for a in ALPHAS:
        v = np.quantile(s, a)
        out[f"var{a}"], out[f"es{a}"] = v, s[s <= v].mean()
    return out


def gjr_paths(th, h0, rng):
    """arch GJR(1,1)-t in percent units: th = (mu, omega, alpha, gamma, beta, nu); sums of H days in return units."""
    mu, om, a, g, b, nu = th
    h, tot = np.full(N, h0), np.zeros(N)
    for _ in range(H):
        e = np.sqrt(h) * rng.standard_t(nu, N) * np.sqrt((nu - 2) / nu)
        tot += mu + e
        h = om + (a + g * (e < 0)) * e**2 + b * h
    return tot / 100


def job(args):
    """All origins of one refit block: MSGJ1-t Monte Carlo and GJR-t Monte Carlo per target."""
    from model import horizon
    p, sds, gjr, ts = args
    rows = []
    for t, sd0, g in zip(ts, sds, gjr):
        X = horizon(p, sd0, H, N, seed=t)
        rng = np.random.default_rng(t)
        rows.append({k: {"ms": tail(X @ w), "gjr": tail(gjr_paths(g[k], g[k + "_h0"], rng))} for k, w in TARGETS.items()})
    return ts, rows


def build(workers):
    from arch import arch_model

    import daily
    from model import garch_sig
    y, d = daily.load()
    st = daily.seed(y, d, "MSGJ1-t")  # evaluated params (validation + holdout blocks), in memory
    i0v, i0h = st["starts"][0], int((d.index < daily.HOLD0).sum())
    nh = i0h + len(pickle.load(open(daily.RUNS["MSGJ1-t"][1], "rb"))["dates"])  # end of the evaluated holdout run
    origins = [t for t in range(i0v, nh - H + 1) if t + H <= i0h or t >= i0h]  # windows never straddle 2020
    tasks = []
    for k, b0 in enumerate(st["starts"]):
        ts = [t for t in origins if b0 <= t < b0 + R]
        if not ts:
            continue
        p = st["params"][k]
        sds = np.asarray(garch_sig(p, y[:ts[-1] + 1]))[ts, 0]  # row t uses y up to t-1
        gjr = [{} for _ in ts]
        for name, w in TARGETS.items():
            x = 100 * (y @ w)
            kw = dict(mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
            res = arch_model(x[:b0], **kw).fit(disp="off")
            v = np.asarray(arch_model(x[:ts[-1] + 1], **kw).fix(res.params).conditional_volatility)
            th = tuple(res.params[c] for c in ("mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"))
            for g, t in zip(gjr, ts):
                g[name], g[name + "_h0"] = th, v[t] ** 2
        tasks.append((p, sds, gjr, ts))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        done = list(ex.map(job, tasks))
    ts = np.concatenate([t for t, _ in done])
    rows = [r for _, rr in done for r in rr]
    out = {"dates": d.index[ts], "i0h": i0h, "t": ts}
    for k, w in TARGETS.items():
        out[k] = {"y": np.array([(y[t:t + H] @ w).sum() for t in ts])}
        for m in ("ms", "gjr"):
            out[k][m] = {c: np.array([r[k][m][c] for r in rows]) for c in rows[0][k][m]}
    # sqrt(10) x the evaluated MSGJ1-t 1-day VaR/ES (validation + holdout runs)
    one = [pickle.load(open(f, "rb")) for f in daily.RUNS["MSGJ1-t"]]
    for k in TARGETS:
        cat = {c: np.concatenate([o["targets"][k][c] for o in one]) for c in one[0]["targets"][k] if c[:3] in ("var", "es0")}
        out[k]["sqrt10"] = {c: np.sqrt(H) * v[ts - i0v] for c, v in cat.items()}
    pickle.dump(out, open(OUT, "wb"))
    return out


def nw_t(u):
    """t-stat of mean(u) with Newey-West (Bartlett, L lags) variance."""
    u = u[np.isfinite(u)]
    n, e = len(u), u - u.mean()
    v = e @ e / n + 2 * sum((1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n for l in range(1, L + 1))
    return u.mean() / np.sqrt(v / n)


def summary(out):
    periods = {"Validation 2013–2019 (blind)": out["t"] < out["i0h"], "Holdout 2020+ (already seen)": out["t"] >= out["i0h"]}
    L_ = ["# §8.6 10-day VaR/ES (MSGJ1-t Monte Carlo)", "",
          f"Sum of the next {H} sessions' log returns, lower tail. {N:,} paths per origin. Tests: Newey–West HAC, "
          f"{L} lags. Hit-rate p is two-sided. DM > 0 means MSGJ1-t has the lower FZ0 loss.", ""]
    verdict = {}
    for pname, mask in periods.items():
        L_ += [f"## {pname}: {mask.sum()} overlapping windows ({out['dates'][mask][0].date()} → "
               f"{out['dates'][mask][-1].date()})", "",
               "| Target | α | Model | Hit rate | p (HAC) | mean loss beyond VaR | mean ES |", "|---|---|---|---|---|---|---|"]
        rej = {"ms": 0, "gjr": 0, "sqrt10": 0}
        for k in TARGETS:
            yk = out[k]["y"][mask]
            for a in ALPHAS:
                for m in ("ms", "sqrt10", "gjr"):
                    v, e = out[k][m][f"var{a}"][mask], out[k][m][f"es{a}"][mask]
                    I = yk <= v
                    p = 2 * norm.sf(abs(nw_t(I - a))) if I.any() else float("nan")  # no hits: HAC variance is 0
                    rej[m] += (p < 0.05) or not I.any()
                    beyond = yk[I].mean() if I.any() else float("nan")
                    L_.append(f"| {k} | {a:.1%} | {m} | {I.mean():.2%} | {p:.3f} | {100 * beyond:.2f}% | {100 * e.mean():.2f}% |")
        L_ += ["", "| Target | FZ0 DM vs GJR-t (2.5%) | FZ0 DM vs √10 (2.5%) |", "|---|---|---|"]
        dms = {}
        for k in TARGETS:
            yk, f = out[k]["y"][mask], {m: fz0(out[k]["y"][mask], out[k][m]["var0.025"][mask], out[k][m]["es0.025"][mask], 0.025)
                                        for m in ("ms", "gjr", "sqrt10")}
            dms[k] = (nw_t(f["gjr"] - f["ms"]), nw_t(f["sqrt10"] - f["ms"]))
            L_.append(f"| {k} | {dms[k][0]:.2f} | {dms[k][1]:.2f} |")
        L_ += ["", f"Coverage rejections at 5% (of 6; no hits counts as a rejection): "
                   f"MSGJ1-t {rej['ms']}, √10 {rej['sqrt10']}, GJR-t {rej['gjr']}.", ""]
        verdict[pname] = (rej, dms)
    (rv, dv), (rh, dh) = verdict.values()
    T1, T2 = rv["ms"] <= 1, all(v[0] > -1.96 for v in dv.values())
    Hc = rh["ms"] <= rh["gjr"] and all(v[0] > -1.96 for v in dh.values())
    L_ += ["## Pre-registered checks (§8.6)", "", "| # | Check | Result | Pass |", "|---|---|---|---|",
           f"| T1 | validation coverage, ≤ 1 of 6 rejections | {rv['ms']} | {'yes' if T1 else 'NO'} |",
           f"| T2 | validation FZ0 DM vs GJR-t > −1.96 on all targets | "
           f"{', '.join(f'{k} {v[0]:.2f}' for k, v in dv.items())} | {'yes' if T2 else 'NO'} |",
           f"| H | holdout: rejections ≤ GJR-t's ({rh['ms']} vs {rh['gjr']}) and T2 | "
           f"{', '.join(f'{k} {v[0]:.2f}' for k, v in dh.items())} | {'yes' if Hc else 'NO'} |", "",
           f"**{'All pass: 10-day VaR/ES ships in the daily report.' if T1 and T2 and Hc else 'Fails: 10-day VaR/ES stays out of the daily report.'}**"]
    DEST.write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary(pickle.load(open(OUT, "rb")))
    else:
        summary(pickle.load(open(OUT, "rb")) if OUT.exists() else build(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
