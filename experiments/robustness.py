"""DESIGN §8.8: robustness and economic-intuition battery for MSGJ1-t (the production VaR/ES model).

Pass/fail uses 2008-2019 only; holdout (2020+) numbers are reported next to them, NOT blind (approved 2026-09-28).
Stage 1: F19 fit (+ no-jump fit, c = 1.5 / 3 fits), 20-start optimizer check, validation re-runs (rolling 5y / 8y,
R = 63, no jumps). Stage 2: 200 parametric-bootstrap refits from F19, 200 bootstrap LRs under the no-jump fit.
Every job caches to experiments/results/robust/<job>.pkl, so a rerun resumes.

Run: python3 experiments/robustness.py [workers]      Summary only: python3 experiments/robustness.py --summary
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
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from phase4_bench import ALPHAS, LADDER, TARGETS, Mix, MixT, berkowitz, cat, dm, kupiec_christoffersen, score

OUT = ROOT / "experiments/results/robust"
DEST = ROOT / "experiments/results/robustness_summary.md"
B, BURN, WORKERS = 200, 500, 7
SPEC = LADDER["MSGJ1-t"]
VARIANTS = {"roll5": (5 * 252, 21), "roll8": (8 * 252, 21), "R63": (None, 63), "nojump": (None, 21)}


def data():
    import daily
    y, d = daily.load()
    i0h = int((d.index < daily.HOLD0).sum())
    st = daily.seed(y, d, "MSGJ1-t")
    return y, d, i0h, st


def spec(sig_min, **kw):
    return replace(SPEC, sig_min=sig_min, **kw)


def cached(name, fn):
    path = OUT / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    res = fn()
    pickle.dump(res, open(path, "wb"))
    return res


def simulate(p, T, seed):
    """One series of length T from p, after BURN discarded days, started at the unconditional sds."""
    from model import horizon
    return horizon(p, np.asarray(p["sig"])[0], BURN + T, 1, seed=seed, path=True)[BURN:, 0]


# ---------- jobs (run in spawned workers) ----------

def job(name):
    from fit import fit
    y, d, i0h, st = data()
    y19, sm = y[:i0h], st["sig_min"]
    if name == "F19":
        return cached(name, lambda: fit(y19, spec(sm), n_starts=6)[:2])
    if name == "F19nj":
        return cached(name, lambda: fit(y19, spec(sm, jumps=""), n_starts=6)[:2])
    if name.startswith("c="):
        c = float(name[2:])
        return cached(name, lambda: fit(y19, spec(sm, c=c), n_starts=3, init=job("F19")[0])[:2])
    if name == "A1":
        return cached(name, lambda: fit(y19, spec(sm), n_starts=20, seed=1)[2])
    if name.startswith("A2_"):
        i = int(name[3:])
        def run():
            p0 = job("F19")[0]
            return fit(simulate(p0, len(y19), seed=i), spec(sm), n_starts=3, init=p0)[0]
        return cached(name, run)
    if name.startswith("B3_"):
        i = int(name[3:])
        def run():
            pj, pn = job("F19")[0], job("F19nj")[0]
            ys = simulate(pn, len(y19), seed=10_000 + i)
            llj = fit(ys, spec(sm), n_starts=3, init=pj)[1]
            lln = fit(ys, spec(sm, jumps=""), n_starts=3, init=pn)[1]
            return 2 * (llj - lln)
        return cached(name, run)
    if name in VARIANTS:
        return cached(name, lambda: oos(y, st["starts"][0], i0h, sm, *VARIANTS[name], jumps="" if name == "nojump" else "EFC"))
    raise ValueError(name)


def oos(y, i0, n, sm, W, R, jumps):
    """Validation 1-day forecasts under a variant: window W (None = expanding), refit every R, jump set."""
    from fit import fit
    from model import garch_sig, predictive
    s, rng, p, parts, params = spec(sm, jumps=jumps), np.random.default_rng(0), None, {k: [] for k in TARGETS}, []
    for b0 in range(i0, n, R):
        b1 = min(b0 + R, n)
        p, _, _ = fit(y[0 if W is None else max(0, b0 - W):b0], s, n_starts=6 if p is None else 3, init=p)
        sig = np.asarray(garch_sig(p, y[:b1]))[b0:b1]
        for k, w in TARGETS.items():
            parts[k].append(score(MixT(*predictive(p, np.ones((b1 - b0, 1)), w, sig), p["nu"]), y[b0:b1] @ w, rng))
        params.append(p)
    return {k: cat(v) for k, v in parts.items()}, params


# ---------- checks (main process) ----------

def var1(p, y, t, a=0.01):
    """1-day VaR for day t from params p (filter on y[:t]), for E, F, USD."""
    from model import garch_sig, predictive
    sig = np.asarray(garch_sig(p, np.vstack([y[:t], np.zeros((1, 2))])))[-1:]
    return np.array([MixT(*predictive(p, np.ones((1, 1)), w, sig), p["nu"]).var(a)[0] for w in TARGETS.values()])


def uncond_var(p):
    """Per-day variance of E and F implied by p: diffusion level + compound-Poisson jump variance."""
    lam = np.asarray(p["lam"])[0]
    return np.array([np.asarray(p["sig"])[0, i] ** 2 + lam[i] * (p["jm"][i] ** 2 + p["js"][i] ** 2)
                     + lam[2] * (p["mC"][i] ** 2 + p["sC"][i] ** 2) for i in range(2)])


def jump_any(p, y):
    """P(at least one jump on day t | y_t), K = 1."""
    from model import _terms
    lt, n = _terms(p, y)
    lt = np.asarray(lt)[:, 0]
    post = np.exp(lt - logsumexp(lt, axis=1, keepdims=True))
    return post[:, n.sum(1) > 0].sum(1)


def persistence(p):
    return np.asarray(p["alpha"])[0] + np.asarray(p["gamma"])[0] / 2 + np.asarray(p["beta"])[0]


def scalars(p):
    """Named scalar parameters for tables."""
    g = {k: np.asarray(v) for k, v in p.items() if k not in ("Q", "peg")}
    out = {"nu": float(g["nu"]), "rho": float(g["rho"][0]), "rhoC": float(g["rhoC"])}
    for i, s in enumerate("EF"):
        for k in ("mu", "sig", "alpha", "gamma", "beta"):
            out[f"{k}_{s}"] = float(g[k][0, i])
        out[f"jm_{s}"], out[f"js_{s}"], out[f"mC_{s}"], out[f"sC_{s}"] = (float(g[k][i]) for k in ("jm", "js", "mC", "sC"))
        out[f"pers_{s}"] = float(persistence(p)[i])
    out["lam_E"], out["lam_F"], out["lam_C"] = (float(v) for v in g["lam"][0])
    return out


def summary():
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    import daily
    from event_scoring import flag, gjr_sd, pct_expanding
    from model import horizon
    from phase4_bench import blocks
    from tenday import tail
    y, d, i0h, st = data()
    y19, i0v = y[:i0h], st["starts"][0]
    nv = sum(b < i0h for b in st["starts"])  # validation refits
    F19, ll19 = job("F19")
    P = scalars(F19)
    res, L = {"F19": P, "ll19": ll19}, [f"# §8.8 robustness battery (MSGJ1-t), F19 = fit on 2008-08-01 → 2019-12-31", ""]
    verdict = {}

    # A1 optimizer
    lls = np.array(job("A1"))
    near = int((lls >= lls.max() - 0.5).sum())
    verdict["A1"] = near >= 10 and abs(lls.max() - ll19) <= 0.5
    res["A1"] = lls
    L += [f"**A1** 20 starts: {near} within 0.5 of the best ({lls.max():.2f}); F19 {ll19:.2f}. "
          f"Spread of all starts: {lls.min():.1f} … {lls.max():.1f}.", ""]

    # A2 bootstrap
    boot = pd.DataFrame([scalars(job(f"A2_{i}")) for i in range(B)])
    res["A2"] = boot
    core = [f"{k}_{s}" for k in ("alpha", "gamma", "beta") for s in "EF"] + ["rho", "nu"]
    L += ["**A2** parametric bootstrap, B = 200 (90% percentile CIs):", "",
          "| Param | F19 | boot median | 5% | 95% | |median − F19| / CI width |", "|---|---|---|---|---|---|"]
    ok = True
    for k in boot.columns:
        lo, med, hi = boot[k].quantile([0.05, 0.5, 0.95])
        r = abs(med - P[k]) / (hi - lo) if hi > lo else np.nan
        if k in core:
            ok &= r < 0.5
        L.append(f"| {k}{'' if k in core else ' (reported)'} | {P[k]:.4g} | {med:.4g} | {lo:.4g} | {hi:.4g} | {r:.2f} |")
    verdict["A2"] = bool(ok)
    L.append("")

    # A3 refit stability + revisions
    ps = st["params"]
    pers = np.array([persistence(p) for p in ps])
    nus = np.array([float(p["nu"]) for p in ps])
    rev = []
    for k in range(1, len(ps)):
        t = st["starts"][k]
        rev.append(np.abs(var1(ps[k], y, t) / var1(ps[k - 1], y, t) - 1))
    rev = np.array(rev)
    res["A3"] = {"pers": pers, "nu": nus, "rev": rev, "dates": d.index[st["starts"]]}
    pv, rvv = slice(0, nv), rev[: nv - 1]
    verdict["A3"] = bool((pers[pv] < 1).all() and ((nus[pv] > 4) & (nus[pv] < 30)).all()
                         and np.median(rvv) < 0.05 and np.quantile(rvv, 0.95) < 0.15)
    for lab, sl, rv in (("validation", pv, rvv), ("holdout (not blind)", slice(nv, None), rev[nv - 1:])):
        L.append(f"**A3** {lab}: max persistence E {pers[sl, 0].max():.4f}, F {pers[sl, 1].max():.4f}; "
                 f"nu {nus[sl].min():.1f} … {nus[sl].max():.1f}; VaR revision at refits median "
                 f"{100 * np.median(rv):.2f}%, 95th pct {100 * np.quantile(rv, 0.95):.2f}%, max {100 * rv.max():.1f}%.")
    L.append("")

    # A4 10-day MC error
    org = np.linspace(i0v, i0h - 11, 10).astype(int)
    rel = {k: [] for k in daily.TEN}
    for t in org:
        k = max(i for i, b in enumerate(st["starts"]) if b <= t)
        from model import garch_sig
        sd0 = np.asarray(garch_sig(ps[k], np.vstack([y[:t], np.zeros((1, 2))])))[-1, 0]
        v = {n: [] for n in daily.TEN}
        for sd in range(20):
            X = horizon(ps[k], sd0, 10, 100_000, seed=1000 + sd)
            for n in daily.TEN:
                v[n].append(tail(X @ TARGETS[n])["var0.01"])
        for n in daily.TEN:
            rel[n].append(np.std(v[n]) / abs(np.mean(v[n])))
    res["A4"] = rel
    verdict["A4"] = all(max(r) < 0.01 for r in rel.values())
    L += ["**A4** 10-day 1% VaR, relative Monte Carlo sd over 20 seeds (max over 10 origins): "
          + ", ".join(f"{n} {100 * max(r):.2f}%" for n, r in rel.items()), ""]

    # B1, B2, B3-DM: validation re-runs vs production
    prod = pickle.load(open(ROOT / "experiments/results/phase4/MSGJ1-t.pkl", "rb"))["targets"]
    res["B"] = {}
    L += ["**B1/B2/B3** validation log score, DM of production vs variant (> 0 = production better), and coverage:", "",
          "| Variant | DM E | DM F | DM USD | Kupiec rejections (of 6) |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        tg, _ = job(v)
        dms = [dm(prod[k]["ls"] - tg[k]["ls"]) for k in TARGETS]
        rej = sum(kupiec_christoffersen(tg[k]["y"] <= tg[k][f"var{a}"], a)[0] < 0.05 for k in TARGETS for a in ALPHAS)
        res["B"][v] = dms
        L.append(f"| {v} | {dms[0]:.2f} | {dms[1]:.2f} | {dms[2]:.2f} | {rej} |")
    verdict["B1"] = all(x > -1.96 for v in ("roll5", "roll8") for x in res["B"][v])
    verdict["B2"] = all(x > -1.96 for x in res["B"]["R63"])
    L.append("")

    # B3 bootstrap LR
    Fnj, llnj = job("F19nj")
    lr = 2 * (ll19 - llnj)
    lrb = np.array([job(f"B3_{i}") for i in range(B)])
    pb = (1 + (lrb >= lr).sum()) / (B + 1)
    dm_nj = res["B"]["nojump"]
    res["B3"] = {"lr": lr, "lrb": lrb, "p": pb}
    jumps_ok = pb < 0.05 and sum(x > 0 for x in dm_nj) >= 2
    L += [f"**B3** jumps vs none on F19 data: LR {lr:.2f}, bootstrap p {pb:.3f} (B = {B}; null LR 95th pct "
          f"{np.quantile(lrb, 0.95):.2f}); validation DM {', '.join(f'{x:.2f}' for x in dm_nj)} → "
          f"**jumps {'supported' if jumps_ok else 'NOT supported'}**.", ""]

    # B4 sub-period coverage (validation pass/fail; holdout reported)
    hold = pickle.load(open(daily.RUNS["MSGJ1-t"][1], "rb"))
    per = {"2013–15": ("2013", "2015"), "2016–17": ("2016", "2017"), "2018–19": ("2018", "2019"),
           "2020–21": ("2020", "2021"), "2022–23": ("2022", "2023"), "2024–26": ("2024", "2026")}
    alld = {k: {c: np.concatenate([prod[k][c], hold["targets"][k][c]]) for c in ("y", "var0.01", "var0.025", "pit", "ls")}
            for k in TARGETS}
    dates = d.index[i0v:i0v + len(alld["E"]["y"])]
    L += ["**B4** hit rates by sub-period (Kupiec p in brackets):", "", "| Period | " + " | ".join(
        f"{k} {a:.1%}" for k in TARGETS for a in ALPHAS) + " |", "|---" * 7 + "|"]
    rej_v, res["B4"] = 0, {}
    for lab, (a0, a1) in per.items():
        m = (dates >= a0) & (dates <= f"{a1}-12-31")
        cells = []
        for k in TARGETS:
            for a in ALPHAS:
                I = alld[k]["y"][m] <= alld[k][f"var{a}"][m]
                p = kupiec_christoffersen(I, a)[0]
                rej_v += (p < 0.05) and a1 <= "2019"
                cells.append(f"{I.mean():.2%} ({p:.2f})")
                res["B4"][(lab, k, a)] = (I.mean(), p)
        L.append(f"| {lab}{'' if a1 <= '2019' else ' (not blind)'} | " + " | ".join(cells) + " |")
    verdict["B4"] = rej_v <= 2
    L += ["", f"Validation rejections: {rej_v} of 18.", ""]

    # B5 jump floor c
    L += ["**B5** jump-size floor c (reported):", "", "| c | loglik | λ_E | λ_F | λ_C | js_E | js_F | 1% VaR E (2019-12-31) |",
          "|---|---|---|---|---|---|---|---|"]
    res["B5"] = {}
    for c, (p, ll) in (("2", (F19, ll19)), ("1.5", job("c=1.5")), ("3", job("c=3"))):
        q = scalars(p)
        v = var1(p, y, i0h)[0]
        res["B5"][c] = (ll, q, v)
        L.append(f"| {c} | {ll:.2f} | {q['lam_E']:.4f} | {q['lam_F']:.4f} | {q['lam_C']:.4f} | {q['js_E']:.4f} | "
                 f"{q['js_F']:.4f} | {100 * v:.2f}% |")
    L.append("")

    # C1-C4 signs
    refit = pd.DataFrame([scalars(p) for p in ps])
    res["refit"] = refit
    def sign_check(k, sgn):
        lo, hi = boot[k].quantile([0.05, 0.95])
        ci = (lo > 0) if sgn > 0 else (hi < 0)
        return bool(ci and (sgn * refit[k][:nv] > 0).all()), (lo, hi), float((sgn * refit[k][nv:] > 0).mean())
    L.append("**C1–C4** economic signs (CI from A2; refit shares = share of refits with the expected sign):")
    L.append("")
    for c_, k, sgn, lab in (("C1", "gamma_E", 1, "leverage γ_E > 0"), ("C2", "gamma_F", 1, "lira asymmetry γ_F > 0"),
                            ("C3", "rho", -1, "ρ < 0")):
        ok, (lo, hi), hshare = sign_check(k, sgn)
        verdict[c_] = ok
        L.append(f"- {c_} {lab}: F19 {P[k]:.3f}, 90% CI [{lo:.3f}, {hi:.3f}], validation refits "
                 f"{(sgn * refit[k][:nv] > 0).mean():.0%}, holdout refits {hshare:.0%}.")
    c4v = ((refit["mC_E"][:nv] < 0) & (refit["mC_F"][:nv] > 0)).mean()
    c4h = ((refit["mC_E"][nv:] < 0) & (refit["mC_F"][nv:] > 0)).mean()
    verdict["C4"] = bool(P["mC_E"] < 0 < P["mC_F"] and c4v >= 0.9)
    L += [f"- C4 co-jump m_C,E < 0 < m_C,F: F19 ({P['mC_E']:.4f}, {P['mC_F']:.4f}); validation refits {c4v:.0%}, "
          f"holdout refits {c4h:.0%}. FX-only jump mean jm_F: F19 {P['jm_F']:.4f}, holdout refits > 0: "
          f"{(refit['jm_F'][nv:] > 0).mean():.0%}.", ""]

    # C5 jump days
    ev = pd.read_csv(ROOT / "data/events.csv", parse_dates=["onset"])
    pj = jump_any(F19, y19)
    top = np.argsort(-pj)[:10]
    pos = {o: int(np.searchsorted(d.index, o)) for o in ev["onset"] if d.index[0] <= o <= d.index[i0h - 1]}
    rows, hits = [], 0
    for t in sorted(top):
        e = next((ev.loc[ev.onset == o, "event"].iat[0] for o, i in pos.items() if i - 5 <= t < i + 20), "")
        hits += bool(e)
        rows.append((d.index[t].date(), pj[t], 100 * y19[t, 0], 100 * y19[t, 1], e))
    verdict["C5"] = hits >= 4
    res["C5"] = {"rows": rows, "pj": pj}
    L += [f"**C5** 10 highest P(any jump) days, 2008-08 → 2019: {hits} of 10 in an event window.", "",
          "| Date | P(jump) | rE % | rF % | Event window |", "|---|---|---|---|---|",
          *[f"| {a} | {b:.3f} | {c:.2f} | {e_:.2f} | {f} |" for a, b, c, e_, f in rows], ""]
    # holdout jump days (reported): production params of the block
    ph = []
    for k in range(nv, len(ps)):
        b0, b1 = st["starts"][k], min(st["starts"][k] + 21, len(y))
        ph.append(jump_any(ps[k], y[:b1])[b0:b1])
    ph = np.concatenate(ph)
    th = np.argsort(-ph)[:10] + i0h
    L += ["Holdout (not blind), real-time params: " + "; ".join(
        f"{d.index[t].date()} (rE {100 * y[t, 0]:.1f}%, rF {100 * y[t, 1]:.1f}%)" for t in sorted(th)), ""]
    res["C5h"] = [(d.index[t].date(), y[t]) for t in sorted(th)]

    # C6 unconditional sd
    # By simulation: jump shocks also enter the GJR recursion, so sig^2 + jump variance (uncond_var, the first,
    # wrong computation, kept for the record) understates the unconditional variance.
    uv = horizon(F19, np.asarray(F19["sig"])[0], 3000, 4000, seed=7, path=True)[500:].reshape(-1, 2).std(0)
    uv0 = np.sqrt(uncond_var(F19))
    ss = y19.std(0)
    verdict["C6"] = bool((np.abs(uv / ss - 1) < 0.2).all())
    res["C6"] = (uv, ss, uv0)
    L += [f"**C6** implied daily sd (simulated, 10M days) E {100 * uv[0]:.2f}% vs sample {100 * ss[0]:.2f}%; F "
          f"{100 * uv[1]:.2f}% vs {100 * ss[1]:.2f}%. First computation (σ² + jump variance, omits jump feedback "
          f"into GJR; a bug): E {100 * uv0[0]:.2f}%, F {100 * uv0[1]:.2f}%.", ""]

    # C7 news impact + FX share (reported; data kept for figures)
    from model import garch_sig
    h0 = np.asarray(garch_sig(F19, np.vstack([y19, np.zeros((1, 2))])))[-1, 0] ** 2
    shocks = np.linspace(-0.08, 0.08, 161)
    a_, g_, b_ = (np.asarray(F19[k])[0] for k in ("alpha", "gamma", "beta"))
    om = np.asarray(F19["sig"])[0] ** 2 * (1 - a_ - g_ / 2 - b_)
    dd = np.array([-1.0, 1.0])
    nic = {i: np.sqrt(om[i] + (a_[i] + g_[i] * (dd[i] * shocks > 0)) * shocks**2 + b_[i] * h0[i]) for i in range(2)}
    share, sdates = [], []
    for k, p in enumerate(ps):
        b0, b1 = st["starts"][k], min(st["starts"][k] + 21, len(y))
        s_ = np.asarray(garch_sig(p, y[:b1]))[b0:b1, 0]
        q = scalars(p)
        jE, jF = uncond_var(p) - np.asarray(p["sig"])[0] ** 2
        vE, vF = s_[:, 0] ** 2 + jE, s_[:, 1] ** 2 + jF
        cv = q["rho"] * s_[:, 0] * s_[:, 1] + q["lam_C"] * (q["rhoC"] * q["sC_E"] * q["sC_F"] + q["mC_E"] * q["mC_F"])
        share.append((vF - cv) / (vE + vF - 2 * cv))
        sdates.append(d.index[b0:b1])
    res["C7"] = {"shocks": shocks, "nic": nic, "share": np.concatenate(share), "dates": np.concatenate(sdates), "nv": i0h - i0v}
    sh = res["C7"]["share"]
    L += [f"**C7** FX share of XU100-in-USD variance: validation median {np.median(sh[:i0h - i0v]):.0%}, holdout median "
          f"{np.median(sh[i0h - i0v:]):.0%}, max {sh.max():.0%} on {pd.Timestamp(res['C7']['dates'][sh.argmax()]).date()}.", ""]

    # D1-D3 PIT diagnostics
    L += ["**D1–D3** PIT diagnostics (p-values):", "", "| Sample | Target | Berkowitz | LB z | LB z² | ARCH-LM |",
          "|---|---|---|---|---|---|"]
    res["D"], okD1, okD2, okD3 = {}, True, True, True
    for lab, src in (("validation", prod), ("holdout (not blind)", hold["targets"])):
        for k in TARGETS:
            z = norm.ppf(np.clip(src[k]["pit"], 1e-6, 1 - 1e-6))
            pb_ = berkowitz(src[k]["pit"])
            lb = acorr_ljungbox(z, lags=[10])["lb_pvalue"].iat[0]
            lb2 = acorr_ljungbox(z**2, lags=[10])["lb_pvalue"].iat[0]
            ar = het_arch(z, nlags=5)[1]
            res["D"][(lab, k)] = (pb_, lb, lb2, ar)
            if lab == "validation":
                okD1 &= pb_ >= 0.01
                okD2 &= lb >= 0.01 and lb2 >= 0.01
                okD3 &= ar >= 0.01
            L.append(f"| {lab} | {k} | {pb_:.3f} | {lb:.3f} | {lb2:.3f} | {ar:.3f} |")
    verdict.update(D1=bool(okD1), D2=bool(okD2), D3=bool(okD3))
    L.append("")

    # D4 hits by flag state
    fl = daily.seed(y, d, "MSGJ1")
    nh = i0h + len(hold["dates"])
    g = gjr_sd(y[:nh], blocks(i0v, i0h) + blocks(i0h, nh), fl["params"])[:, 0]
    f = flag(pct_expanding(g, i0v), daily.TAU, daily.M)  # f[j]: flag at the close of session i0v + j
    on = np.r_[False, f[:-1]]  # state when the forecast for session i0v + j was made
    res["D4"] = {}
    L += ["**D4** 1% / 2.5% VaR hit rate by flag state (flag at the previous close):", "",
          "| Sample | Target | flag on: days | 1% | 2.5% | flag off: 1% | 2.5% |", "|---|---|---|---|---|---|---|"]
    for lab, m in (("validation", np.arange(len(on)) < i0h - i0v), ("holdout (not blind)", np.arange(len(on)) >= i0h - i0v)):
        for k in TARGETS:
            hit = {a: alld[k]["y"][:len(on)] <= alld[k][f"var{a}"][:len(on)] for a in ALPHAS}
            r = [(hit[a][m & on].mean(), hit[a][m & ~on].mean()) for a in ALPHAS]
            res["D4"][(lab, k)] = (int((m & on).sum()), r)
            L.append(f"| {lab} | {k} | {(m & on).sum()} | {r[0][0]:.2%} | {r[1][0]:.2%} | {r[0][1]:.2%} | {r[1][1]:.2%} |")
    L.append("")

    L += ["## Pre-registered checks (§8.8)", "", "| # | Pass |", "|---|---|",
          *[f"| {k} | {'yes' if v else 'NO'} |" for k, v in verdict.items()],
          f"| B3 (finding) | jumps {'supported' if jumps_ok else 'not supported'} |", ""]
    res["verdict"], res["jumps_ok"] = verdict, jumps_ok
    DEST.write_text("\n".join(L) + "\n")
    pickle.dump(res, open(OUT / "summary.pkl", "wb"))
    print("\n".join(L))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summary" not in sys.argv:
        w = int(sys.argv[1]) if len(sys.argv) > 1 else WORKERS
        with ProcessPoolExecutor(w, mp_context=get_context("spawn")) as ex:
            list(ex.map(job, ["F19", "F19nj"]))
            list(ex.map(job, ["A1", *VARIANTS, "c=1.5", "c=3"] + [f"A2_{i}" for i in range(B)] + [f"B3_{i}" for i in range(B)]))
    summary()
