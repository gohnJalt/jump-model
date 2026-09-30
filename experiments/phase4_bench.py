"""Phase 4: benchmark ladder + external benchmarks, real-time out-of-sample on the validation period.

Protocol (DESIGN §4.4): expanding window, refit every R sessions; the forecast for day t uses params fitted on data
before its block and the filter run through t-1. Scored on 2013-01-01 .. 2019-12-31 (§8.1). The holdout is never loaded.
Targets: E = XU100 TRY (rE), F = USDTRY (rF), USD = XU100 in USD (rE - rF).
Metrics (§7.1-7.2): log score, PIT (Berkowitz, Ljung-Box), VaR/ES at 1% and 2.5% (Kupiec, Christoffersen,
Acerbi-Szekely Z2), FZ0 loss, Diebold-Mariano vs REF (per target) and vs DCC-t (bivariate log score).

Run: python3 experiments/phase4_bench.py [workers]      Summary only: python3 experiments/phase4_bench.py --summary
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
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from fit import Spec, default_sig_min, fit
from model import garch_sig, hamilton, log_dens, predictive

OUT = ROOT / "experiments/results/phase4"
VAL0, VAL1, R, NSIM, HS_WIN = "2013-01-01", "2019-12-31", 21, 1000, 500
ALPHAS = (0.01, 0.025)
TARGETS = {"E": np.array([1.0, 0.0]), "F": np.array([0.0, 1.0]), "USD": np.array([1.0, -1.0])}
LADDER = {"GBM": Spec(1, ""), "Merton": Spec(1), "MS-GBM2": Spec(2, ""), "MS-GBM3": Spec(3, ""),
          "MS-JD2": Spec(2), "MS-JD3": Spec(3),
          "MSGJ1": Spec(1, garch=True), "MSGJ2": Spec(2, garch=True), "MSGJ3": Spec(3, garch=True),  # §12 Q9
          "TVTP-MSGJ2": Spec(2, garch=True, n_cov=1),
          "MSGJ1-t": Spec(1, garch=True, t=True)}  # §8.5 v2 candidate  # §12 Q10 (c): transitions driven by lagged log CDS
REF = "GJR-t"  # univariate reference for DM tests (the bar after Phase 4 round 1); bivariate reference is DCC-t
EXTERNAL = ("GARCH-t", "GJR-t", "EWMA", "HS", "DCC-t")


def load():
    d = pd.read_parquet(ROOT / "data/processed/returns.parquet")[:VAL1]
    return d[["rE", "rF"]].to_numpy(), d.index, int((d.index < VAL0).sum())


def load_x(dates, cols=("z_cds",)):
    """TVTP covariates on the returns calendar (src/data.py covariates(): lagged, burn-in standardized)."""
    x = pd.read_parquet(ROOT / "data/processed/covariates.parquet").reindex(dates)[list(cols)]
    assert not x.isna().any().any(), "covariates missing on some sessions: rerun src/data.py"
    return x.to_numpy()


def blocks(i0, n):
    return [(b, min(b + R, n)) for b in range(i0, n, R)]


# ---------- predictive distributions (per day) ----------

class Mix:
    """Gaussian mixture per day: weights W (T,C), means/sds (C,) or (T,C)."""

    def __init__(self, W, m, sd):
        self.W, self.m, self.sd = W, np.broadcast_to(m, W.shape), np.broadcast_to(sd, W.shape)

    def cdf(self, y):
        return (self.W * norm.cdf((y[:, None] - self.m) / self.sd)).sum(1)

    def logpdf(self, y):
        return logsumexp(np.log(self.W + 1e-300) + norm.logpdf(y[:, None], self.m, self.sd), axis=1)

    def var(self, a):  # vectorized bisection on the cdf
        lo, hi = (self.m - 15 * self.sd).min(1), (self.m + 15 * self.sd).max(1)
        for _ in range(60):
            mid = (lo + hi) / 2
            below = self.cdf(mid) < a
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2

    def es(self, a, v):  # E[Y | Y <= v]
        z = (v[:, None] - self.m) / self.sd
        return (self.W * (self.m * norm.cdf(z) - self.sd * norm.pdf(z))).sum(1) / a

    def sample(self, rng, n):
        cw, u = self.W.cumsum(1), rng.random((len(self.W), n))
        idx = np.array([np.searchsorted(cw[t], u[t] * cw[t, -1]) for t in range(len(cw))]).clip(max=self.W.shape[1] - 1)
        rows = np.arange(len(cw))[:, None]
        return self.m[rows, idx] + self.sd[rows, idx] * rng.standard_normal(idx.shape)


class MixT(Mix):
    """Mixture of standardized Student-t(nu) components (§8.5): sd is each component's sd, scale k = sd·sqrt((nu-2)/nu)."""

    def __init__(self, W, m, sd, nu):
        super().__init__(W, m, sd)
        self.nu, self.k = float(nu), self.sd * np.sqrt((float(nu) - 2) / float(nu))

    def cdf(self, y):
        return (self.W * student_t.cdf((y[:, None] - self.m) / self.k, self.nu)).sum(1)

    def logpdf(self, y):
        return logsumexp(np.log(self.W + 1e-300) + student_t.logpdf((y[:, None] - self.m) / self.k, self.nu)
                         - np.log(self.k), axis=1)

    def es(self, a, v):  # E[Y 1{Y<=v}] per component: m F(z) - k f(z) (nu + z^2) / (nu - 1)
        z = (v[:, None] - self.m) / self.k
        tail = self.m * student_t.cdf(z, self.nu) - self.k * student_t.pdf(z, self.nu) * (self.nu + z**2) / (self.nu - 1)
        return (self.W * tail).sum(1) / a

    def sample(self, rng, n):
        cw, u = self.W.cumsum(1), rng.random((len(self.W), n))
        idx = np.array([np.searchsorted(cw[t], u[t] * cw[t, -1]) for t in range(len(cw))]).clip(max=self.W.shape[1] - 1)
        rows = np.arange(len(cw))[:, None]
        return self.m[rows, idx] + self.k[rows, idx] * rng.standard_t(self.nu, idx.shape)


class T:
    """Location + unit-variance Student-t(ν) scaled by sd (arch's standardized t)."""

    def __init__(self, loc, sd, nu):
        self.loc, self.k, self.nu = loc, sd * np.sqrt((nu - 2) / nu), nu

    def cdf(self, y):
        return student_t.cdf((y - self.loc) / self.k, self.nu)

    def logpdf(self, y):
        return student_t.logpdf((y - self.loc) / self.k, self.nu) - np.log(self.k)

    def var(self, a):
        return self.loc + self.k * student_t.ppf(a, self.nu)

    def es(self, a, v):
        q = student_t.ppf(a, self.nu)
        return self.loc - self.k * student_t.pdf(q, self.nu) / a * (self.nu + q**2) / (self.nu - 1)

    def sample(self, rng, n):
        return self.loc[:, None] + self.k[:, None] * rng.standard_t(self.nu, (len(self.loc), n))


class Emp:
    """Historical simulation: the previous HS_WIN returns, equally weighted. No density, so no log score."""

    def __init__(self, win):
        self.win = win

    def cdf(self, y):
        return (self.win <= y[:, None]).mean(1)

    def logpdf(self, y):
        return np.full(len(y), np.nan)

    def var(self, a):
        return np.quantile(self.win, a, axis=1)

    def es(self, a, v):
        w = np.where(self.win <= v[:, None], self.win, np.nan)
        return np.nanmean(w, axis=1)

    def sample(self, rng, n):
        return self.win[np.arange(len(self.win))[:, None], rng.integers(0, self.win.shape[1], (len(self.win), n))]


def score(dist, y, rng):
    """Per-day outputs for one block/target: PIT, log score, VaR/ES per α, and NSIM null draws for the Z2 test."""
    out = {"y": y, "pit": dist.cdf(y), "ls": dist.logpdf(y), "sims": dist.sample(rng, NSIM)}
    for a in ALPHAS:
        v = dist.var(a)
        out[f"var{a}"], out[f"es{a}"] = v, dist.es(a, v)
    return out


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------- model jobs ----------

def run_ladder(name, y, i0, x=None):
    s = replace(LADDER[name], sig_min=default_sig_min(y[:i0]))  # floor fixed from the burn-in window
    rng, p, parts, biv, params = np.random.default_rng(0), None, {k: [] for k in TARGETS}, [], []
    xb = lambda b: None if x is None else x[:b]
    for b0, b1 in blocks(i0, len(y)):
        p, _, _ = fit(y[:b0], s, n_starts=6 if p is None else 3, init=p, x=xb(b0))
        _, _, pred = hamilton(p, y[:b1], x=xb(b1))
        pred = np.asarray(pred)[b0:b1]
        sig = np.asarray(garch_sig(p, y[:b1]))[b0:b1] if s.garch else None
        biv.append(np.asarray(logsumexp(np.log(pred) + np.asarray(log_dens(p, y[:b1]))[b0:b1], axis=1)))
        for k, w in TARGETS.items():
            args = predictive(p, pred, w, sig)
            parts[k].append(score(MixT(*args, p["nu"]) if "nu" in p else Mix(*args), y[b0:b1] @ w, rng))
        params.append(p)
    return {k: cat(v) for k, v in parts.items()}, np.concatenate(biv), params


def _arch_block(x, b0, b1, o):
    """Univariate GARCH(1,1)/GJR-t on x[:b0] (percent units), filtered with fixed params through b1-1.
    Returns loc, sd (in return units) for days b0..b1-1, nu, standardized residuals for days 0..b1-1."""
    from arch import arch_model
    kw = dict(mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t")
    res = arch_model(100 * x[:b0], **kw).fit(disp="off")
    fx = arch_model(100 * x[:b1], **kw).fix(res.params)
    sd = np.asarray(fx.conditional_volatility)  # at t: forecast made at t-1
    loc = res.params["mu"]
    return loc / 100, sd[b0:b1] / 100, res.params["nu"], (100 * x[:b1] - loc) / sd, sd / 100


def run_garch(name, y, i0):
    rng, out = np.random.default_rng(1), {}
    for k, w in TARGETS.items():
        x, parts = y @ w, []
        for b0, b1 in blocks(i0, len(y)):
            loc, sd, nu, _, _ = _arch_block(x, b0, b1, o=1 if name == "GJR-t" else 0)
            parts.append(score(T(np.full(b1 - b0, loc), sd, nu), x[b0:b1], rng))
        out[k] = cat(parts)
    return out, None, None


def run_ewma(y, i0, lam=0.94):
    """RiskMetrics: zero mean, Gaussian, EWMA variance."""
    rng, out = np.random.default_rng(2), {}
    for k, w in TARGETS.items():
        x = y @ w
        v = np.empty(len(x))
        v[0] = x[:60].var()
        for t in range(1, len(x)):
            v[t] = lam * v[t - 1] + (1 - lam) * x[t - 1] ** 2
        n = len(x) - i0
        out[k] = score(Mix(np.ones((n, 1)), np.zeros((n, 1)), np.sqrt(v[i0:])[:, None]), x[i0:], rng)
    return out, None, None


def run_hs(y, i0):
    from numpy.lib.stride_tricks import sliding_window_view
    rng, out = np.random.default_rng(3), {}
    for k, w in TARGETS.items():
        x = y @ w
        win = sliding_window_view(x, HS_WIN)[i0 - HS_WIN:len(x) - HS_WIN]  # row t = x[t-HS_WIN : t]
        out[k] = score(Emp(win), x[i0:], rng)
    return out, None, None


def _dcc(e, a, b):
    """DCC(1,1) correlation path; R[t] uses e up to t-1."""
    S = np.corrcoef(e.T)
    Q, rho = S.copy(), np.empty(len(e))
    for t in range(len(e)):
        rho[t] = Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1])
        Q = (1 - a - b) * S + a * np.outer(e[t], e[t]) + b * Q
    return rho


def _mvt_logpdf(e, rho, nu):
    """Unit-variance bivariate Student-t log density with correlation rho."""
    q = (e[:, 0] ** 2 - 2 * rho * e[:, 0] * e[:, 1] + e[:, 1] ** 2) / (1 - rho**2)
    return (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) - 0.5 * np.log(1 - rho**2)
            - (nu + 2) / 2 * np.log1p(q / (nu - 2)))


def run_dcc(y, i0):
    """Two-step DCC-t: GARCH(1,1)-t marginals, then DCC(1,1) + bivariate t(ν) on standardized residuals.
    Scores the bivariate log density and the USD target (a linear combination of a bivariate t is t(ν))."""
    rng, parts, biv = np.random.default_rng(4), [], []
    for b0, b1 in blocks(i0, len(y)):
        mE, sE, _, eE, hE = _arch_block(y[:, 0], b0, b1, 0)
        mF, sF, _, eF, hF = _arch_block(y[:, 1], b0, b1, 0)
        e = np.column_stack([eE, eF])
        nll = lambda th: -_mvt_logpdf(e[:b0], _dcc(e[:b0], *th[:2]), th[2]).sum()
        th = minimize(nll, [0.03, 0.95, 8.0], method="L-BFGS-B", bounds=[(1e-6, 0.3), (0.5, 0.999), (2.1, 100)]).x
        if th[0] + th[1] >= 1:
            th[:2] = th[:2] / (th[0] + th[1]) * 0.999
        rho = _dcc(e, *th[:2])[b0:b1]
        biv.append(_mvt_logpdf(e[b0:b1], rho, th[2]) - np.log(sE) - np.log(sF))
        sd = np.sqrt(sE**2 + sF**2 - 2 * rho * sE * sF)
        parts.append(score(T(np.full(b1 - b0, mE - mF), sd, th[2]), y[b0:b1, 0] - y[b0:b1, 1], rng))
    return {"USD": cat(parts)}, np.concatenate(biv), None


def run(name):
    path = OUT / f"{name}.pkl"
    if path.exists():
        return name
    y, dates, i0 = load()
    fn = {"GARCH-t": run_garch, "GJR-t": run_garch}.get(name)
    if name in LADDER:
        res = run_ladder(name, y, i0, load_x(dates) if LADDER[name].n_cov else None)
    elif fn:
        res = fn(name, y, i0)
    else:
        res = {"EWMA": run_ewma, "HS": run_hs, "DCC-t": run_dcc}[name](y, i0)
    tg, biv, params = res
    pickle.dump({"name": name, "dates": dates[i0:], "targets": tg, "biv": biv, "params": params}, open(path, "wb"))
    return name


# ---------- tests ----------

def kupiec_christoffersen(I, a):
    n, n1 = len(I), I.sum()
    pi = n1 / n
    uc = -2 * (xlogy(n - n1, 1 - a) + xlogy(n1, a) - xlogy(n - n1, 1 - pi) - xlogy(n1, pi))
    i0, i1 = I[:-1], I[1:]
    n00, n01, n10, n11 = ((~i0 & ~i1).sum(), (~i0 & i1).sum(), (i0 & ~i1).sum(), (i0 & i1).sum())
    p01, p11, p = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1), (n01 + n11) / (n - 1)
    ind = -2 * (xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
                - xlogy(n00, 1 - p01) - xlogy(n01, p01) - xlogy(n10, 1 - p11) - xlogy(n11, p11))
    return chi2.sf(uc, 1), chi2.sf(ind, 1), chi2.sf(uc + ind, 2)


def z2(y, v, e, a):
    """Acerbi-Szekely Z2 in return space (ES < 0). 0 under H0; negative = tail losses worse than forecast."""
    return 1 - ((y * (y <= v)) / (len(y) * a * e)).sum(axis=0)


def fz0(y, v, e, a):
    return -((y <= v) * (v - y)) / (a * e) + v / e + np.log(-e) - 1


def berkowitz(pit):
    z = norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6))
    X = np.column_stack([np.ones(len(z) - 1), z[:-1]])
    beta, *_ = np.linalg.lstsq(X, z[1:], rcond=None)
    s2 = ((z[1:] - X @ beta) ** 2).mean()
    ll1 = norm.logpdf(z[1:], X @ beta, np.sqrt(s2)).sum()
    return chi2.sf(2 * (ll1 - norm.logpdf(z[1:]).sum()), 3)


def dm(d):
    """Diebold-Mariano t-stat of mean(d), Newey-West HAC."""
    d = d[np.isfinite(d)]
    n, L = len(d), int(4 * (len(d) / 100) ** (2 / 9))
    u = d - d.mean()
    v = u @ u / n + 2 * sum((1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n for l in range(1, L + 1))
    return d.mean() / np.sqrt(v / n)


# ---------- summary ----------

def summary(out=OUT, dest=ROOT / "experiments/results/phase4_summary.md",
            title=f"Phase 4: validation-period out-of-sample ({VAL0} .. {VAL1})"):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    M = {f.stem: pickle.load(open(f, "rb")) for f in sorted(out.glob("*.pkl"))}
    order = [m for m in list(LADDER) + list(EXTERNAL) if m in M]
    L = [f"# {title}, refit every {R} sessions\n"]
    for k in TARGETS:
        rows, ref = [], M.get(REF, {}).get("targets", {}).get(k)
        for m in order:
            r = M[m]["targets"].get(k)
            if r is None:
                continue
            u = r["pit"] - 0.5
            row = {"model": m, "logscore": np.mean(r["ls"]),
                   "DM_ls_vs_ref": dm(r["ls"] - ref["ls"]) if ref is not None and m != REF and np.isfinite(r["ls"]).any() else np.nan,
                   "PIT_Berk_p": berkowitz(r["pit"]),
                   "PIT_LB_p": acorr_ljungbox(u, [10])["lb_pvalue"].iloc[0],
                   "PIT2_LB_p": acorr_ljungbox(u**2, [10])["lb_pvalue"].iloc[0]}
            for a in ALPHAS:
                y, v, e = r["y"], r[f"var{a}"], r[f"es{a}"]
                I = y <= v
                _, _, cc = kupiec_christoffersen(I, a)
                zs = z2(r["sims"], v[:, None], e[:, None], a)
                zobs = z2(y, v, e, a)
                row[f"hit{a}"] = I.mean()
                row[f"Kupiec{a}"] = kupiec_christoffersen(I, a)[0]
                row[f"CC{a}"] = cc
                row[f"Z2_p{a}"] = (zs <= zobs).mean()
                f = fz0(y, v, e, a)
                row[f"FZ0_{a}"] = np.nanmean(f)
                row[f"DM_FZ0_{a}"] = (dm(fz0(ref["y"], ref[f"var{a}"], ref[f"es{a}"], a) - f)
                                      if ref is not None and m != REF else np.nan)
            rows.append(row)
        tab = pd.DataFrame(rows).set_index("model")
        L += [f"\n## Target {k}", tab.round(4).to_string()]
    ref = M.get("DCC-t", {}).get("biv")
    rows = [{"model": m, "biv_logscore": M[m]["biv"].mean(),
             "DM_vs_DCC": dm(M[m]["biv"] - ref) if ref is not None and m != "DCC-t" else np.nan}
            for m in order if M[m]["biv"] is not None]
    if rows:
        L += ["\n## Bivariate log score (E, F jointly)", pd.DataFrame(rows).set_index("model").round(4).to_string()]
    L.append(f"\nDM > 0: model beats the reference ({REF} per target, DCC-t bivariate); |DM| > 1.96 ≈ 5% two-sided.")
    txt = "\n".join(L)
    dest.write_text(txt)
    print(txt)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = sorted(list(LADDER) + list(EXTERNAL), key=lambda m: m not in ("MSGJ3", "MSGJ2", "MS-JD3", "MS-JD2", "MS-GBM3"))  # slow first
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for name in ex.map(run, jobs):
            print("done", name, flush=True)
    summary()
