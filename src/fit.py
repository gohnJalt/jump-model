"""Estimation (DESIGN §3.4, §4): exact MLE, L-BFGS-B on the Hamilton-filter likelihood, multi-start, regime ordering.

Ladder (D4) via Spec: GBM = Spec(1, ""), Merton JD = Spec(1), MS-GBM = Spec(K, ""), MS-JD = Spec(K).
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import expit
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from scipy.special import logit

from model import grid, hamilton

LAM_MAX = 25 / 252  # §3.4: at most 25 jumps/yr per component
PERS_MAX = 0.999    # GARCH persistence alpha + gamma/2 + beta, strictly stationary
NU_MIN, NU_MAX = 2.1, 200.0  # Student-t dof: finite variance, and ~Gaussian at the top


@dataclass(frozen=True)
class Spec:
    K: int
    jumps: str = "EFC"               # active jump components: E equity-only, F FX-only, C co-jump
    peg: bool = False                # D11 pre-float FX indicator (needs a `pre` mask)
    c: float = 2.0                   # jump sd floor, in multiples of the calmest diffusion sd (§3.4)
    sig_min: tuple = (1e-4, 1e-4)    # diffusion sd floor; use default_sig_min(y)
    garch: bool = False              # MS-GARCH-J: per-regime GJR variance recursions (HMP), sig = unconditional level
    n_cov: int = 0                   # TVTP: number of covariates driving the transition rates (0 = static)
    t: bool = False                  # §8.5: Student-t(nu) mixture components instead of Gaussian


def default_sig_min(y, window=60):
    """§3.4: 1st percentile of the rolling sd, per series. Blocks variance collapse onto a single day."""
    return tuple(np.nanpercentile(sliding_window_view(y, window, axis=0).std(-1), 1, axis=0))


def unpack(th, s):
    """Unconstrained θ -> params dict (jax-traceable). Transforms enforce every §3.4 bound."""
    i = 0

    def take(k):
        nonlocal i
        i += k
        return th[i - k:i]

    K, smin, z = s.K, jnp.array(s.sig_min), jnp.zeros(s.K)
    p = {"mu": take(2 * K).reshape(K, 2), "sig": smin + jnp.exp(take(2 * K).reshape(K, 2)), "rho": jnp.tanh(take(K))}
    p["lam"] = jnp.column_stack([LAM_MAX * expit(take(K)) if ch in s.jumps else z for ch in "EFC"])
    floor = s.c * p["sig"].min(0)
    jm, js = [0.0, 0.0], [1.0, 1.0]
    for j, ch in enumerate("EF"):
        if ch in s.jumps:
            jm[j], js[j] = take(1)[0], floor[j] + jnp.exp(take(1)[0])
    p["jm"], p["js"] = jnp.array(jm), jnp.array(js)
    p["mC"], p["sC"], p["rhoC"] = jnp.zeros(2), jnp.ones(2), 0.0
    if "C" in s.jumps:
        p["mC"], p["sC"], p["rhoC"] = take(2), floor + jnp.exp(take(2)), jnp.tanh(take(1)[0])
    Q = jnp.zeros((K, K)).at[~np.eye(K, dtype=bool)].set(jnp.exp(take(K * (K - 1))))
    p["Q"] = Q - jnp.diag(Q.sum(1))
    p["peg"] = (take(1)[0], smin[1] + jnp.exp(take(1)[0])) if s.peg else None
    if s.garch:  # persistence in (0, PERS_MAX), split into (alpha, gamma/2, beta) shares by softmax
        pers = PERS_MAX * expit(take(2 * K).reshape(K, 2))
        sh = jax.nn.softmax(jnp.concatenate([take(4 * K).reshape(K, 2, 2), jnp.zeros((K, 2, 1))], -1), -1)
        p["alpha"], p["gamma"], p["beta"] = pers * sh[..., 0], 2 * pers * sh[..., 1], pers * sh[..., 2]
    if s.n_cov:
        off = ~np.eye(K, dtype=bool)
        p["delta"] = jnp.zeros((K, K, s.n_cov)).at[off].set(take(K * (K - 1) * s.n_cov).reshape(K * (K - 1), s.n_cov))
    if s.t:
        p["nu"] = NU_MIN + (NU_MAX - NU_MIN) * expit(take(1)[0])
    assert i == len(th)
    return p


def pack(p, s):
    """Inverse of unpack."""
    smin, floor = np.array(s.sig_min), s.c * p["sig"].min(0)
    th = [p["mu"], np.log(p["sig"] - smin), np.arctanh(p["rho"])]
    lg = lambda x: np.log(np.maximum(x, 1e-12))  # fitted params can sit exactly on a bound (warm starts)
    th += [logit(np.clip(p["lam"][:, j] / LAM_MAX, 1e-9, 1 - 1e-9)) for j, ch in enumerate("EFC") if ch in s.jumps]
    th += [[p["jm"][i], lg(p["js"][i] - floor[i])] for i, ch in enumerate("EF") if ch in s.jumps]
    if "C" in s.jumps:
        th += [p["mC"], lg(p["sC"] - floor), [np.arctanh(np.clip(p["rhoC"], -0.999999, 0.999999))]]
    th.append(np.log(p["Q"][~np.eye(s.K, dtype=bool)]))
    if s.peg:
        th.append([p["peg"][0], np.log(p["peg"][1] - smin[1])])
    if s.garch:
        pers = np.maximum(p["alpha"] + p["gamma"] / 2 + p["beta"], 1e-12)
        sh = np.stack([p["alpha"], p["gamma"] / 2, p["beta"]], -1) / pers[..., None]
        th += [logit(np.clip(pers / PERS_MAX, 1e-9, 1 - 1e-9)), lg(sh[..., :2]) - lg(sh[..., 2:])]
    if s.n_cov:
        th.append(np.asarray(p["delta"])[~np.eye(s.K, dtype=bool)])
    if s.t:
        th.append([logit(np.clip((p["nu"] - NU_MIN) / (NU_MAX - NU_MIN), 1e-9, 1 - 1e-9))])
    return np.concatenate([np.ravel(x) for x in th])


def start(y, s, pre=None):
    """Data-driven start: diffusion sd spread around the sample sd, 5 jumps/yr, ~50-session regime durations."""
    post = ~pre if s.peg else slice(None)
    sd, K = y[post].std(0), s.K
    sig = np.maximum(np.outer(np.linspace(0.6, 1.8, K) if K > 1 else [1.0], sd), 1.5 * np.array(s.sig_min))
    floor = s.c * sig.min(0)
    Q = np.full((K, K), 1 / (50 * max(K - 1, 1)))
    np.fill_diagonal(Q, 0)
    return {"mu": np.tile(y[post].mean(0), (K, 1)), "sig": sig,
            "rho": np.full(K, np.corrcoef(y[post].T)[0, 1]),
            "lam": np.column_stack([np.full(K, 5 / 252 if ch in s.jumps else 0.0) for ch in "EFC"]),
            "jm": np.zeros(2), "js": 2 * floor, "mC": np.zeros(2), "sC": 2 * floor, "rhoC": 0.0,
            "Q": Q - np.diag(Q.sum(1)),
            "peg": (y[pre, 1].mean(), max(y[pre, 1].std(), 1.5 * s.sig_min[1])) if s.peg else None} | (
        {"alpha": np.full((K, 2), 0.04), "gamma": np.full((K, 2), 0.06), "beta": np.full((K, 2), 0.88)} if s.garch else {}) | (
        {"delta": np.zeros((K, K, s.n_cov))} if s.n_cov else {}) | ({"nu": 8.0} if s.t else {})


def order_regimes(p):
    """Label switching (§3.4): sort regimes by total equity variance (diffusion + jumps)."""
    v = (p["sig"][:, 0] ** 2 + p["lam"][:, 0] * (p["js"][0] ** 2 + p["jm"][0] ** 2)
         + p["lam"][:, 2] * (p["sC"][0] ** 2 + p["mC"][0] ** 2))
    o = np.argsort(v)
    keys = ("mu", "sig", "rho", "lam", "alpha", "gamma", "beta")
    return (p | {k: p[k][o] for k in keys if k in p} | {"Q": p["Q"][np.ix_(o, o)]}
            | ({"delta": p["delta"][np.ix_(o, o)]} if "delta" in p else {}))


def _objective(y, s, pre, x=None):
    """Jitted (-loglik, exact gradient) in θ; the jump grid is fixed at the λ bound so shapes are static."""
    n = grid([LAM_MAX if ch in s.jumps else 0.0 for ch in "EFC"])
    f = jax.jit(jax.value_and_grad(lambda th: -hamilton(unpack(th, s), y, pre, n, x)[0]))

    def nll(th):
        v, g = f(jnp.asarray(th))
        ok = np.isfinite(v) and np.isfinite(g).all()
        return (float(v), np.asarray(g)) if ok else (1e10, np.zeros_like(th))
    return nll


def to_numpy(p):
    return {k: (np.asarray(v) if k != "peg" else (v if v is None else tuple(float(x) for x in v))) for k, v in p.items()}


def fit(y, s, pre=None, n_starts=10, seed=0, init=None, x=None):
    """Multi-start L-BFGS-B (§4.2). Returns (params, loglik, logliks of all starts, best first).

    A local optimum within 2 log-lik units of the best one is a warning sign (§4.2).
    init: params in y units (e.g. the previous refit) used as start 0 instead of the data-driven start.
    x: TVTP covariates (T, s.n_cov), known at the close of each day and standardized by the caller.
    ponytail: no EM start; add it if the §6.1 study shows multi-start L-BFGS recovers parameters poorly.
    """
    assert s.peg == (pre is not None and bool(pre.any())), "peg=True needs pre-float rows, and only then"
    assert s.n_cov == (0 if x is None else np.shape(x)[1]), "Spec.n_cov must match the covariate columns"
    # Optimize in standardized units: raw drifts (~1e-4/day) next to O(1) θ entries stall L-BFGS (measured).
    sd = y.std(0)
    z, sz = y / sd, replace(s, sig_min=tuple(np.array(s.sig_min) / sd))
    th0 = pack(start(z, sz, pre), sz)
    nll = _objective(jnp.asarray(z), sz, None if pre is None else jnp.asarray(pre), None if x is None else jnp.asarray(x))
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_starts):
        x0 = pack(rescale(init, 1 / sd), sz) if (i == 0 and init is not None) else th0 + (rng.normal(0, 0.5, th0.size) if i else 0)
        r = minimize(nll, x0, jac=True, method="L-BFGS-B")
        runs.append((r.fun, r.x))
    runs.sort(key=lambda r: r[0])
    jac = np.log(sd).sum() * len(y)  # change of variables back to y units
    return order_regimes(rescale(to_numpy(unpack(runs[0][1], sz)), sd)), -runs[0][0] - jac, [-f - jac for f, _ in runs]


def rescale(p, sd):
    """Params fitted on y/sd -> params for y."""
    q = p | {k: p[k] * sd for k in ("mu", "sig", "jm", "js", "mC", "sC")}
    if p["peg"] is not None:
        q["peg"] = (p["peg"][0] * sd[1], p["peg"][1] * sd[1])
    return q
