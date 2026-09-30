"""MS-JD core (DESIGN §3): bivariate Markov-switching Merton jump-diffusion, exact likelihood at daily frequency.

y[:, 0] = rE (log XU100, TRY), y[:, 1] = rF (log USDTRY). Units: per trading session (Δ = 1).
Params dict p, K regimes:
  mu (K,2), sig (K,2), rho (K,)   diffusion per regime
  lam (K,3)                       jump intensities per session: equity-only, FX-only, co-jump
  jm (2,), js (2,)                mean/sd of equity-only (index 0) and FX-only (index 1) jump sizes
  mC (2,), sC (2,), rhoC          co-jump size ~ N2(mC, VC)
  Q (K,K)                         CTMC generator
  peg None | (mu_F, sig_F)        D11: FX drift/diffusion on pre-float days (mask `pre`), same in every regime
  alpha, gamma, beta (K,2)        optional (MS-GARCH-J, §12 Q9): per-regime GJR recursions; sig is then the
                                  unconditional sd level, omega = sig^2 (1 - alpha - gamma/2 - beta)
  delta (K,K,d)                   optional TVTP (§12 Q10 c): q_ij(t) = Q_ij · exp(delta_ij · x_t), zero diagonal.
                                  x (T,d) are covariates known at the close of t (caller standardizes them),
                                  so Q is the generator at x = 0 and the t -> t+1 transition uses x_t.
  nu                              optional (§8.5): every mixture component is a bivariate Student-t(nu) with the
                                  same covariance as the Gaussian one (t-scale mixture, nests Gaussian as nu -> inf)
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from jax.scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import poisson

jax.config.update("jax_enable_x64", True)

TAIL = 1e-6  # omitted Poisson mass per component (§3.3)


def stationary(Q):
    K = Q.shape[0]
    A = jnp.vstack([Q.T, jnp.ones(K)])
    return jnp.linalg.lstsq(A, jnp.r_[jnp.zeros(K), 1.0])[0]


def grid(lam_max):
    """Jump-count combos (M,3); each component truncated where its Poisson tail < TAIL at lam_max (3,).
    Static (numpy) so the likelihood jit-compiles; fit.py sizes it at the λ bound."""
    n = [range(int(poisson.isf(TAIL, l)) + 1) if l > 0 else range(1) for l in np.asarray(lam_max)]
    return np.array(list(itertools.product(*n)), float)


def P_path(p, x):
    """TVTP transition matrices P_t = expm(Q(x_t)), shape (T,K,K); P_t drives the move from t to t+1."""
    Q0, D = jnp.asarray(p["Q"]), jnp.asarray(p["delta"])
    off = ~np.eye(Q0.shape[0], dtype=bool)
    rates = jnp.where(off, jnp.where(off, Q0, 1.0)[None] * jnp.exp(jnp.einsum("ijd,td->tij", D, jnp.asarray(x))), 0.0)
    return jax.vmap(expm)(rates - jax.vmap(jnp.diag)(rates.sum(-1)))


def _transitions(p, T, x):
    """(P_t for t < T, initial regime distribution): stationary at x_0 for TVTP, else at Q."""
    if "delta" in p:
        assert x is not None and len(x) == T, "TVTP params need covariates x with one row per observation"
        Ps = P_path(p, x)
        return Ps, stationary(Ps[0] - jnp.eye(Ps.shape[1]))
    Q = jnp.asarray(p["Q"])
    return jnp.broadcast_to(expm(Q), (T,) + Q.shape), stationary(Q)


def garch_sig(p, y):
    """Haas-Mittnik-Paolella (2004) regime-specific GJR variance paths: each regime's h runs on all observed shocks,
    so there is no path dependence and the Hamilton filter stays exact. Returns sd (T,K,2); row t uses y up to t-1.
    Asymmetry: equity variance reacts to falls (rE < mu), FX variance to lira depreciation (rF > mu)."""
    mu, a, g, b = (jnp.asarray(p[k]) for k in ("mu", "alpha", "gamma", "beta"))
    v = jnp.asarray(p["sig"]) ** 2
    om, d = v * (1 - a - g / 2 - b), jnp.array([-1.0, 1.0])

    def step(h, yt):
        e = yt[None] - mu
        return om + (a + g * (d * e > 0)) * e**2 + b * h, jnp.sqrt(h)

    return jax.lax.scan(step, v, jnp.asarray(y))[1]


def _moments(p, mu, sig, n):
    """Mixture components per (regime k, jump counts m): log weights (K,M); means mE, mF and covariance entries
    a = Var_E, c = Var_F, b = Cov, each (1|T, K, M)."""
    nE, nF, nC = (n[:, i][None, None] for i in range(3))  # (1,1,M)
    logw = (xlogy(n[None], p["lam"][:, None]) - p["lam"][:, None] - gammaln(n + 1)[None]).sum(-1)
    sE, sF, r = sig[..., 0, None], sig[..., 1, None], p["rho"][None, :, None]
    js, sC = p["js"], p["sC"]
    a = sE**2 + nE * js[0] ** 2 + nC * sC[0] ** 2
    c = sF**2 + nF * js[1] ** 2 + nC * sC[1] ** 2
    b = r * sE * sF + nC * p["rhoC"] * sC[0] * sC[1]
    mE = mu[..., 0, None] + nE * p["jm"][0] + nC * p["mC"][0]
    mF = mu[..., 1, None] + nF * p["jm"][1] + nC * p["mC"][1]
    return logw, mE, mF, a, b, c


def predictive(p, pred, w, sig=None):
    """One-step predictive of w'y as a Gaussian mixture, given P(S_t|F_{t-1}) (T,K) and no peg.
    sig: conditional sds (T,K,2) of the forecast days for GARCH specs (rows of garch_sig); None = static.
    Returns weights (T,C) (rows sum to 1), means and sds (1,C) or (T,C), with C = K·M."""
    sig = jnp.asarray(p["sig"])[None] if sig is None else jnp.asarray(sig)
    logw, mE, mF, a, b, c = (np.asarray(v) for v in _moments(p, jnp.asarray(p["mu"])[None], sig,
                                                              grid(np.asarray(p["lam"]).max(0))))
    W = (np.asarray(pred)[:, :, None] * np.exp(logw)[None]).reshape(len(pred), -1)
    var = w[0] ** 2 * a + 2 * w[0] * w[1] * b + w[1] ** 2 * c
    C = W.shape[1]
    return W / W.sum(1, keepdims=True), (w[0] * mE + w[1] * mF).reshape(-1, C), np.sqrt(var).reshape(-1, C)


def horizon(p, sd0, H=10, N=100_000, seed=0, path=False):
    """Monte Carlo sums of the next H sessions' returns (N,2) (DESIGN §8.6). sd0 (2,): the first day's conditional
    sds, i.e. the garch_sig row for that day. path=True returns every day instead, shape (H,N,2): with N = 1 it
    simulates one series (parametric bootstrap, §8.8).
    Each day: Poisson jump counts, then that component's bivariate normal / t(nu) (same moments as _terms);
    the GJR variances update on the simulated shocks as in garch_sig."""
    assert len(p["mu"]) == 1, "K = 1 only"  # ponytail: K > 1 needs regime draws from the filter and expm(Q)
    rng = np.random.default_rng(seed)
    g = {k: np.asarray(v) for k, v in p.items() if k != "peg"}
    mu, a, gm, b, v = (g[k][0] for k in ("mu", "alpha", "gamma", "beta", "sig"))
    om, d, r = v**2 * (1 - a - gm / 2 - b), np.array([-1.0, 1.0]), g["rho"][0]
    h = np.tile(np.asarray(sd0, float) ** 2, (N, 1))
    tot, out = np.zeros((N, 2)), []
    for _ in range(H):
        n = rng.poisson(g["lam"][0], (N, 3))
        sE, sF = np.sqrt(h[:, 0]), np.sqrt(h[:, 1])
        va = h[:, 0] + n[:, 0] * g["js"][0] ** 2 + n[:, 2] * g["sC"][0] ** 2
        vc = h[:, 1] + n[:, 1] * g["js"][1] ** 2 + n[:, 2] * g["sC"][1] ** 2
        cv = r * sE * sF + n[:, 2] * g["rhoC"] * g["sC"][0] * g["sC"][1]
        m = mu + n[:, :2] * g["jm"] + n[:, 2:] * g["mC"]
        z = rng.standard_normal((N, 2))
        if "nu" in g:  # standardized bivariate t: one chi-square mixing draw per path-day
            z *= np.sqrt((g["nu"] - 2) / rng.chisquare(g["nu"], (N, 1)))
        l11 = np.sqrt(va)
        e = np.column_stack([l11 * z[:, 0], cv / l11 * z[:, 0] + np.sqrt(vc - cv**2 / va) * z[:, 1]])
        yt = m + e
        u = yt - mu
        h = om + (a + gm * (d * u > 0)) * u**2 + b * h
        tot += yt
        if path:
            out.append(yt)
    return np.array(out) if path else tot


def _terms(p, y, pre=None, n=None):
    """log[w_km · φ(y_t; m_tkm, C_tkm)] of the Poisson-weighted Gaussian mixture, shape (T,K,M), plus the grid."""
    mu, sig = p["mu"][None], p["sig"][None]  # (1,K,2): broadcast over t unless the peg overrides some rows
    if pre is not None and p["peg"] is not None:
        m = jnp.asarray(pre)[:, None]
        mu = jnp.stack([jnp.broadcast_to(mu[..., 0], (len(y), mu.shape[1])), jnp.where(m, p["peg"][0], mu[..., 1])], -1)
        sig = jnp.stack([jnp.broadcast_to(sig[..., 0], (len(y), sig.shape[1])), jnp.where(m, p["peg"][1], sig[..., 1])], -1)
    if "alpha" in p:
        assert p["peg"] is None, "peg + GARCH not supported"
        sig = garch_sig(p, y)
    n = grid(np.asarray(p["lam"]).max(0)) if n is None else n
    logw, mE, mF, a, b, c = _moments(p, mu, sig, n)
    x, z = y[:, None, None, 0] - mE, y[:, None, None, 1] - mF
    det = a * c - b**2
    q = (c * x**2 - 2 * b * x * z + a * z**2) / det
    if "nu" in p:  # standardized bivariate t: covariance a, b, c as in the Gaussian case
        nu = p["nu"]
        logphi = (gammaln((nu + 2) / 2) - gammaln(nu / 2) - jnp.log(jnp.pi * (nu - 2)) - 0.5 * jnp.log(det)
                  - (nu + 2) / 2 * jnp.log1p(q / (nu - 2)))
    else:
        logphi = -jnp.log(2 * jnp.pi) - 0.5 * jnp.log(det) - 0.5 * q
    return logw[None] + logphi, n


def log_dens(p, y, pre=None, n=None):
    """log f(y_t | S_t = k), shape (T,K)."""
    return logsumexp(_terms(p, y, pre, n)[0], axis=2)


def hamilton(p, y, pre=None, n=None, x=None):
    """Hamilton filter. Returns (loglik, filtered P(S_t|F_t), predicted P(S_t|F_{t-1})); start = stationary.
    x: covariates (T,d), required when p has TVTP "delta"."""
    lf = log_dens(p, y, pre, n)
    Ps, xi0 = _transitions(p, len(y), x)

    def step(xi_pr, inp):  # xi_pr = P(S_t | F_{t-1})
        l, P = inp
        m = l.max()
        w = xi_pr * jnp.exp(l - m)
        s = w.sum()
        return (w / s) @ P, (w / s, xi_pr, jnp.log(s) + m)

    _, (filt, pred, ll) = jax.lax.scan(step, xi0, (lf, Ps))
    return ll.sum(), filt, pred


def smooth(p, filt, pred, x=None):
    """Kim smoother: P(S_t | F_T). Diagnostics only; the risk monitor uses filtered probabilities (§1)."""
    Ps = np.asarray(_transitions(p, len(filt), x)[0])
    filt, pred = np.asarray(filt), np.asarray(pred)
    sm = filt.copy()
    for t in range(len(filt) - 2, -1, -1):
        sm[t] = filt[t] * ((sm[t + 1] / pred[t + 1]) @ Ps[t].T)
    return sm


def jump_post(p, y, probs, pre=None):
    """P(equity-only, FX-only, co-jump on day t), shape (T,3), given regime probs (filtered or smoothed)."""
    lt, n = _terms(p, y, pre)
    post = np.asarray((probs[..., None] * jnp.exp(lt - logsumexp(lt, axis=2, keepdims=True))).sum(1))  # (T,M)
    return np.column_stack([post[:, n[:, i] > 0].sum(1) for i in range(3)])


def simulate(p, T, seed=0, pre=None, x=None):
    """Draw from the discretized model (regime constant within a session). Returns y (T,2), regimes, jump counts.
    x: covariates (T,d) for TVTP params."""
    rng = np.random.default_rng(seed)
    p = {k: np.asarray(v) if k != "peg" else v for k, v in p.items()}
    Ps, pi = (np.asarray(v) for v in _transitions(p, T, x))
    K = len(p["Q"])
    s = np.empty(T, int)
    pi = np.clip(pi, 0, None)
    s[0] = rng.choice(K, p=pi / pi.sum())
    for t in range(1, T):
        pr = np.clip(Ps[t - 1][s[t - 1]], 0, None)
        s[t] = rng.choice(K, p=pr / pr.sum())
    mu, sig, r = p["mu"][s].copy(), p["sig"][s].copy(), p["rho"][s]
    if pre is not None and p["peg"] is not None:
        mu[pre, 1], sig[pre, 1] = p["peg"]
    n = rng.poisson(p["lam"][s])
    e = rng.standard_normal((T, 6))
    rc, sq = p["rhoC"], np.sqrt(n)
    z = np.column_stack([e[:, 0], r * e[:, 0] + np.sqrt(1 - r**2) * e[:, 1]])  # unit-variance diffusion shocks
    J = np.column_stack([n[:, 0] * p["jm"][0] + sq[:, 0] * p["js"][0] * e[:, 2] + n[:, 2] * p["mC"][0] + sq[:, 2] * p["sC"][0] * e[:, 4],
                         n[:, 1] * p["jm"][1] + sq[:, 1] * p["js"][1] * e[:, 3] + n[:, 2] * p["mC"][1]
                         + sq[:, 2] * p["sC"][1] * (rc * e[:, 4] + np.sqrt(1 - rc**2) * e[:, 5])])
    if "alpha" not in p:
        return mu + sig * z + J, s, n
    a, g, b, d = p["alpha"], p["gamma"], p["beta"], np.array([-1.0, 1.0])
    h = p["sig"] ** 2
    om, y = h * (1 - a - g / 2 - b), np.empty((T, 2))
    for t in range(T):  # every regime's variance runs on the realized shocks (HMP), the active one drives y
        y[t] = mu[t] + np.sqrt(h[s[t]]) * z[t] + J[t]
        u = y[t] - p["mu"]
        h = om + (a + g * (d * u > 0)) * u**2 + b * h
    return y, s, n
