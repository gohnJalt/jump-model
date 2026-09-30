"""DESIGN §6.2 correctness checks. Run: python3 tests/test_model.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fit import Spec, fit, pack, unpack
from model import garch_sig, hamilton, horizon, jump_post, log_dens, predictive, simulate, smooth, stationary


def truth(K=2, jumps=True, peg=False):
    Q = np.full((K, K), 0.02)
    np.fill_diagonal(Q, 0)
    lam = np.full((K, 3), 0.01 if jumps else 0.0) * np.arange(1, K + 1)[:, None]
    return {"mu": np.tile([4e-4, 2e-4], (K, 1)), "sig": np.outer(np.arange(1, K + 1), [0.012, 0.006]),
            "rho": np.linspace(-0.2, -0.5, K), "lam": lam,
            "jm": np.array([-0.03, 0.02]), "js": np.array([0.03, 0.02]),
            "mC": np.array([-0.05, 0.04]), "sC": np.array([0.04, 0.03]), "rhoC": -0.6,
            "Q": Q - np.diag(Q.sum(1)), "peg": (1.5e-3, 1e-3) if peg else None}


# 1. K=1, no jumps -> exactly the bivariate Gaussian likelihood
p = truth(1, jumps=False)
y, _, _ = simulate(p, 500, seed=1)
cov = np.diag(p["sig"][0]) @ [[1, p["rho"][0]], [p["rho"][0], 1]] @ np.diag(p["sig"][0])
assert np.isclose(hamilton(p, y)[0], multivariate_normal(p["mu"][0], cov).logpdf(y).sum())

# 1b. Student-t components (§8.5): K=1, no jumps -> exactly the standardized bivariate t; nu -> inf -> Gaussian
from scipy.stats import multivariate_t
pt = p | {"nu": 5.0}
assert np.isclose(hamilton(pt, y)[0], multivariate_t(p["mu"][0], cov * 3 / 5, df=5.0).logpdf(y).sum())
assert abs(hamilton(p | {"nu": 1e7}, y)[0] - hamilton(p, y)[0]) < 1e-3
s_t = Spec(1, garch=True, t=True)
q = truth(1) | {"alpha": np.full((1, 2), 0.05), "gamma": np.full((1, 2), 0.04), "beta": np.full((1, 2), 0.9), "nu": 6.5}
q2 = unpack(pack(q, s_t), s_t)
assert np.isclose(float(q2["nu"]), 6.5) and np.allclose(q2["beta"], q["beta"])

# 2. Filter == brute force over all regime paths (K=2, jumps, peg on the first two days)
p, pre = truth(2, peg=True), np.array([1, 1, 0, 0, 0], bool)
y, _, _ = simulate(p, 5, seed=2, pre=pre)
lf, P, pi = np.asarray(log_dens(p, y, pre)), expm(p["Q"]), np.asarray(stationary(p["Q"]))
paths = [np.log(pi[s[0]]) + sum(np.log(P[a, b]) for a, b in zip(s, s[1:])) + lf[range(5), s].sum()
         for s in itertools.product(range(2), repeat=5)]
assert np.isclose(hamilton(p, y, pre)[0], logsumexp(paths))

# 3. MS-GBM special case == statsmodels MarkovRegression (switching mean + variance), at statsmodels' own MLE.
#    FX is regime-invariant with rho=0, so our loglik = statsmodels' equity loglik + the FX Gaussian loglik.
import statsmodels.api as sm

p = truth(2, jumps=False) | {"rho": np.zeros(2)}
p["sig"][:, 1] = 0.006
y, _, _ = simulate(p, 1500, seed=3)
res = sm.tsa.MarkovRegression(y[:, 0], k_regimes=2, switching_variance=True).fit(disp=False)
v = dict(zip(res.model.param_names, res.params))
q = p | {"mu": np.array([[v["const[0]"], 0], [v["const[1]"], 0]]),
         "sig": np.array([[v["sigma2[0]"] ** 0.5, 0.006], [v["sigma2[1]"] ** 0.5, 0.006]]),
         "Q": logm(res.model.regime_transition_matrix(res.params)[:, :, 0].T).real}
ll, filt, pred = hamilton(q, y)
assert np.isclose(ll - norm(0, 0.006).logpdf(y[:, 1]).sum(), res.llf, atol=1e-4), (ll, res.llf)
assert np.abs(smooth(q, filt, pred) - res.smoothed_marginal_probabilities).max() < 1e-4

# 4. pack/unpack round trip, with every component and the peg
s = Spec(3, "EFC", peg=True, sig_min=(1e-3, 5e-4))
th = pack(unpack(np.random.default_rng(4).normal(size=pack(truth(3, peg=True), s).size), s), s)
assert np.allclose(pack(unpack(th, s), s), th)

# 5. Jump posteriors find big, rare equity jumps
p = truth(1, jumps=False) | {"lam": np.array([[0.02, 0, 0]]), "jm": np.array([-0.08, 0]), "js": np.array([0.02, 1])}
y, _, n = simulate(p, 3000, seed=5)
hit = jump_post(p, y, hamilton(p, y)[1])[:, 0] > 0.5
true = n[:, 0] > 0
assert (hit & true).sum() / hit.sum() > 0.9 and (hit & true).sum() / true.sum() > 0.9

# 5b. Predictive mixture of w'y has the analytic mean (all jump components on)
p = truth(2)
pred = np.array([[0.3, 0.7]])
for w in ([1, 0], [0, 1], [1, -1]):
    W, m, sd = predictive(p, pred, np.array(w, float))
    ey = (pred[0][:, None] * (p["mu"] + p["lam"][:, :1] * [p["jm"][0], 0] + p["lam"][:, 1:2] * [0, p["jm"][1]]
                             + p["lam"][:, 2:] * p["mC"])).sum(0) @ w
    assert np.isclose((W[0] * m).sum(), ey, rtol=1e-4), ((W[0] * m).sum(), ey)

# 7. MS-GARCH-J: (a) alpha=gamma=beta=0 is exactly the static model; (b) the JAX GJR recursion + filter match an
#    independent numpy/scipy implementation (K=1, no jumps, rho=0 -> product of two univariate GJR-normal densities)
from scipy.stats import norm as N

p = truth(2)
y, _, _ = simulate(p, 300, seed=8)
g0 = p | {k: np.zeros((2, 2)) for k in ("alpha", "gamma", "beta")}
assert np.isclose(hamilton(g0, y)[0], hamilton(p, y)[0])

p = truth(1, jumps=False) | {"rho": np.zeros(1), "alpha": np.array([[0.05, 0.08]]), "gamma": np.array([[0.10, 0.04]]),
                             "beta": np.array([[0.85, 0.80]])}
y, _, _ = simulate(p, 400, seed=9)
v, ll = p["sig"][0] ** 2, 0.0
h, d = v.copy(), np.array([-1.0, 1.0])
om = v * (1 - p["alpha"][0] - p["gamma"][0] / 2 - p["beta"][0])
for t in range(len(y)):
    ll += N.logpdf(y[t], p["mu"][0], np.sqrt(h)).sum()
    u = y[t] - p["mu"][0]
    h = om + (p["alpha"][0] + p["gamma"][0] * (d * u > 0)) * u**2 + p["beta"][0] * h
assert np.isclose(hamilton(p, y)[0], ll), (hamilton(p, y)[0], ll)
s = Spec(2, garch=True, sig_min=(1e-3, 5e-4))
q = truth(2) | {"alpha": np.full((2, 2), 0.05), "gamma": np.full((2, 2), 0.04), "beta": np.full((2, 2), 0.9)}
assert np.allclose(pack(unpack(pack(q, s), s), s), pack(q, s))
sig = garch_sig(q, y)[-5:]
W, m, sd = predictive(q, np.full((5, 2), 0.5), np.array([1.0, -1.0]), sig=np.asarray(sig))
assert W.shape == sd.shape == (5, W.shape[1])

# 8. TVTP: (a) delta = 0 equals the static model; (b) filter == brute force over paths with time-varying P_t;
#    (c) pack/unpack round trip with GARCH + TVTP; (d) smoother rows sum to 1
from model import P_path
from scipy.linalg import expm as sexpm

p = truth(2)
x = np.random.default_rng(10).normal(size=(300, 1))
y, _, _ = simulate(p, 300, seed=10)
assert np.isclose(hamilton(p | {"delta": np.zeros((2, 2, 1))}, y, x=x)[0], hamilton(p, y)[0])

p = truth(2) | {"delta": np.array([[[0.0], [1.2]], [[-0.8], [0.0]]])}
x = np.random.default_rng(11).normal(size=(5, 1))
y, _, _ = simulate(p, 5, seed=11, x=x)
lf = np.asarray(log_dens(p, y))
Pt = [sexpm(np.where(np.eye(2, dtype=bool), 0, p["Q"] * np.exp(p["delta"][..., 0] * x[t, 0]))
            - np.diag((np.where(np.eye(2, dtype=bool), 0, p["Q"] * np.exp(p["delta"][..., 0] * x[t, 0]))).sum(1)))
      for t in range(5)]
assert np.allclose(np.asarray(P_path(p, x)), Pt)
w, v = np.linalg.eig(Pt[0].T)
pi0 = np.real(v[:, np.argmin(np.abs(w - 1))]); pi0 /= pi0.sum()
paths = [np.log(pi0[s[0]]) + sum(np.log(Pt[t][s[t], s[t + 1]]) for t in range(4)) + lf[range(5), s].sum()
         for s in itertools.product(range(2), repeat=5)]
ll, filt, pred = hamilton(p, y, x=x)
assert np.isclose(ll, logsumexp(paths))
assert np.allclose(smooth(p, filt, pred, x=x).sum(1), 1)

s = Spec(2, garch=True, n_cov=2, sig_min=(1e-3, 5e-4))
q = truth(2) | {"alpha": np.full((2, 2), 0.05), "gamma": np.full((2, 2), 0.04), "beta": np.full((2, 2), 0.9),
                "delta": np.array([[[0.0, 0.0], [0.5, -0.3]], [[-0.7, 0.2], [0.0, 0.0]]])}
assert np.allclose(pack(unpack(pack(q, s), s), s), pack(q, s))

# 5b. 10-day Monte Carlo (§8.6): H=1 matches the exact predictive mixture; no GARCH/jumps -> sum is N(10 mu, 10 cov)
from scipy.stats import t as student_t
q = truth(1) | {"alpha": np.full((1, 2), 0.05), "gamma": np.full((1, 2), 0.04), "beta": np.full((1, 2), 0.9), "nu": 6.0}
y, _, _ = simulate(truth(1), 300, seed=5)
sig1 = np.asarray(garch_sig(q, np.vstack([y, np.zeros((1, 2))])))[-1:]
for w in (np.array([1.0, 0.0]), np.array([1.0, -1.0])):
    W, m, sd = predictive(q, np.ones((1, 1)), w, sig1)
    x = np.sort(horizon(q, sig1[0, 0], H=1, N=400_000, seed=1) @ w)
    cdf = (W * student_t.cdf((x[[4000, 10000]][:, None] - m) / (sd * np.sqrt(4 / 6)), 6)).sum(1)
    assert np.allclose(cdf, [0.01, 0.025], atol=1.5e-3), cdf
g0 = truth(1, jumps=False) | {k: np.zeros((1, 2)) for k in ("alpha", "gamma", "beta")}
x = horizon(g0, g0["sig"][0], H=10, N=200_000, seed=2)
cov = np.diag(g0["sig"][0]) @ [[1, g0["rho"][0]], [g0["rho"][0], 1]] @ np.diag(g0["sig"][0])
assert np.allclose(x.mean(0), 10 * g0["mu"][0], atol=3e-4) and np.allclose(np.cov(x.T), 10 * cov, rtol=0.02)

# 6. Fit: K=1 Gaussian reproduces the closed-form MLE; MS-JD fit beats the true parameters' loglik
p = truth(1, jumps=False)
y, _, _ = simulate(p, 1000, seed=6)
f, _, _ = fit(y, Spec(1, ""), n_starts=1)
assert np.allclose(f["mu"][0], y.mean(0), atol=1e-5) and np.allclose(f["sig"][0], y.std(0), rtol=1e-3)

p = truth(2)
y, _, _ = simulate(p, 2000, seed=7)
s = Spec(2, sig_min=(1e-3, 5e-4))
f, llf, lls = fit(y, s, n_starts=1)
assert llf >= hamilton(p, y)[0] - 1e-6, (llf, hamilton(p, y)[0])
assert fit(y, s, n_starts=1, init=f)[1] >= llf - 1e-3  # warm start from the optimum stays there
print("ok  | MS-JD K=2 T=2000: loglik fit", round(llf, 1), "truth", round(hamilton(p, y)[0], 1), "starts", np.round(lls, 1))
