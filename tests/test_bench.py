"""Checks for the Phase 4 predictive distributions and tests. Run: python3 tests/test_bench.py"""
import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from phase4_bench import MixT, Mix, T, dm, kupiec_christoffersen, z2

a, n = 0.01, 3
# Mix with one component == Normal: VaR by bisection, ES closed form
d = Mix(np.ones((n, 1)), np.zeros((n, 1)), np.full((n, 1), 2.0))
v = d.var(a)
assert np.allclose(v, norm.ppf(a, 0, 2)) and np.allclose(d.es(a, v), -2 * norm.pdf(norm.ppf(a)) / a)

# T: ES closed form == numerical E[Y | Y <= VaR]
d = T(np.array([0.1]), np.array([1.5]), 5.0)
v = d.var(a)
num = integrate.quad(lambda x: x * np.exp(d.logpdf(np.array([x]))[0]), -np.inf, v[0])[0] / a
assert np.isclose(d.es(a, v)[0], num, rtol=1e-6), (d.es(a, v), num)

# MixT (§8.5): one component == T; a 2-component mixture's VaR hits a and ES matches Monte Carlo
d1, d2 = MixT(np.ones((1, 1)), np.array([[0.1]]), np.array([[1.5]]), 5.0), T(np.array([0.1]), np.array([1.5]), 5.0)
v = d1.var(a)
assert np.allclose(v, d2.var(a), atol=1e-8) and np.allclose(d1.es(a, v), d2.es(a, v), rtol=1e-6)
dm2 = MixT(np.array([[0.7, 0.3]]), np.array([[0.0, -1.0]]), np.array([[1.0, 3.0]]), 4.0)
v = dm2.var(a)
draws = dm2.sample(np.random.default_rng(1), 2_000_000)[0]
assert np.isclose(dm2.cdf(v)[0], a) and np.isclose(draws[draws <= v[0]].mean(), dm2.es(a, v)[0], rtol=0.02)

# Under H0 (y drawn from the forecast), Z2 is centered at 0 and hits are ~α
rng = np.random.default_rng(0)
d = T(np.zeros(20000), np.ones(20000), 5.0)
y = d.sample(rng, 1)[:, 0]
v = d.var(a)
assert abs(z2(y, v, d.es(a, v), a)) < 0.15 and kupiec_christoffersen(y <= v, a)[0] > 0.01
assert abs(dm(rng.standard_normal(5000))) < 3
print("ok")
