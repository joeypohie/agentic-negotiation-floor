"""Rational play: what an optimising client would do.

A belief is an array of CDF values on a shared price grid, so the same
functions serve the prior and the Bayesian posteriors from Step 4.
"""
import numpy as np
from scipy import stats


def price_grid(m, g, lo=6.0, hi=6.0, per_g=40):
    """Candidate prices, m - lo*g to m + hi*g, step g/per_g."""
    n = int(round((lo + hi) * per_g)) + 1
    return np.linspace(m - lo * g, m + hi * g, n)


def limit_grid(m, g, hi=6.0, n=400):
    """Candidate client limits. One-sided: r = m + g*E2 and E2 > 0."""
    return m + g * np.linspace(0.0, hi, n)


def prior_G(q_grid, m, g, sigma_ln=0.5):
    """G0(q) = P(dealer's cost <= q) under c = m - g*LogNormal(0, sigma_ln)."""
    return 1.0 - stats.lognorm(s=sigma_ln, scale=1.0).cdf((m - q_grid) / g)


def uniform_G(q_grid, a, b):
    """Uniform belief over c on [a, b]. Testing only."""
    return np.clip((q_grid - a) / (b - a), 0.0, 1.0)


def value_of_countering(r, q_grid, G_vals):
    """Best expected value from countering, and the counter achieving it."""
    ev = np.where(q_grid <= r, (r - q_grid) * G_vals, -np.inf)
    i = int(np.argmax(ev))
    return float(ev[i]), float(q_grid[i])


def threshold(r, q_grid, G_vals):
    """theta: the client accepts a quote p iff p <= theta."""
    V, _ = value_of_countering(r, q_grid, G_vals)
    return r - V


def client_policy(rs, q_grid, G_vals):
    """theta and q* at every limit in rs. Vectorised over prices."""
    ev = np.where(q_grid[None, :] <= rs[:, None],
                  (rs[:, None] - q_grid[None, :]) * G_vals[None, :],
                  -np.inf)
    idx = ev.argmax(axis=1)
    V = ev[np.arange(len(rs)), idx]
    return rs - V, q_grid[idx]
