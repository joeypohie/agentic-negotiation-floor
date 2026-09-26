"""Step 4 — the client's Bayesian belief about the dealer's cost.

A belief is an array of CDF values on the price grid (see rational.py), so
whatever this module returns drops straight into value_of_countering and
client_policy.

    U:  the client has no evidence, so its correct belief is the prior G0.
    B:  it has the band, so its correct belief is prior x band likelihood.
"""
import numpy as np
from scipy import stats

from rational import prior_G


def prior_density_c(q_grid, m, g, sigma_ln=0.5):
    """g0(q): the prior DENSITY of the dealer's cost at each grid price.

    c = m - g*E1 with E1 ~ LogNormal(0, sigma_ln), so the density is zero at
    and above m. Bayes multiplies densities, never CDFs.
    """
    u = (m - q_grid) / g                                  # how far below m, in g
    inside = u > 0
    dens = np.zeros_like(q_grid, dtype=float)
    dens[inside] = stats.lognorm(s=sigma_ln, scale=1.0).pdf(u[inside]) / g
    return dens


def band_posterior(q_grid, m, g, avg, s, sigma_ln=0.5):
    """The client's posterior over c after seeing the band.

    Likelihood: avg | c ~ Normal(c, s^2), where s includes both the averaging
    noise and the dealer-specific deviation.

    Returns (w, G):
        w  probability mass at each grid price (sums to 1)
        G  posterior CDF at each grid price, P(c <= q)
    """
    with np.errstate(divide="ignore"):                    # log(0) = -inf is intended
        log_post = (np.log(prior_density_c(q_grid, m, g, sigma_ln))
                    - 0.5 * ((avg - q_grid) / s) ** 2)
    log_post -= log_post.max()                            # largest term becomes exp(0) = 1
    w = np.exp(log_post)
    w /= w.sum()
    G = np.cumsum(w) - w / 2                              # CDF at the grid point, not the cell's edge
    return w, G


def client_belief(q_grid, scn, g, band=None):
    """The benchmark client's belief CDF: the prior in U, the band posterior in B."""
    if band is None:
        return prior_G(q_grid, scn.m, g)
    _, G = band_posterior(q_grid, scn.m, g, band.avg, band.s)
    return G
