"""Rational play: what an optimising client would do.

A belief is an array of CDF values on a shared price grid, so the same
functions serve the prior and the Bayesian posteriors from Step 4.

Code is for a rational client
"""
import numpy as np
from scipy import stats
from env import play, score

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


# ---------- Step 5: the rational dealer ----------

def prior_F_weights(rs, m, g, sigma_ln=0.5):
    """F0 as probability mass at each limit in rs: r = m + g*LogNormal(0, sigma_ln)."""
    u = (rs - m) / g                                      # how far above m, in g
    dens = np.zeros_like(u, dtype=float)
    inside = u > 0
    dens[inside] = stats.lognorm(s=sigma_ln, scale=1.0).pdf(u[inside])
    return dens / dens.sum()


def dealer_expected_profit(c, p_grid, F_w, theta, q_star):
    """Expected profit of each candidate quote, averaged over the client's limit.

    A client with limit r lifts p iff p <= theta(r); otherwise it counters
    q*(r), which the dealer takes iff q* >= c. theta, q_star and F_w are
    aligned on the same limits.
    """
    counter_profit = np.where(q_star >= c, q_star - c, 0.0)             # per limit
    lifted = p_grid[:, None] <= theta[None, :]                          # (prices, limits)
    payoff = np.where(lifted, p_grid[:, None] - c, counter_profit[None, :])
    return payoff @ F_w


def dealer_best_quote(m, c, g, sigma_ln=0.5):
    """p*: the dealer's best quote, holding F0 over r and assuming the client holds G0."""
    q = price_grid(m, g)
    rs = limit_grid(m, g)
    theta, q_star = client_policy(rs, q, prior_G(q, m, g, sigma_ln))
    ev = dealer_expected_profit(c, q, prior_F_weights(rs, m, g, sigma_ln), theta, q_star)
    ev = np.where(q >= c, ev, -np.inf)                    # never quote below cost
    return float(q[np.argmax(ev)])

# ---------- Step 5: predictions U* and B* ----------

def rational_bots(q_grid, G_client):
    """Rational play in env.play's bot interface.

    Dealer quotes p*; the client, holding G_client, lifts iff p <= theta and
    otherwise counters q*; the dealer takes any counter at or above its cost.
    """
    def quote(dv, g):
        return dealer_best_quote(dv.m, dv.c, g)

    def respond(cv, g, p):
        V, q_star = value_of_countering(cv.r, q_grid, G_client)
        return ("accept", None) if p <= cv.r - V else ("counter", q_star)

    def decide(dv, g, q):
        return "accept" if q >= dv.c else "reject"

    return quote, respond, decide


def play_rational(scn, g, G_client):
    """One rational episode, played against the TRUE r and c, and scored."""
    q = price_grid(scn.m, g)
    return score(scn, play(scn, g, *rational_bots(q, G_client)))


def predictions(scn, g, G_band):
    """U* (client holds the prior), B* (client holds the band posterior), and VoD_s."""
    q = price_grid(scn.m, g)
    U = play_rational(scn, g, prior_G(q, scn.m, g))
    B = play_rational(scn, g, G_band)
    return U, B, B.client_surplus - U.client_surplus
