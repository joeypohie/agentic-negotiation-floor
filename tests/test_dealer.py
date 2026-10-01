"""Step 5, part 1 — the rational dealer's quote."""
from pathlib import Path

import numpy as np
import yaml

from generator import make_scenario
from rational import (price_grid, limit_grid, prior_G, client_policy,
                      prior_F_weights, dealer_expected_profit, dealer_best_quote)

ROOT = Path(__file__).resolve().parent.parent
G = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]["g_points"]
M = 100.0


def test_expected_profit_matches_simulation():
    """The grid average over F0 must agree with simply drawing clients and playing them.

    Independent of prior_F_weights: limits come from the generator's own formula.
    """
    rng = np.random.default_rng(0)
    c = M - G
    q = price_grid(M, G)
    G0 = prior_G(q, M, G)

    rs_grid = limit_grid(M, G)
    theta, q_star = client_policy(rs_grid, q, G0)
    grid_ev = dealer_expected_profit(c, q, prior_F_weights(rs_grid, M, G), theta, q_star)

    rs_mc = M + G * rng.lognormal(0.0, 0.5, size=20_000)
    th_mc, qs_mc = client_policy(rs_mc, q, G0)
    for p in (M - 0.5 * G, M, M + 0.5 * G, M + G, M + 2 * G):
        i = int(np.argmin(np.abs(q - p)))
        pay = np.where(q[i] <= th_mc, q[i] - c, np.where(qs_mc >= c, qs_mc - c, 0.0))
        se = pay.std() / np.sqrt(len(pay))
        assert abs(grid_ev[i] - pay.mean()) < 4 * se + 0.01 * G, f"p = m{(q[i]-M)/G:+.2f}g"


# ---------- Gate (a): p* never below c ----------

def test_gate_a_quote_never_below_cost():
    for sd in range(1, 1001):
        scn = make_scenario(sd, G)
        assert dealer_best_quote(scn.m, scn.c, G) >= scn.c, f"seed {sd}"

