from pathlib import Path

import numpy as np
import pytest
import yaml

from rational import (price_grid, limit_grid, prior_G, uniform_G,
                      value_of_countering, client_policy)

ROOT = Path(__file__).resolve().parent.parent
G = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]["g_points"]
M = 100.0


@pytest.mark.parametrize("a_off,b_off,r_off", [(-2, 0, 1), (-3, -1, 2),
                                               (-1, 0, 0.5), (-4, -2, 3)])
def test_uniform_belief_matches_closed_form(a_off, b_off, r_off):
    """With a uniform belief on [a,b], calculus gives q* = (r+a)/2, clipped."""
    q = price_grid(M, G)
    step = q[1] - q[0]
    a, b, r = M + a_off*G, M + b_off*G, M + r_off*G
    _, q_star = value_of_countering(r, q, uniform_G(q, a, b))
    expected = min(max((r + a) / 2, a), b)
    assert abs(q_star - expected) <= step


def test_theta_increases_with_r():
    """dtheta/dr = 1 - G(q*), which lies in [0,1]. So theta rises, never faster than r."""
    q  = price_grid(M, G)
    rs = limit_grid(M, G)
    theta, _ = client_policy(rs, q, prior_G(q, M, G))

    slope = np.diff(theta) / np.diff(rs)
    assert np.all(slope >= -1e-9)
    assert np.all(slope <= 1 + 1e-9)
    assert np.all(theta < rs)