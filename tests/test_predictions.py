"""Step 5, part 2 — rational play at the truth, VoD, and scale invariance."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator import Scenario, make_scenario, simulate_disclosure
from rational import price_grid, dealer_best_quote, predictions
from benchmarks import band_posterior, client_belief

ROOT = Path(__file__).resolve().parent.parent
CAL = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]
G = CAL["g_points"]
D = CAL["disclosure"]
N = 1000


@pytest.fixture(scope="module")
def played():
    """U*, B* and VoD on seeds 1..N (held-out seeds start at 1001)."""
    rows = []
    for sd in range(1, N + 1):
        scn = make_scenario(sd, G)
        band = simulate_disclosure(scn, G, D["K"], D["omega"], D["tau"])
        G_band = client_belief(price_grid(scn.m, G), scn, G, band)
        rows.append((scn, *predictions(scn, G, G_band)))
    return rows


def test_rational_play_has_no_violations(played):
    """Every rational move is individually sensible, so none of the four violations can fire."""
    for scn, U, B, _ in played:
        assert not U.violations and not B.violations, (scn.seed, U.violations, B.violations)


# ---------- Gate (b): disclosure is worth at least 5% of surplus ----------

def test_gate_b_vod_is_material(played):
    vod = np.array([v for *_, v in played])
    surplus = np.array([scn.r - scn.c for scn, *_ in played])
    share = vod.sum() / surplus.sum()                       # ratio of sums
    assert vod.mean() > 0
    assert share >= 0.05, f"VoD = {share:.1%} of surplus"


# ---------- Gate (c): predictions don't depend on LAMBDA, in g units ----------

def _in_g_units(m, g, E1, E2, z):
    """Everything the benchmark predicts for one hand-built scenario, in g from m.

    Built unrounded on purpose: the generator rounds c and r to 4 dp, which is
    a different number of g at each LAMBDA, so real scenarios can't be exactly
    invariant. This gate is about the benchmark functions.
    """
    scn = Scenario(seed=0, m=m, c=m - g * E1, r=m + g * E2)
    s = g * np.sqrt(D["omega"] ** 2 / D["K"] + D["tau"] ** 2)
    q = price_grid(m, g)
    _, G_band = band_posterior(q, m, g, avg=scn.c + z * s, s=s)
    U, B, vod = predictions(scn, g, G_band)
    return np.array([(dealer_best_quote(m, scn.c, g) - m) / g,
                     U.client_surplus / g, B.client_surplus / g, vod / g])


@pytest.mark.parametrize("E1,E2,z", [(1.0, 1.0, 0.0), (0.6, 1.8, -1.2),
                                     (2.3, 0.7, 0.8), (1.4, 1.2, 1.5)])
def test_gate_c_scale_invariance(E1, E2, z):
    sigma = CAL["sigma_points"]
    base = _in_g_units(100.0, 0.1 * sigma, E1, E2, z)
    for lam, m in [(0.05, 100.0), (0.2, 100.0), (0.1, 98.37)]:
        assert np.allclose(_in_g_units(m, lam * sigma, E1, E2, z), base, atol=1e-9), (lam, m)
