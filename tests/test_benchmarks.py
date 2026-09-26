"""Step 4 gates — the client's Bayesian posterior over the dealer's cost."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats

from generator import make_scenario, simulate_disclosure
from rational import price_grid, prior_G
from benchmarks import band_posterior, client_belief

ROOT = Path(__file__).resolve().parent.parent
CAL = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]
G_ = CAL["g_points"]
D = CAL["disclosure"]
N = 1000
PRIOR_MEAN_GAP = stats.lognorm(s=0.5, scale=1.0).mean()      # E[E1] = 1.1331


@pytest.fixture(scope="module")
def posteriors():
    """For each scenario: the true cost, the posterior CDF at it, and both means."""
    rows = []
    for sd in range(1, N + 1):
        scn = make_scenario(sd, G_)
        band = simulate_disclosure(scn, G_, D["K"], D["omega"], D["tau"])
        q = price_grid(scn.m, G_)
        w, G = band_posterior(q, scn.m, G_, band.avg, band.s)
        rows.append({
            "pit": float(np.interp(scn.c, q, G)),              # posterior CDF at the truth
            "post_err": abs(float(w @ q) - scn.c),
            "prior_err": abs((scn.m - PRIOR_MEAN_GAP * G_) - scn.c),
        })
    return rows


# ---------- Gate (a): the posterior is calibrated ----------

def test_posterior_is_calibrated(posteriors):
    """If the update is correct, the posterior CDF at the true cost is uniform on [0, 1]."""
    pit = [r["pit"] for r in posteriors]
    assert stats.kstest(pit, "uniform").pvalue > 0.01


# ---------- Gate (b): the posterior beats the prior ----------

def test_posterior_mean_beats_prior_mean(posteriors):
    post = np.mean([r["post_err"] for r in posteriors])
    prior = np.mean([r["prior_err"] for r in posteriors])
    assert post < prior


# ---------- Sanity: the machinery itself ----------

def test_uninformative_band_recovers_prior():
    """A band with enormous uncertainty carries no information, so the posterior
    must reproduce the exact prior. This guards against a grid convention that
    would create a phantom value of disclosure."""
    m = 100.0
    q = price_grid(m, G_)
    _, G = band_posterior(q, m, G_, avg=m - G_, s=1e6)
    assert np.max(np.abs(G - prior_G(q, m, G_))) < 1e-3


def test_posterior_is_a_proper_distribution():
    scn = make_scenario(1, G_)
    band = simulate_disclosure(scn, G_, D["K"], D["omega"], D["tau"])
    q = price_grid(scn.m, G_)
    w, G = band_posterior(q, scn.m, G_, band.avg, band.s)
    eps = 1e-12                                              # floating-point slack from summing 481 terms
    assert np.all(w >= 0) and abs(w.sum() - 1) < eps
    assert np.all(np.diff(G) >= -eps) and G.min() >= -eps and G.max() <= 1 + eps
    assert w[q >= scn.m].sum() == 0                          # no mass at or above m


def test_client_belief_switches_on_condition():
    scn = make_scenario(1, G_)
    q = price_grid(scn.m, G_)
    band = simulate_disclosure(scn, G_, D["K"], D["omega"], D["tau"])
    assert np.array_equal(client_belief(q, scn, G_), prior_G(q, scn.m, G_))      # U
    assert not np.allclose(client_belief(q, scn, G_, band), prior_G(q, scn.m, G_))  # B
