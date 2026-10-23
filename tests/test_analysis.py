"""Step 8 analysis: the stated belief is turned into a CDF and a best counter correctly."""
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from analysis import monotone, stated_cdf, own_best_counter
from generator import make_scenario
from rational import price_grid, prior_G, value_of_countering
from render import BELIEF_GAPS

ROOT = Path(__file__).resolve().parent.parent
G = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]["g_points"]


def test_monotone_fixes_dips_and_keeps_rising_answers():
    assert np.all(np.diff(monotone([0.1, 0.05, 0.2, 0.3, 0.25, 0.5])) >= 0)
    assert np.allclose(monotone([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])


def test_stating_the_prior_reproduces_the_rational_client():
    prior_at_points = [stats.lognorm(s=0.5).sf(k) for k in BELIEF_GAPS]
    gaps = []
    for sd in range(1, 51):
        s = make_scenario(sd, G)
        q = price_grid(s.m, G)
        assert np.max(np.abs(stated_cdf(prior_at_points, s, G, q) - prior_G(q, s.m, G))) < 0.03
        _, q_exact = value_of_countering(s.r, q, prior_G(q, s.m, G))
        gaps.append((own_best_counter(prior_at_points, s, G) - q_exact) / G)
    assert abs(np.mean(gaps)) < 0.02 and np.max(np.abs(gaps)) <= 0.15
