# tests/test_generator.py
import numpy as np, yaml
from scipy import stats
from generator import make_scenario

G = yaml.safe_load(open("config.yaml"))["calibration"]["g_points"]
SCEN = [make_scenario(s, G) for s in range(1, 1001)]


def test_ordering():
    bad = [s for s in SCEN if not (s.c < s.m < s.r)]
    assert not bad, f"{len(bad)} scenarios violate c < m < r, e.g. {bad[:3]}"


def test_gap_distributions():
    ref = stats.lognorm(s=0.5, scale=1.0)
    for name, gaps in [("cost",  np.array([(s.m - s.c)/G for s in SCEN])),
                       ("limit", np.array([(s.r - s.m)/G for s in SCEN]))]:
        p = stats.kstest(gaps, ref.cdf).pvalue
        assert p > 0.01, f"{name} gap fails KS against LogNormal(0, 0.5): p={p:.4f}"


def test_reproducible():
    assert make_scenario(7, G) == make_scenario(7, G)
