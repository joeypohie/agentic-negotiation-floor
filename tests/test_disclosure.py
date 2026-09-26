"""Step 3 gate — the disclosed band is unbiased, the right width, and honest about coverage."""
from pathlib import Path
import numpy as np, yaml
from generator import make_scenario, simulate_disclosure

ROOT = Path(__file__).resolve().parent.parent
CAL  = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]
G    = CAL["g_points"]
D    = CAL["disclosure"]
N    = 1000


def _bands():
    for sd in range(1, N + 1):
        scn = make_scenario(sd, G)
        yield scn, simulate_disclosure(scn, G, D["K"], D["omega"], D["tau"])


def test_band_is_unbiased_and_right_width():
    pairs = list(_bands())
    s = pairs[0][1].s
    err = np.array([(b.avg - scn.c) for scn, b in pairs])

    assert abs(err.mean()) < 3 * s / np.sqrt(N)          # mean within 3 standard errors of 0
    assert abs(err.std(ddof=1) / s - 1) < 0.10           # sd within 10% of the stated s


def test_stated_range_covers_two_times_in_three():
    cover = np.mean([b.low <= scn.c <= b.high for scn, b in _bands()])
    assert 0.64 <= cover <= 0.73                          # 68.3% ± 3 standard errors


def test_band_reproducible():
    scn = make_scenario(7, G)
    args = (G, D["K"], D["omega"], D["tau"])
    assert simulate_disclosure(scn, *args) == simulate_disclosure(scn, *args)
