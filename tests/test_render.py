"""
Checks that the client briefing is correct before LLM reads it 

Importantly, confirms that neither briefing contains: 
- dealer's true cost
- how the band is generated; the dealer's deviation from the typical dealer
- the 10 individual reported trades. Client only knows average and range
- the word "condition", so that LLM cannot tell it is part of the experiment 
"""

import re
from pathlib import Path

import pytest
import yaml

from generator import make_scenario, simulate_disclosure
from render import client_brief, band_block, fp, fv, P10, P50, P90

ROOT = Path(__file__).resolve().parent.parent
CAL = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]
G = CAL["g_points"]
D = CAL["disclosure"]
SEEDS = range(1, 51)                                   # the main scenarios


def _states(text, number):
    """True if `number` appears in `text` as a whole number, not inside a longer one."""
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", text) is not None


def _pair(sd):
    scn = make_scenario(sd, G)
    band = simulate_disclosure(scn, G, D["K"], D["omega"], D["tau"])
    return scn, band, client_brief(scn, G), client_brief(scn, G, band)


@pytest.mark.parametrize("sd", SEEDS)
def test_b_is_u_plus_exactly_the_band(sd):
    scn, band, u, b = _pair(sd)
    block = band_block(scn, band) + "\n\n"
    assert b.count(block) == 1
    assert b.replace(block, "", 1) == u


@pytest.mark.parametrize("sd", SEEDS)
def test_parity_every_benchmark_number_is_stated(sd):
    scn, band, u, b = _pair(sd)
    for text in (u, b):
        assert fp(scn.m) in text                        # evaluated price, 3 dp
        assert fv(scn.r) in text                        # own limit, 4 dp
        for q in (P10, P50, P90):                       # the prior, in points
            assert fv(q * G) in text
        assert "accepts any counteroffer at or above its own cost" in text
        assert "maximise your expected profit" in text
    for x in (band.avg, band.low, band.high):           # the band's numbers, in B
        assert fv(x) in b
    assert "Reported purchases" not in u                # and no band paragraph in U


@pytest.mark.parametrize("sd", SEEDS)
def test_no_leaks(sd):
    scn, band, u, b = _pair(sd)
    for text in (u, b):
        assert not _states(text, fv(scn.c))             # the dealer's true cost
        assert not _states(text, fv(band.d))            # this dealer's deviation
        for p in band.prints:                           # individual reported trades
            assert not _states(text, fv(p))
        assert "condition" not in text.lower()          # no hint of the experiment


def test_deterministic():
    assert _pair(7)[2:] == _pair(7)[2:]