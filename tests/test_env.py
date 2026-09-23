"""
This file contains the test for the environment and scripted bots.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator import make_scenario, price_dp
from env import play, score
from bots import (scripted_dealer_quote, scripted_client_respond,
                  scripted_dealer_decide, random_bots)

ROOT = Path(__file__).resolve().parent.parent
G = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]["g_points"]

VIOLS = ("dealer_quote_below_cost", "client_counter_above_limit",
         "client_accepted_above_limit", "dealer_rejected_profitable_counter")


# ---------- Gate A: calculation of g outcomes match an independent hand calculation ----------

def predict(scn, g):
    """Independent prediction of the scripted-bot outcome.

    Deliberately does NOT use play() or score(). If these two ever share
    code, the gate stops testing anything.
    """
    p = round(scn.m + g, price_dp)
    price = p if p <= scn.r - 0.5 * g else round(scn.m, price_dp)
    return price, (scn.r - price) / (scn.r - scn.c)


def test_gate_a_matches_hand_calculation():
    for s in range(1, 21):
        scn = make_scenario(s, G)
        out = score(scn, play(scn, G, scripted_dealer_quote,
                              scripted_client_respond, scripted_dealer_decide))
        exp_price, exp_cap = predict(scn, G)
        assert out.price == exp_price, f"seed {s}: {out.price} vs {exp_price}"
        assert abs(out.client_capture - exp_cap) < 1e-12, f"seed {s} capture"
        assert not out.violations, f"seed {s} unexpected violation {out.violations}"


# ---------- Gate B: 1,000 fuzzed episodes to check if irrational offers will break the scenario ----------

def fuzz(seed=12345, n=1000):
    rng = np.random.default_rng(seed)
    quote, respond, decide = random_bots(rng)

    proposals = []
    def recording_respond(cv, g, p):
        action, q = respond(cv, g, p)
        proposals.append((action, q, p))
        return action, q

    scns, eps, outs = [], [], []
    for s in range(1, n + 1):
        scn = make_scenario(s, G)
        ep = play(scn, G, quote, recording_respond, decide)
        scns.append(scn); eps.append(ep); outs.append(score(scn, ep))
    return scns, eps, outs, proposals


@pytest.fixture(scope="module")
def fuzzed():
    return fuzz()


def test_fuzz_outcomes_consistent(fuzzed):
    scns, _, outs, _ = fuzzed
    for scn, out in zip(scns, outs):
        assert (out.price is None) == (out.status == "counter_rejected"), out.seed
        assert out.no_deal == (1 if out.price is None else 0), out.seed
        if out.price is not None:
            S = scn.r - scn.c
            assert abs(out.dealer_surplus + out.client_surplus - S) < 1e-9, out.seed


def test_fuzz_rewrite_handled(fuzzed):
    _, eps, _, proposals = fuzzed
    rewrites = 0
    for (action, q, p), ep in zip(proposals, eps):
        if action == "counter" and q >= p:
            rewrites += 1
            assert ep.status == "quote_accepted", ep.seed
            assert ep.counter is None, ep.seed
            assert ep.price == p, ep.seed
    assert rewrites > 0, "rewrite branch never exercised"


def test_fuzz_coverage(fuzzed):
    _, _, outs, _ = fuzzed
    status = Counter(o.status for o in outs)
    viol = Counter(v for o in outs for v in o.violations)
    assert len(status) == 3, f"only reached {sorted(status)}"
    for name in VIOLS:
        assert viol[name] > 0, f"{name} never triggered"


def test_fuzz_replay():
    assert fuzz(12345)[1] == fuzz(12345)[1]
