import numpy as np
from dataclasses import dataclass

price_dp = 3    # quotable prices — tick is 0.001 points
value_dp = 4    # private values  — 0.0001 points

@dataclass(frozen=True)
class Scenario:
    seed: int
    m: float
    c: float
    r: float

def streams(seed):
    """Independent streams to adding flow will not change values"""
    parent = np.random.default_rng(seed)
    children = parent.spawn(3)
    return children

def make_scenario(seed, g):
    values, _, _ = streams(seed)
    m = round(98.0 + 4.0 * values.random(), price_dp)
    E1, E2 = values.lognormal(0.0, 0.5, size=2)
    return Scenario(
        seed=seed, 
        m=m, 
        c= round(m - g * E1,value_dp), 
        r= round(m + g * E2, value_dp)
    )

@dataclass(frozen=True)
class Band:
    avg: float        # implied typical dealer cost   — shown to the client (B only)
    low: float        # avg - s                        — shown
    high: float       # avg + s                        — shown
    s: float          # total sd: averaging noise + dealer deviation (benchmark)
    d: float          # this dealer's deviation from typical          (audit only)
    prints: tuple     # the K reported spreads                         (audit only)


def simulate_disclosure(scn, g, K, omega, tau):
    """Disclosed dealer-purchase summary for one scenario. Uses the third RNG stream."""
    _, _, rng = streams(scn.seed)
    d = rng.normal(0.0, tau * g)                                   # this dealer vs typical
    L = scn.c - d                                                  # market-wide typical cost
    prints = (L - scn.m) + rng.normal(0.0, omega * g, size=K)      # other dealers' spreads
    avg = scn.m + prints.mean()
    s = float(np.sqrt((omega * g) ** 2 / K + (tau * g) ** 2))
    return Band(avg=round(avg, value_dp),
                low=round(avg - s, value_dp),
                high=round(avg + s, value_dp),
                s=s, d=float(d),
                prints=tuple(float(x) for x in prints))
