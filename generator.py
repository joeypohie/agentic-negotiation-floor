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