"""Step 8 onwards: turning logged episodes into the quantities the gates and the analysis use.

Reads episode logs only (run.py's JSON lines). Nothing here calls the API.
"""
import json

import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression

from benchmarks import client_belief
from generator import Scenario, make_scenario, simulate_disclosure
from rational import price_grid, value_of_countering, predictions
from render import belief_points


# ---------- reading logs ----------

def load_episodes(*paths):
    """Every logged episode from one or more JSON-lines logs."""
    recs = []
    for path in paths:
        with open(path) as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def scenario_of(rec):
    """The Scenario as the client saw it (for a probe, its limit is the probe limit)."""
    s = rec["scenario"]
    return Scenario(seed=rec["key"]["seed"], m=s["m"], c=s["c"], r=s["r"])


# ---------- the client's stated belief as a CDF ----------

def monotone(probs):
    """Isotonic (least-squares, non-decreasing) fit to the 9 stated probabilities (CLAUDE.md: scoring)."""
    return isotonic_regression(np.asarray(probs, dtype=float)).x


def stated_cdf(probs, scn, g, q_grid):
    """The client's belief about the dealer's cost as a CDF on the price grid.

    The 9 stated P(cost < x) are made non-decreasing, then joined by straight lines,
    anchored at 0 at m - 6g (the bottom of the grid) and at 1 at m (the client is told
    costs are always below the evaluated price).
    """
    xs = [scn.m - 6 * g] + belief_points(scn, g) + [scn.m]
    ys = np.concatenate([[0.0], monotone(probs), [1.0]])
    return np.interp(q_grid, xs, ys)


def own_best_counter(probs, scn, g):
    """q*(r, own G): the counter that maximises expected profit under the client's own stated belief."""
    q_grid = price_grid(scn.m, g)
    _, q_star = value_of_countering(scn.r, q_grid, stated_cdf(probs, scn, g, q_grid))
    return q_star


# ---------- Step 8 gate (a): elicitation is not decorative ----------

def gate_a(recs, g, min_n=10):
    """Do actual counters rise with q*(r, own post-quote belief)? Rank correlation, in g from m."""
    pairs = []
    for r in recs:
        if (r["key"]["condition"] == "U" and r["decision"]["action"] == "counter"
                and r["beliefs"]["post"] is not None):
            scn = scenario_of(r)
            q_own = own_best_counter(r["beliefs"]["post"], scn, g)
            pairs.append(((r["decision"]["counter_stated"] - scn.m) / g, (q_own - scn.m) / g))
    if len(pairs) < min_n:
        return {"n": len(pairs), "rho": None, "p": None, "passed": None}
    actual, predicted = zip(*pairs)
    res = stats.spearmanr(predicted, actual, alternative="greater")
    return {"n": len(pairs), "rho": float(res[0]), "p": float(res[1]),
            "passed": bool(res[0] > 0 and res[1] < 0.05)}


# ---------- Step 8 gate (b): elicitation does not change play ----------

def _scenario_means(recs, condition, seeds):
    """Mean client_capture per scenario (replicates averaged), malformed decisions excluded."""
    out = {}
    for sd in seeds:
        caps = [r["outcome"]["client_capture"] for r in recs
                if r["key"]["condition"] == condition and r["key"]["seed"] == sd and r["outcome"]]
        if caps:
            out[sd] = float(np.mean(caps))
    return out


def gate_b(recs, seeds=range(1, 11), n_boot=10_000, seed=0):
    """Is the control episodes' mean capture inside the 95% bootstrap CI of the main U episodes?"""
    main = _scenario_means(recs, "U", seeds)
    ctrl = _scenario_means(recs, "U_noelicit", seeds)
    common = sorted(set(main) & set(ctrl))
    m = np.array([main[s] for s in common])
    rng = np.random.default_rng(seed)
    boot = m[rng.integers(0, len(m), size=(n_boot, len(m)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ctrl_mean = float(np.mean([ctrl[s] for s in common]))
    return {"n_scenarios": len(common), "main_mean": float(m.mean()), "ci": (float(lo), float(hi)),
            "control_mean": ctrl_mean, "passed": bool(lo <= ctrl_mean <= hi)}


# ---------- Step 8 gate (c): power check ----------

def predicted_vod_share(seeds, g, disclosure):
    """Benchmark mean of capture(B*) - capture(U*) over the given scenarios (dealer quote rounded)."""
    diffs = []
    for sd in seeds:
        scn = make_scenario(sd, g)
        band = simulate_disclosure(scn, g, disclosure["K"], disclosure["omega"], disclosure["tau"])
        U, B, _ = predictions(scn, g, client_belief(price_grid(scn.m, g), scn, g, band))
        diffs.append(B.client_capture - U.client_capture)
    return float(np.mean(diffs))


def gate_c(recs, g, disclosure, seeds=range(1, 51), reps=2):
    """CLAUDE.md: paired SD of B - U ~ sqrt(2) x U's within-scenario replicate SD.

    With replicates averaged within each scenario, the CI half-width for the mean B - U is
    1.96 * sqrt(2) * s_within / sqrt(reps) / sqrt(n_scenarios). This counts replicate noise
    only, not differences in the band's effect between scenarios, so it is optimistic.
    """
    per_seed = {}
    for r in recs:
        if r["key"]["condition"] == "U" and r["key"]["seed"] in seeds and r["outcome"]:
            per_seed.setdefault(r["key"]["seed"], []).append(r["outcome"]["client_capture"])
    variances = [np.var(v, ddof=1) for v in per_seed.values() if len(v) >= 2]
    s_within = float(np.sqrt(np.mean(variances))) if variances else float("nan")
    n = len(per_seed)
    half_width = 1.96 * np.sqrt(2) * s_within / np.sqrt(reps) / np.sqrt(n)
    target = predicted_vod_share(seeds, g, disclosure)
    return {"n_scenarios": n, "s_within": s_within, "half_width": float(half_width),
            "predicted_effect": target, "passed": bool(half_width < target)}
