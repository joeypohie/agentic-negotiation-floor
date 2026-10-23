"""Step 10: Determine effect of disclosure, and how the client uses it.

    python disclosure_results.py logs/main_ornith1.5-35b.jsonl

Reads episode logs only; nothing here calls the API. Definitions are in
analysis_reference.md and prereg.md. The shared basics (reading logs, turning
stated beliefs into a CDF) come from analysis.py, which is left exactly as it
was when the Step 8 checks ran.
"""
import sys
from collections import Counter

import numpy as np
import yaml
from scipy import stats

from analysis import load_episodes, scenario_of, monotone, stated_cdf
from benchmarks import band_posterior, client_belief
from generator import make_scenario, simulate_disclosure
from rational import price_grid, prior_G, value_of_countering, predictions
from render import belief_points

SEEDS = range(1, 51)
ARMS = ("U", "B", "U_noelicit", "B_noelicit")


# ---------- comparing two arms, paired by scenario ----------

def bootstrap_ci(values, n_boot=10_000, seed=0):
    """Mean and 95% bootstrap CI of per-scenario values (resampling scenarios)."""
    v = np.asarray(values, dtype=float)
    boot = v[np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(v.mean()), (float(lo), float(hi))


def per_scenario(recs, condition, value, seeds=SEEDS):
    """{seed: mean of value(rec) over that scenario's replicates}. value(rec) may return None to skip a rec."""
    out = {}
    for sd in seeds:
        vals = [value(r) for r in recs if r["key"]["condition"] == condition and r["key"]["seed"] == sd]
        vals = [v for v in vals if v is not None]
        if vals:
            out[sd] = float(np.mean(vals))
    return out


def paired(recs, hi, lo, value, seeds=SEEDS):
    """Mean of value(hi arm) - value(lo arm), paired by scenario, with a 95% bootstrap CI."""
    a, b = per_scenario(recs, hi, value, seeds), per_scenario(recs, lo, value, seeds)
    common = sorted(set(a) & set(b))
    diff, ci = bootstrap_ci([a[s] - b[s] for s in common])
    return {"n": len(common), "hi_mean": float(np.mean([a[s] for s in common])),
            "lo_mean": float(np.mean([b[s] for s in common])), "diff": diff, "ci": ci}


def interaction(recs, value, seeds=SEEDS):
    """(B - U) - (B_noelicit - U_noelicit), paired by scenario, with a 95% bootstrap CI."""
    arms = {c: per_scenario(recs, c, value, seeds) for c in ARMS}
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    d = [(arms["B"][s] - arms["U"][s]) - (arms["B_noelicit"][s] - arms["U_noelicit"][s]) for s in common]
    diff, ci = bootstrap_ci(d)
    return {"n": len(common), "diff": diff, "ci": ci}


# ---------- per-episode values ----------

def capture(r):
    return r["outcome"]["client_capture"] if r["outcome"] else None


def surplus(r):
    return r["outcome"]["client_surplus"] if r["outcome"] else None


def counter_g(r, g):
    """The stated counter, in g from m; None if the client accepted or the decision was malformed."""
    if r["decision"]["action"] != "counter":
        return None
    return (r["decision"]["counter_stated"] - r["scenario"]["m"]) / g


def dist_to_band(r, g):
    """A1: |stated counter - band avg|, in g. The avg is in every log line, shown only in B arms."""
    if r["decision"]["action"] != "counter":
        return None
    return abs(r["decision"]["counter_stated"] - r["band"]["avg"]) / g


# ---------- the rational benchmark, per scenario ----------

def benchmark(seeds, g, disclosure):
    """Per scenario: rational U* and B* outcomes (rounded quote), VoD in points, the rational
    counters under each arm's correct belief, and the benchmark CDFs at the 9 belief prices."""
    out = {}
    for sd in seeds:
        scn = make_scenario(sd, g)
        band = simulate_disclosure(scn, g, disclosure["K"], disclosure["omega"], disclosure["tau"])
        q = price_grid(scn.m, g)
        G0 = prior_G(q, scn.m, g)
        _, Gb = band_posterior(q, scn.m, g, band.avg, band.s)
        U, B, vod = predictions(scn, g, client_belief(q, scn, g, band))
        pts = belief_points(scn, g)
        out[sd] = {"U": U, "B": B, "vod": vod,
                   "q_star_U": value_of_countering(scn.r, q, G0)[1],
                   "q_star_B": value_of_countering(scn.r, q, Gb)[1],
                   "cdf_U": np.interp(pts, q, G0), "cdf_B": np.interp(pts, q, Gb)}
    return out


def efficiency(recs, hi, lo, bench, seeds=SEEDS):
    """E_client: sum over scenarios of the LLM's surplus gain / sum of VoD (ratio of sums)."""
    a, b = per_scenario(recs, hi, surplus, seeds), per_scenario(recs, lo, surplus, seeds)
    common = sorted(set(a) & set(b))
    return float(sum(a[s] - b[s] for s in common) / sum(bench[s]["vod"] for s in common))


# ---------- rates and the no-deal test ----------

def arm_summary(recs, condition, g):
    rs = [r for r in recs if r["key"]["condition"] == condition]
    ok = [r for r in rs if r["outcome"]]
    status = {k: sum(r["episode"]["status"] == k for r in ok)
              for k in ("quote_accepted", "counter_accepted", "counter_rejected")}
    viol = {}
    for r in ok:
        for v in r["outcome"]["violations"]:
            viol[v] = viol.get(v, 0) + 1
    qs = [counter_g(r, g) for r in rs if counter_g(r, g) is not None]
    return {"n": len(rs), "malformed": len(rs) - len(ok), "capture": float(np.mean([capture(r) for r in ok])),
            "no_deal": float(np.mean([r["outcome"]["no_deal"] for r in ok])), **status,
            "counters": len(qs), "mean_counter_g": float(np.mean(qs)) if qs else None, "violations": viol}


def mcnemar_no_deal(recs, hi, lo):
    """Exact McNemar test on no-deal, episodes paired by (seed, replicate)."""
    nd = {(r["key"]["seed"], r["key"]["replicate"], r["key"]["condition"]): r["outcome"]["no_deal"]
          for r in recs if r["outcome"] and r["key"]["condition"] in (hi, lo)}
    pairs = [(nd[(s, rep, lo)], nd[(s, rep, hi)]) for (s, rep, c) in nd if c == lo and (s, rep, hi) in nd]
    only_hi = sum(1 for x, y in pairs if y and not x)         # no deal only in the band arm
    only_lo = sum(1 for x, y in pairs if x and not y)
    p = stats.binomtest(only_hi, only_hi + only_lo).pvalue if only_hi + only_lo else 1.0
    return {"pairs": len(pairs), "no_deal_only_with_band": only_hi, "no_deal_only_without": only_lo, "p": float(p)}


# ---------- anchoring (A1, A2) ----------

def anchoring(recs, condition, g, near=0.1):
    """A2: share of counters within `near` g of the band avg; rank correlation of counter with limit."""
    rs = [r for r in recs if r["key"]["condition"] == condition and counter_g(r, g) is not None]
    d = [dist_to_band(r, g) for r in rs]
    q = [counter_g(r, g) for r in rs]
    lim = [(r["scenario"]["r"] - r["scenario"]["m"]) / g for r in rs]
    rho = stats.spearmanr(lim, q)[0] if len(rs) >= 5 else None
    levels = Counter(round(x / 0.05) * 0.05 for x in q).most_common(5)
    return {"counters": len(rs), "near_avg": float(np.mean([x <= near for x in d])) if d else None,
            "rho_limit": None if rho is None else float(rho), "levels": [(round(k, 2), n) for k, n in levels]}


# ---------- beliefs (elicited arms; descriptive) ----------

def belief_error(r, bench, which="pre"):
    """Mean squared difference between the stated (made non-decreasing) and benchmark probabilities."""
    probs = r["beliefs"][which]
    if probs is None:
        return None
    ref = bench[r["key"]["seed"]]["cdf_B" if r["band"]["shown"] else "cdf_U"]
    return float(np.mean((monotone(probs) - ref) ** 2))


def belief_median_to_band(r, g, which="pre"):
    """Uptake: |median of the stated belief - band avg|, in g."""
    probs = r["beliefs"][which]
    if probs is None:
        return None
    scn = scenario_of(r)
    q = price_grid(scn.m, g)
    cdf = stated_cdf(probs, scn, g, q)
    median = float(np.interp(0.5, cdf + np.arange(len(cdf)) * 1e-12, q))   # tiny ramp keeps cdf increasing
    return abs(median - r["band"]["avg"]) / g


def planning_gap(r, bench, g):
    """|stated counter - rational counter under the correct belief for this arm|, in g."""
    if r["decision"]["action"] != "counter":
        return None
    b = bench[r["key"]["seed"]]
    return abs(r["decision"]["counter_stated"] - (b["q_star_B"] if r["band"]["shown"] else b["q_star_U"])) / g


def quote_shift(r):
    """Mean change in stated probability from before to after the quote, over the 9 prices."""
    pre, post = r["beliefs"]["pre"], r["beliefs"]["post"]
    if pre is None or post is None:
        return None
    return float(np.mean(monotone(post) - monotone(pre)))


def four_cell(err, gap):
    """Reading from the CIs of the B - U change in belief error and in planning gap."""
    belief_better = err["ci"][1] < 0
    action_better = gap["ci"][1] < 0
    return {(True, True): "substitution", (False, True): "anchoring",
            (True, False): "planning failure", (False, False): "inert"}[(belief_better, action_better)]


def quotes_identical(recs):
    """Manipulation check: the dealer's quote is the same in every arm for each scenario."""
    by_seed = {}
    for r in recs:
        by_seed.setdefault(r["key"]["seed"], set()).add(r["quote"])
    return all(len(v) == 1 for v in by_seed.values())


# ---------- the report ----------

def fmt(res, unit=""):
    lo, hi = res["ci"]
    return f"{res['diff']:+.3f}{unit}  95% CI [{lo:+.3f}, {hi:+.3f}]  over {res['n']} scenarios"


def main(paths):
    cal = yaml.safe_load(open("config.yaml"))["calibration"]
    g, D = cal["g_points"], cal["disclosure"]
    recs = [r for r in load_episodes(*paths) if r["key"]["condition"] in ARMS]
    bad = [r for r in recs if r["key"]["condition"].endswith("_noelicit") and list(r["turns"]) != ["noelicit"]]
    if bad:
        sys.exit(f"ERROR: {len(bad)} no-elicit episodes were asked belief questions")
    seeds = sorted({r["key"]["seed"] for r in recs})
    bench = benchmark(seeds, g, D)

    print(f"served {', '.join(sorted({r['run']['served'] for r in recs}))} | seeds {seeds[0]}-{seeds[-1]} | "
          f"manipulation check (same quote in every arm): {'OK' if quotes_identical(recs) else 'FAILED'}")

    print("\n== Arms")
    for c in ARMS:
        a = arm_summary(recs, c, g)
        mc = "n/a" if a["mean_counter_g"] is None else f"{a['mean_counter_g']:+.2f}g"
        print(f"  {c:<11} n {a['n']:>3} (malformed {a['malformed']})  capture {a['capture']:.3f}  "
              f"no deal {a['no_deal']:.0%}  quote accepted {a['quote_accepted']}  counters {a['counters']} "
              f"(accepted {a['counter_accepted']}, rejected {a['counter_rejected']})  mean counter {mc}  "
              f"violations {a['violations'] or 'none'}")

    print("\n== Primary (H1): B_noelicit - U_noelicit, client_capture")
    p = paired(recs, "B_noelicit", "U_noelicit", capture, seeds)
    print(f"  {fmt(p)}   (B_noelicit {p['hi_mean']:.3f}, U_noelicit {p['lo_mean']:.3f})")
    print(f"  H1 {'SUPPORTED' if p['ci'][0] > 0 else 'NOT SUPPORTED'} (decision rule: CI lower bound above 0)")
    rat = sum(b["B"].client_capture - b["U"].client_capture for b in bench.values()) / len(bench)
    print(f"  rational benchmark B* - U*: {rat:+.3f}")

    print("\n== Secondary")
    s = paired(recs, "B", "U", capture, seeds)
    print(f"  B - U (elicited), capture:          {fmt(s)}   (B {s['hi_mean']:.3f}, U {s['lo_mean']:.3f})")
    print(f"  interaction (B-U) - (Bn-Un):        {fmt(interaction(recs, capture, seeds))}")
    print(f"  U - U_noelicit (asking, no band):   {fmt(paired(recs, 'U', 'U_noelicit', capture, seeds))}")
    print(f"  E_client: no-elicit {efficiency(recs, 'B_noelicit', 'U_noelicit', bench, seeds):+.2f}, "
          f"elicited {efficiency(recs, 'B', 'U', bench, seeds):+.2f}   (share of the rational gain captured)")
    for hi, lo in (("B_noelicit", "U_noelicit"), ("B", "U")):
        m = mcnemar_no_deal(recs, hi, lo)
        print(f"  no deal {hi} vs {lo}: only with band {m['no_deal_only_with_band']}, only without "
              f"{m['no_deal_only_without']} of {m['pairs']} pairs, McNemar p {m['p']:.3f}")

    print("\n== Anchoring")
    for hi, lo in (("B_noelicit", "U_noelicit"), ("B", "U")):
        a1 = paired(recs, hi, lo, lambda r: dist_to_band(r, g), seeds)
        print(f"  A1 |counter - band avg|, {hi} - {lo}: {fmt(a1, 'g')}   "
              f"({hi} {a1['hi_mean']:.2f}g, {lo} {a1['lo_mean']:.2f}g)")
    for c in ARMS:
        a2 = anchoring(recs, c, g)
        near = "n/a" if a2["near_avg"] is None else f"{a2['near_avg']:.0%}"
        rho = "n/a" if a2["rho_limit"] is None else f"{a2['rho_limit']:+.2f}"
        print(f"  A2 {c:<11} within 0.1g of avg {near:>4}  rho(counter, limit) {rho:>5}  "
              f"common levels (g from m) {a2['levels']}")

    print("\n== Beliefs (elicited arms; descriptive)")
    err = paired(recs, "B", "U", lambda r: belief_error(r, bench, "pre"), seeds)
    print(f"  belief error (pre-quote), B - U:    {fmt(err)}   (B {err['hi_mean']:.4f}, U {err['lo_mean']:.4f})")
    up = paired(recs, "B", "U", lambda r: belief_median_to_band(r, g, "pre"), seeds)
    print(f"  uptake |belief median - avg|, B - U: {fmt(up, 'g')}   "
          f"-> {'band registered' if up['ci'][1] < 0 else 'no clear uptake'}")
    gap = paired(recs, "B", "U", lambda r: planning_gap(r, bench, g), seeds)
    print(f"  planning gap |counter - rational|, B - U: {fmt(gap, 'g')}")
    print(f"  four-cell reading: {four_cell(err, gap)}")
    for c in ("U", "B"):
        qs = [x for x in (quote_shift(r) for r in recs if r["key"]["condition"] == c) if x is not None]
        print(f"  quote inference {c}: mean change in stated P(cost below x) after the quote {sum(qs) / len(qs):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python disclosure_results.py logs/main_ornith1.5-35b.jsonl")
    main(sys.argv[1:])
