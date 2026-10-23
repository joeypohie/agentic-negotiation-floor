"""Model selection rule, step 2: the five viability checks for each pilot log (model_selection.md).

    python pilot_summary.py logs/pilot_*.jsonl

It reports only whether each model can play the game, plus the tie-breakers
(malformed-turn rate, median seconds per episode). It never reports outcome
measures such as capture or surplus, which the rule forbids using to choose.
"""
import json
import statistics
import sys

from scipy import stats

N_PROBE = 20                          # probe episodes expected per model (seeds 501-520)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def viability(recs):
    """The five checks. Each entry: (value shown, passed)."""
    turns = [t for r in recs for t in r["turns"].values()]
    malformed = sum(t["malformed"] for t in turns) / len(turns) if turns else 1.0

    counters = [(r, r["decision"]["counter_stated"]) for r in recs if r["decision"]["action"] == "counter"]
    sensible = [q < r["quote"] and q <= r["scenario"]["r"] for r, q in counters]
    share_sensible = sum(sensible) / len(sensible) if sensible else None

    probe = [r for r in recs if r["key"]["condition"] == "U_probe"]
    # trades done at the quote: an "accept", or a counter at or above the quote, which the rules treat as accepting
    probe_accepts = sum(r["episode"] is not None and r["episode"]["status"] == "quote_accepted" for r in probe)

    pairs = [(r["scenario"]["r"] - r["scenario"]["m"], q - r["scenario"]["m"])
             for r, q in counters if r["key"]["condition"] == "U_probe"]
    rho = stats.spearmanr(*zip(*pairs))[0] if len(pairs) >= 5 else None

    beliefs = [b for r in recs for b in (r["beliefs"]["pre"], r["beliefs"]["post"]) if b is not None]
    coherent = [all(b2 - b1 >= -0.01 - 1e-9 for b1, b2 in zip(b, b[1:])) for b in beliefs]
    share_coherent = sum(coherent) / len(coherent) if coherent else None

    return {
        "1 readable replies":      (f"{malformed:.1%} malformed of {len(turns)} turns", malformed < 0.02),
        "2 sensible counters":     ("no counters" if share_sensible is None else
                                    f"{share_sensible:.0%} of {len(sensible)}",
                                    share_sensible is None or share_sensible >= 0.90),
        "3 refuses losing trades": (f"accepted {probe_accepts} of {len(probe)} probes",
                                    len(probe) == N_PROBE and probe_accepts <= 2),
        "4 counters track limit":  (f"only {len(pairs)} probe counters, not assessed" if rho is None else
                                    f"rho {rho:+.2f} over {len(pairs)} probe counters",
                                    rho is None or rho > 0),                  # NaN (no variation) fails
        "5 coherent beliefs":      ("no readable beliefs" if share_coherent is None else
                                    f"{share_coherent:.0%} of {len(coherent)}",
                                    share_coherent is not None and share_coherent >= 0.90),
    }, malformed


def main(paths):
    rows = []
    for path in paths:
        recs = load(path)
        served = {r["run"]["served"] for r in recs}
        n_u = sum(r["key"]["condition"] == "U" for r in recs)
        n_p = sum(r["key"]["condition"] == "U_probe" for r in recs)
        checks, malformed = viability(recs)
        secs = statistics.median(r["seconds"] for r in recs)
        ok = all(passed for _, passed in checks.values())
        rows.append((ok, malformed, secs, path))
        print(f"\n=== {path}\nserved {', '.join(sorted(map(str, served)))} | U episodes {n_u} | "
              f"probe episodes {n_p} | median {secs:.1f}s per episode")
        if n_u != 20 or n_p != N_PROBE:
            print("  WARNING: expected 20 U and 20 probe episodes; is the run finished?")
        for name, (shown, passed) in checks.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name:<24} {shown}")
        print(f"  => {'PASSES' if ok else 'FAILS'} the viability gate")

    print("\nRule step 3 among models that pass: lowest malformed rate; within 1 point, "
          "fastest; then larger model.")
    for ok, malformed, secs, path in sorted(rows, key=lambda x: (not x[0], x[1], x[2])):
        print(f"  {'pass' if ok else 'FAIL'}  malformed {malformed:5.1%}  median {secs:5.1f}s  {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python pilot_summary.py logs/pilot_*.jsonl")
    main(sys.argv[1:])
