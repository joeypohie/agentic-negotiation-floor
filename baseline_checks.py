"""Step 8 gates for one model's U run (CLAUDE.md, Step 8).

    python baseline_checks.py logs/main_ornith1.5-35b.jsonl

The log must hold U and U_noelicit, each on seeds 1-50 with 2 replicates.
"""
import sys

import yaml

from analysis import load_episodes, gate_a, gate_b, gate_c


def main(paths):
    cal = yaml.safe_load(open("config.yaml"))["calibration"]
    g, D = cal["g_points"], cal["disclosure"]
    recs = load_episodes(*paths)
    served = sorted({r["run"]["served"] for r in recs})
    n_u = sum(r["key"]["condition"] == "U" for r in recs)
    n_c = sum(r["key"]["condition"] == "U_noelicit" for r in recs)
    print(f"served {', '.join(served)} | U episodes {n_u} | U_noelicit episodes {n_c}")
    if n_u != 100 or n_c != 100:
        print("  WARNING: expected 100 U and 100 U_noelicit episodes (seeds 1-50 x 2 each)")
    
    wrong = sum(1 for r in recs if r["key"]["condition"].endswith("_noelicit") and list(r["turns"]) != ["noelicit"])
    if wrong:
        sys.exit(f"  ERROR: {wrong} no-elicit episodes were asked belief questions; fix run.py and rerun them")



    a = gate_a(recs, g)
    verdict = "NOT ASSESSABLE" if a["passed"] is None else ("PASS" if a["passed"] else "FAIL")
    shown = f"only {a['n']} counters with a readable belief (need 10)" if a["passed"] is None else \
        f"rho {a['rho']:+.2f}, one-sided p {a['p']:.3g}, over {a['n']} counters"
    print(f"  {verdict:<14} (a) elicitation not decorative: {shown}")


    b = gate_b(recs)
    print(f"  {'PASS' if b['passed'] else 'FAIL':<14} (b) asking doesn't change play: U - U_noelicit "
        f"{b['diff']:+.3f}, 95% CI [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] over {b['n_scenarios']} scenarios "
        f"(U {b['main_mean']:.3f}, U_noelicit {b['control_mean']:.3f})")

    c = gate_c(recs, g, D)
    print(f"  {'PASS' if c['passed'] else 'RAISE REPS':<14} (c) power: expected B-U half-width {c['half_width']:.3f} "
          f"vs predicted effect {c['predicted_effect']:.3f} (replicate SD {c['s_within']:.3f}, "
          f"{c['n_scenarios']} scenarios; optimistic: ignores between-scenario variation)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python step8_gates.py logs/main_<model>.jsonl")
    main(sys.argv[1:])
