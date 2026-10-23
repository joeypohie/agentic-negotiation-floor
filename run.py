"""Step 6, piece 4: play and log full episodes with an LLM client.

One JSON line per finished episode, keyed by (seed, condition, replicate).
Running the same command again skips finished keys, so a dropped connection
or a closed laptop loses nothing.

    python run.py --model qwen3.5:9b --conditions U --seeds 501-520 --reps 1 --log logs/plumbing.jsonl
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from agents import LLMClient, make_api, parse_belief, parse_decision
from env import DealerView, play, score
from generator import make_scenario, simulate_disclosure, value_dp
from rational import price_grid, prior_G, rational_bots
from render import (client_brief, belief_points, pre_quote_message, post_quote_message,
                    decision_message, noelicit_message)

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("U", "B", "U_noelicit", "U_probe")
PROVIDER = "SoCLaaS"
TURN_SEED = {"pre": 0, "post": 10, "decision": 20, "noelicit": 30}   # retries add 0, 1, 2, ...


def episode_seed(sd, rep):
    """Base API seed. Shared by U, B and U_noelicit for the same scenario and replicate
    (common random numbers), different across replicates."""
    return sd * 1000 + rep * 100

def probe_offset(seed):
    """k for the payoff probe: 0.25g to 2.0g, spread evenly over any 20 consecutive seeds."""
    return 0.25 + 1.75 * ((seed - 1) % 20) / 19


def probe_scenario(scn, g):
    """The payoff probe (pilot only, model_selection.md): the same scenario, dealer and quote,
    but the client's limit is set below the dealer's quote, so accepting loses money."""
    q_grid = price_grid(scn.m, g)
    quote, _, _ = rational_bots(q_grid, prior_G(q_grid, scn.m, g))
    p = quote(DealerView(m=scn.m, c=scn.c), g)
    return replace(scn, r=round(p - probe_offset(scn.seed) * g, value_dp))

def run_metadata(model):
    """What every log line records about the run itself (CLAUDE.md, 'Log with each episode')."""
    cfg_text = (ROOT / "config.yaml").read_text()

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    return {
        "model": model, "provider": PROVIDER, "temperature": yaml.safe_load(cfg_text)["model"]["temperature"],
        "reasoning_mode": "off", "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "numpy": np.__version__, "python": platform.python_version(),
    }


class _MalformedDecision(Exception):
    """The decision turn failed all its attempts, so the episode cannot be played out."""


def play_episode(llm, scn, band, g, condition, rep):
    """Play one episode. Returns everything the log line needs."""
    q_grid = price_grid(scn.m, g)
    dealer_quote, _, dealer_decide = rational_bots(q_grid, prior_G(q_grid, scn.m, g))
    pts = belief_points(scn, g)
    base = episode_seed(scn.seed, rep)
    msgs = [{"role": "system", "content": client_brief(scn, g, band if condition == "B" else None)}]
    turns, seen = {}, {}

    def ask(name, text, parse):
        msgs.append({"role": "user", "content": text})
        t = llm.ask(msgs, parse, base + TURN_SEED[name])
        msgs.append({"role": "assistant", "content": t.raw[-1]})
        turns[name] = t
        return t

    def quote(dv, g_):
        seen["quote"] = dealer_quote(dv, g_)
        return seen["quote"]

    def respond(cv, g_, p):
        read = lambda x: parse_decision(x, scn.m, g)
        if condition == "U_noelicit":
            t = ask("noelicit", noelicit_message(p), read)
        else:
            ask("pre", pre_quote_message(scn, g), lambda x: parse_belief(x, pts))   # the quote is not shown here
            ask("post", post_quote_message(scn, g, p), lambda x: parse_belief(x, pts))
            t = ask("decision", decision_message(p), read)
        if t.malformed:
            raise _MalformedDecision
        action, q, reason = t.parsed
        seen.update(action=action, counter=q, reason=reason)                     # as the client stated it
        return action, q

    try:
        ep = play(scn, g, quote, respond, dealer_decide)
        out = score(scn, ep)
    except _MalformedDecision:
        ep = out = None
    return msgs, turns, seen, ep, out


def make_record(meta, served, scn, band, condition, rep, msgs, turns, seen, ep, out, seconds):
    belief = lambda name: turns[name].parsed if name in turns else None
    return {
        "key": {"seed": scn.seed, "condition": condition, "replicate": rep},
        "run": {**meta, "served": served},
        "scenario": {"m": scn.m, "c": scn.c, "r": scn.r},
        "band": {"shown": condition == "B", "avg": band.avg, "low": band.low, "high": band.high,
                 "s": band.s, "d": band.d, "prints": list(band.prints)},          # d and prints: audit only
        "quote": seen.get("quote"),
        "beliefs": {"pre": belief("pre"), "post": belief("post")},
        "decision": {"action": seen.get("action"), "counter_stated": seen.get("counter"),
                     "reason": seen.get("reason")},
        "malformed_decision": out is None,
        "episode": asdict(ep) if ep else None,
        "outcome": asdict(out) if out else None,
        "turns": {n: {"malformed": t.malformed, "attempts": len(t.raw), "raw": t.raw, "errors": t.errors,
                      "seeds": t.seeds, "served": t.served, "seconds": round(t.seconds, 2)}
                  for n, t in turns.items()},
        "messages": msgs,
        "seconds": round(seconds, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_log(path):
    """Finished keys, and the served model name recorded so far (for the guard across resumes)."""
    done, served = set(), None
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                k = rec["key"]
                done.add((k["seed"], k["condition"], k["replicate"]))
                served = rec["run"]["served"] or served
    return done, served


def run(model, conditions, seeds, reps, log_path, api=None):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    g, D = cfg["calibration"]["g_points"], cfg["calibration"]["disclosure"]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done, served = read_log(log_path)
    llm = LLMClient(model, api if api is not None else make_api(ROOT / ".env"),
                    temperature=cfg["model"]["temperature"], served=served)
    meta = run_metadata(model)

    order = [(sd, cond, rep) for sd in seeds for rep in range(1, reps + 1) for cond in conditions]
    todo = [k for k in order if k not in done]
    print(f"{model}: {len(order)} episodes, {len(order) - len(todo)} already logged, {len(todo)} to run")
    for i, (sd, cond, rep) in enumerate(todo, 1):
        scn = make_scenario(sd, g)
        band = simulate_disclosure(scn, g, D["K"], D["omega"], D["tau"])
        r_original = scn.r
        if cond == "U_probe":
            scn = probe_scenario(scn, g)
        t0 = time.time()
        msgs, turns, seen, ep, out = play_episode(llm, scn, band, g, cond, rep)
        rec = make_record(meta, llm.served, scn, band, cond, rep, msgs, turns, seen, ep, out, time.time() - t0)
        rec["probe"] = ({"k": probe_offset(sd), "r_original": r_original} if cond == "U_probe" else None)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        bad = sum(t["malformed"] for t in rec["turns"].values())
        result = "MALFORMED DECISION" if out is None else out.status       # no outcome numbers while piloting

        print(f"[{i}/{len(todo)}] seed {sd} {cond:<10} rep {rep}  quote {seen.get('quote')}  {result}  "
              f"malformed turns {bad}  ({rec['seconds']:.0f}s)", flush=True)


def parse_seeds(text):
    """'501-520' or '1,2,5' or '501-505,510'."""
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Play and log LLM-client episodes (resumable).")
    ap.add_argument("--model", required=True, help="SoCLaaS model name, e.g. qwen3.5:9b")
    ap.add_argument("--conditions", default="U", help="comma list from U, B, U_noelicit")
    ap.add_argument("--seeds", required=True, help="e.g. 501-520 or 1,2,3")
    ap.add_argument("--reps", type=int, default=1, help="replicates per scenario and condition")
    ap.add_argument("--log", required=True, help="JSON-lines log file, e.g. logs/pilot_gemma4.jsonl")
    a = ap.parse_args(argv)
    conditions = [c.strip() for c in a.conditions.split(",")]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        ap.error(f"unknown condition(s): {sorted(unknown)}")
    run(a.model, conditions, parse_seeds(a.seeds), a.reps, a.log)


if __name__ == "__main__":
    main()
