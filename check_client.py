"""Live check with small model (one U episode's client turns on qwen3.5:9b)."""
import yaml

from agents import LLMClient, make_api, parse_belief, parse_decision
from benchmarks import client_belief
from env import DealerView
from generator import make_scenario
from rational import price_grid, rational_bots
from render import (client_brief, belief_points, pre_quote_message, post_quote_message,
                    decision_message)

g = yaml.safe_load(open("config.yaml"))["calibration"]["g_points"]
scn = make_scenario(901, g)                                  # outside main, pilot and held-out seeds
q_grid = price_grid(scn.m, g)
quote, _, _ = rational_bots(q_grid, client_belief(q_grid, scn, g))
p = quote(DealerView(m=scn.m, c=scn.c), g)                   # the dealer's rounded quote
pts = belief_points(scn, g)

llm = LLMClient("qwen3.5:9b", make_api())
msgs = [{"role": "system", "content": client_brief(scn, g)}]


def turn(user_text, parse, seed, label):
    msgs.append({"role": "user", "content": user_text})
    t = llm.ask(msgs, parse, seed)
    msgs.append({"role": "assistant", "content": t.raw[-1]})
    print(f"\n--- {label}  ({t.seconds:.1f}s, served={t.served}, attempts={len(t.raw)}, "
          f"malformed={t.malformed})")
    print("parsed:", t.parsed)
    if t.errors:
        print("rejected:", t.errors)
    return t


print(f"scenario 901: m={scn.m}  r={scn.r}  quote={p}  (dealer cost hidden: {scn.c})")
turn(pre_quote_message(scn, g), lambda x: parse_belief(x, pts), 901_000, "belief before quote")
turn(post_quote_message(scn, g, p), lambda x: parse_belief(x, pts), 901_010, "belief after quote")
turn(decision_message(p), lambda x: parse_decision(x, scn.m, g), 901_020, "decision")
