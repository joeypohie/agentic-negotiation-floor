"""Check the questions posed to agents, and reading the client's replies."""
import json
from pathlib import Path

import pytest
import yaml

from generator import make_scenario
from render import (belief_points, belief_question, pre_quote_message, post_quote_message,
                    decision_message, noelicit_message, fv, BELIEF_GAPS)
from agents import ParseError, extract_json, parse_belief, parse_decision

ROOT = Path(__file__).resolve().parent.parent
G = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]["g_points"]
SCN = make_scenario(1, G)                    # m = 100.796
PTS = belief_points(SCN, G)
GOOD_BELIEF = {fv(x): p for x, p in zip(PTS, [0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.9])}


# ---------- the questions ----------

def test_belief_points_are_nine_distinct_prices_below_m():
    assert len(PTS) == len(BELIEF_GAPS) == 9
    assert PTS == sorted(PTS) and len({fv(x) for x in PTS}) == 9
    assert all(x < SCN.m for x in PTS)


def test_belief_question_lists_every_point():
    q = belief_question(SCN, G)
    assert all(f'"{fv(x)}"' in q for x in PTS)


def test_messages_show_the_quote_to_3_decimals():
    p = 100.782
    assert "100.782" in post_quote_message(SCN, G, p)
    assert "100.782" in decision_message(p)
    assert "quotes" not in pre_quote_message(SCN, G).replace("not quoted", "")


def test_reason_comes_after_the_action_and_only_when_asked():
    d = decision_message(100.782)
    assert d.index('"action"') < d.index('"reason"')
    assert d.index('"price"') < d.rindex('"reason"')
    assert '"reason"' not in noelicit_message(100.782)


# ---------- finding the JSON in a reply ----------

@pytest.mark.parametrize("reply", [
    '{"action": "accept"}',
    '```json\n{"action": "accept"}\n```',                     # code fences (gemma4)
    'Sure. Here is my answer:\n{"action": "accept"}\nThanks.',  # prose around it
    '<think>hmm {not json}</think>{"action": "accept"}',       # a stray thinking block
    '{"action": "accept", "reason": "the {quote} is fair"}',   # braces inside a string
])
def test_extract_json_finds_the_object(reply):
    assert extract_json(reply)["action"] == "accept"


@pytest.mark.parametrize("reply", ["", None, "I accept.", '{"action": "accept"', "[1, 2, 3]"])
def test_extract_json_rejects_unusable_replies(reply):
    with pytest.raises(ParseError):
        extract_json(reply)


# ---------- beliefs ----------

def test_parse_belief_reads_nine_probabilities_in_order():
    assert parse_belief(json.dumps(GOOD_BELIEF), PTS) == [0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.9]


def test_parse_belief_tolerates_key_format_and_numeric_strings():
    reply = {k + "0": str(v) for k, v in GOOD_BELIEF.items()}         # "100.69440": "0.02"
    assert parse_belief(json.dumps(reply), PTS)[0] == 0.02


def test_parse_belief_keeps_non_monotone_answers():
    vals = [0.1, 0.05, 0.2, 0.3, 0.25, 0.5, 0.6, 0.7, 0.9]             # fixed later, at scoring
    reply = {fv(x): v for x, v in zip(PTS, vals)}
    assert parse_belief(json.dumps(reply), PTS) == vals


@pytest.mark.parametrize("change", [
    lambda b: b.pop(next(iter(b))),                                   # a price missing
    lambda b: b.update({next(iter(b)): 1.5}),                         # above 1
    lambda b: b.update({next(iter(b)): -0.1}),                        # below 0
    lambda b: b.update({next(iter(b)): "high"}),                      # not a number
    lambda b: b.update({next(iter(b)): True}),                        # a boolean
])
def test_parse_belief_rejects_bad_answers(change):
    b = dict(GOOD_BELIEF)
    change(b)
    with pytest.raises(ParseError):
        parse_belief(json.dumps(b), PTS)


def test_parse_belief_rejects_percentages():
    reply = {k: v * 100 for k, v in GOOD_BELIEF.items()}              # 45 instead of 0.45
    with pytest.raises(ParseError):
        parse_belief(json.dumps(reply), PTS)


# ---------- the decision ----------

def test_accept_with_and_without_reason():
    assert parse_decision('{"action": "accept", "reason": "Fair price."}', SCN.m, G) == \
        ("accept", None, "Fair price.")
    assert parse_decision('{"action": "accept"}', SCN.m, G) == ("accept", None, None)
    assert parse_decision('{"action": "accept", "reason": 7}', SCN.m, G) == ("accept", None, None)


def test_counter_reads_price_and_rounds_to_tick():
    assert parse_decision('{"action": "counter", "price": 100.766, "reason": "Band says lower."}',
                          SCN.m, G) == ("counter", 100.766, "Band says lower.")
    assert parse_decision('{"action": "Counter", "price": "100.76549"}', SCN.m, G) == \
        ("counter", 100.765, None)


@pytest.mark.parametrize("reply", [
    '{"action": "counter"}',                                          # no price
    '{"action": "counter", "price": "low"}',                          # not a number
    '{"action": "counter", "price": 150.0}',                          # wildly out of range
    '{"action": "walk away"}',                                        # not an allowed action
    '{"price": 100.7}',                                               # no action
])
def test_parse_decision_rejects_bad_decisions(reply):
    with pytest.raises(ParseError):
        parse_decision(reply, SCN.m, G)
