"""Reads client's response and turns its replies into numbers.

A reply that can't be used raises ParseError, which the runner treats as a
malformed turn and retries (up to 2 times). A missing or unusable `reason`
never raises: the action fields alone decide whether a reply is valid.
"""
import json
import math
import re

from generator import price_dp, value_dp

PRICE_RANGE_G = 20                     # reject counters more than 20g from m (CLAUDE.md)


class ParseError(ValueError):
    """A reply the runner cannot use."""


def extract_json(text):
    """The first JSON object in a reply, ignoring code fences (i.e.: from Gemma), <think> blocks and prose around it."""
    if not text:
        raise ParseError("empty reply")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        raise ParseError("no JSON object in reply")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    return obj


def _number(v, what):
    """A finite number from a JSON value; numeric strings are accepted, booleans are not."""
    if isinstance(v, bool):
        raise ParseError(f"{what} is a boolean")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ParseError(f"{what} is not a number: {v!r}") from None
    if not math.isfinite(x):
        raise ParseError(f"{what} is not finite")
    return x


def parse_belief(text, points):
    """The client's 9 probabilities, in the order of `points` (lowest price first).

    Keys are matched by value to 4 decimals, so "100.6944" and "100.69440" both work.
    The probabilities are returned as given; making them monotone is done at scoring.
    """
    obj = extract_json(text)
    by_price = {}
    for k, v in obj.items():
        try:
            by_price[round(float(k), value_dp)] = v
        except ValueError:
            continue                                   # ignore non-price keys
    out = []
    for x in points:
        key = round(x, value_dp)
        if key not in by_price:
            raise ParseError(f"no probability given for {x:.{value_dp}f}")
        prob = _number(by_price[key], f"probability at {x:.{value_dp}f}")
        if not 0.0 <= prob <= 1.0:
            raise ParseError(f"probability {prob} at {x:.{value_dp}f} is outside 0 to 1")
        out.append(prob)
    return out


def parse_decision(text, m, g):
    """(action, price, reason): ("accept", None, reason) or ("counter", q, reason).

    A counter price is rounded to the 3-decimal tick. Counters above the limit or at
    or above the quote are allowed here; env.play and score deal with those.
    """
    obj = extract_json(text)
    action = str(obj.get("action", "")).strip().lower()
    reason = obj.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    if action == "accept":
        return "accept", None, reason
    if action != "counter":
        raise ParseError(f"unknown action {obj.get('action')!r}")
    if "price" not in obj:
        raise ParseError("counter without a price")
    q = _number(obj["price"], "counter price")
    if abs(q - m) > PRICE_RANGE_G * g:
        raise ParseError(f"counter price {q} is more than {PRICE_RANGE_G}g from the evaluated price")
    return "counter", round(q, price_dp), reason
