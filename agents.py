"""Reads client's response and turns its replies into numbers.

A reply that can't be used raises ParseError, which the runner treats as a
malformed turn and retries (up to 2 times). A missing or unusable `reason`
never raises: the action fields alone decide whether a reply is valid.
"""
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from generator import price_dp, value_dp


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



# Connection to SoCLaaS 

MAX_RETRIES = 2                        # a malformed reply is asked again up to three times
MAX_TOKENS = 400                      


class ServedModelMismatch(RuntimeError):
    """The API answered with a different model from the one recorded for this run."""


def load_env(path=".env"):
    """Read KEY=VALUE lines from .env (no extra packages)."""
    lines = Path(path).read_text().splitlines()
    return {k.strip(): v.strip() for k, v in (l.split("=", 1) for l in lines
                                              if "=" in l and not l.lstrip().startswith("#"))}


def make_api(env_path=".env"):
    """An OpenAI-compatible client pointed at SoCLaaS, using the key and URL in .env."""
    env = load_env(env_path)
    return OpenAI(api_key=env["SOCLAAS_API_KEY"], base_url=env["SOCLAAS_BASE_URL"], timeout=120)


@dataclass
class Turn:
    """One question put to the client, and everything that came back."""
    parsed: object = None                         # the parsed answer, or None if every attempt failed
    raw: list = field(default_factory=list)       # every raw reply, failed ones included
    errors: list = field(default_factory=list)    # why each failed attempt was rejected
    seeds: list = field(default_factory=list)     # the seed sent with each attempt
    served: str = None                            # model name the API reported
    seconds: float = 0.0

    @property
    def malformed(self):
        return self.parsed is None


class LLMClient:
    """Sends a conversation to one model with the study's fixed settings.

    The first reply fixes the served model name for the run, unless `served` is
    given; any later reply from a different model stops the run.
    """

    def __init__(self, model, api, temperature=0.5, served=None):
        self.model = model
        self.api = api
        self.temperature = temperature
        self.served = served

    def _call(self, messages, seed):
        r = self.api.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, seed=seed,
            reasoning_effort="none", max_tokens=MAX_TOKENS)
        if self.served is None:
            self.served = r.model
        elif r.model != self.served:
            raise ServedModelMismatch(f"expected {self.served!r}, got {r.model!r}")
        return r.choices[0].message.content or "", r.model

    def ask(self, messages, parse, seed):
        """Ask once; if the reply can't be parsed, ask again with the next seed, up to MAX_RETRIES times."""
        turn = Turn()
        start = time.time()
        for attempt in range(1 + MAX_RETRIES):
            s = seed + attempt
            text, turn.served = self._call(messages, s)
            turn.raw.append(text)
            turn.seeds.append(s)
            try:
                turn.parsed = parse(text)
                break
            except ParseError as e:
                turn.errors.append(str(e))
        turn.seconds = time.time() - start
        return turn

