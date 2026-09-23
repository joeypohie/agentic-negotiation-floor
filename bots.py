# bots.py
from generator import price_dp

def scripted_dealer_quote(dv, g):
    return round(dv.m + g, price_dp)

def scripted_client_respond(cv, g, p):
    if p <= cv.r - 0.5 * g:
        return "accept", None
    return "counter", round(cv.m, price_dp)

def scripted_dealer_decide(dv, g, q):
    return "accept" if q >= dv.c else "reject"

def random_bots(rng):
    def quote(dv, g):
        return round(dv.m + rng.uniform(-3*g, 3*g), price_dp)

    def respond(cv, g, p):
        if rng.random() < 0.3:
            return "accept", None
        return "counter", round(cv.m + rng.uniform(-3*g, 3*g), price_dp)

    def decide(dv, g, q):
        return "accept" if rng.random() < 0.5 else "reject"   # deliberately irrational

    return quote, respond, decide