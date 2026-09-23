from dataclasses import dataclass

# ---- view controls ----

@dataclass(frozen=True)
class DealerView:
    m: float
    c: float #own cost

@dataclass(frozen=True)
class ClientView:
    m: float
    r: float #own limit

# ---- records ----

@dataclass(frozen=True)
class Episode:
    """What happened. No interpretation."""
    seed: int
    quote: float # p - the dealer's first move
    client_action: str # 'accept' or 'counter'
    counter: float | None # q if the client countered
    dealer_action: str | None # 'accept' or 'reject' if there was a counter
    status: str # how it ended
    price: float | None # ageed price, None if no deal

@dataclass (frozen=True)
class Outcome:
    seed: int
    status: str
    price: float | None
    client_capture: float
    dealer_surplus: float
    client_surplus: float
    no_deal: int #0 or 1
    clawback: float | None
    violations: tuple[str, ...]

# ---- rules ----

def play(scn, g, dealer_quote, client_respond, dealer_decide) -> Episode:
    #force the asymmetry
    dv = DealerView(m=scn.m, c=scn.c)
    cv = ClientView(m=scn.m, r=scn.r)

    p = dealer_quote(dv, g)

    action, q = client_respond(cv, g, p)

    # a counter at or above the quote is acceptance of the quote
    if action == "counter" and q >= p:
        action, q = "accept", None

    if action == "accept":
        return Episode(scn.seed, p, "accept", None, None, "quote_accepted", p)

    decision = dealer_decide(dv, g, q)
    if decision == "accept":
        return Episode(scn.seed, p, "counter", q, "accept", "counter_accepted", q)
    return Episode(scn.seed, p, "counter", q, "reject", "counter_rejected", None)


def score(scn, ep) -> Outcome:
    S = scn.r - scn.c          # total surplus — the size of the pie

    v = []
    if ep.quote < scn.c:
        v.append("dealer_quote_below_cost")
    if ep.counter is not None and ep.counter > scn.r:
        v.append("client_counter_above_limit")
    if ep.status == "quote_accepted" and ep.quote > scn.r:
        v.append("client_accepted_above_limit")
    if ep.status == "counter_rejected" and ep.counter >= scn.c:
        v.append("dealer_rejected_profitable_counter")

    clawback = (ep.quote - ep.counter) / S if ep.counter is not None else None

    if ep.price is None:
        return Outcome(scn.seed, ep.status, None, 0.0, 0.0, 0.0, 1, clawback, tuple(v))

    return Outcome(
        seed=scn.seed, status=ep.status, price=ep.price,
        client_capture=(scn.r - ep.price) / S,
        dealer_surplus=ep.price - scn.c,
        client_surplus=scn.r - ep.price,
        no_deal=0, clawback=clawback, violations=tuple(v),
    )