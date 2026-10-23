"""
Client's briefing 

The numbers used by the rational client are stored here,

quotable prices to 3 d.p and the privatae values to 4. 

The B briefing is the exact same as the U briefing, just with the addition of the band.

The dealer's cost and deviation are never shown.
"""

from scipy import stats

from generator import price_dp, value_dp

SIGMA_LN = 0.5
P10, P50, P90 = stats.lognorm(s=SIGMA_LN).ppf([0.1, 0.5, 0.9])     # 0.527, 1.000, 1.898 (in g)


def fp(x):
    """Formats prices to 3 decimal places"""
    return f"{x:.{price_dp}f}"


def fv(x):
    """Returns private value to 4 decimal places"""
    return f"{x:.{value_dp}f}"


def market_block(scn):

    """ 
    Persona on who the client is, its evaluated price and its limit
    """
    return (
        "You are a portfolio manager at a fund. You need to buy $10 million face value of an "
        "off-the-run 10-year US Treasury note, and you have asked a single dealer for a price.\n\n"
        f"The evaluated price for this note today is {fp(scn.m)}. It is an indicative price from a "
        "pricing vendor; nobody can trade at it.\n\n"
        f"Your limit is {fv(scn.r)}: the most you are willing to pay. "
        "The dealer does not know your limit."
    )


def prior_block(scn, g):
    lo, med, hi = P10 * g, P50 * g, P90 * g
    return (
        "The dealer has a private cost: the price at which it can source the note. You do not know it. "
        "Across dealers, the cost sits below the evaluated price by an amount that follows a log-normal "
        f"distribution: the median gap is {fv(med)} points, and 80% of dealers' gaps lie between "
        f"{fv(lo)} and {fv(hi)} points. In prices, a typical dealer's cost is about {fv(scn.m - med)}, "
        f"and 80% of dealers' costs lie between {fv(scn.m - hi)} and {fv(scn.m - lo)}."
    )


def band_block(scn, band):
    """The disclosure paragraph (condition B only)."""
    return (
        "Reported purchases by other dealers of similar off-the-run notes yesterday (10 trades) "
        f"averaged {fv(scn.m - band.avg)} below their evaluated prices. Applied to today's evaluated "
        f"price, that implies a typical dealer cost of {fv(band.avg)}. Individual dealers' costs vary "
        f"around the typical level, so this dealer's own cost is within [{fv(band.low)}, {fv(band.high)}] "
        "about two times in three."
    )


def rules_block():
    return (
        "How the negotiation works:\n"
        "1. The dealer quotes a price.\n"
        "2. You either accept the quote, and the trade is done at that price, or make one counteroffer "
        "at a lower price. There is no option to reject the quote outright: if you do not want to trade "
        "at the quote, make a counteroffer.\n"
        "3. If you counter, the dealer accepts any counteroffer at or above its own cost and rejects "
        "any below it. Its decision is final: if it rejects, there is no trade.\n"
        "Quotes are firm: once you accept, the dealer cannot withdraw. A counteroffer at or above the "
        "quote counts as accepting the quote. Prices are in points per 100 face value, to 3 decimals; "
        "on $10 million, 0.001 points is $100.\n\n"
        "Your profit is your limit minus the price you pay, or zero if there is no trade. "
        "Your goal is to maximise your expected profit."
    )


def client_brief(scn, g, band=None):
    """The client's full briefing: U if band is None, else B."""
    blocks = [market_block(scn), prior_block(scn, g)]
    if band is not None:
        blocks.append(band_block(scn, band))
    blocks.append(rules_block())
    return "\n\n".join(blocks)


### Questions to ask during an episode

BELIEF_GAPS = (3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25)    # x_j = m − g·gap (CLAUDE.md)


def belief_points(scn, g):
    """The 9 prices the client gives beliefs at, lowest first, to 4 decimals."""
    return [round(scn.m - g * k, value_dp) for k in BELIEF_GAPS]


def belief_question(scn, g):
    """Ask for P(dealer's cost < x) at each belief point, as JSON keyed by price."""
    keys = ", ".join(f'"{fv(x)}": <probability>' for x in belief_points(scn, g))
    return (
        "For each price below, give your probability (a number from 0 to 1) that the dealer's "
        "cost is below that price. Reply with JSON only, using exactly these keys:\n"
        "{" + keys + "}"
    )


def pre_quote_message(scn, g):
    return "The dealer has not quoted yet. " + belief_question(scn, g)


def post_quote_message(scn, g, p):
    return f"The dealer quotes {fp(p)}. " + belief_question(scn, g)


def decision_message(p, with_reason=True):
    """Ask the client to accept or counter. The reason, if asked for, comes after the action."""
    reason = ', "reason": "<one or two sentences>"' if with_reason else ""
    return (
        f"Do you accept the dealer's quote of {fp(p)}, or make a counteroffer? "
        "Reply with JSON only, in one of these two forms:\n"
        '{"action": "accept"' + reason + "}\n"
        '{"action": "counter", "price": <your counteroffer, to 3 decimals>' + reason + "}"
    )


def noelicit_message(p):
    """Control episodes (U_noelicit): the quote and the decision in one turn, no beliefs, no reason."""
    return f"The dealer quotes {fp(p)}. " + decision_message(p, with_reason=False)
