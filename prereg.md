# Pre-registration: disclosure of dealers' trade data to an LLM client

Written 2026-10-23, after the no-disclosure arms (U, U_noelicit) were run and before any episode with the disclosure (B, B_noelicit) exists. The analysis code is fixed at commit `21b2f76` (`disclosure_results.py`); this file is committed before the B arms are run.

## Research question

Does disclosing a summary of other dealers' reported purchases (the "band") to an LLM client improve its outcome when bargaining with a dealer, and does the client use the band as a Bayesian would or anchor on its number?

## Setup (fixed)

- **Game:** the client buys $10mm of an off-the-run 10-year US Treasury note from one dealer. The dealer quotes; the client accepts or makes one counteroffer; the dealer accepts any counter at or above its cost and rejects any below. Private values: the dealer's cost and the client's limit, drawn around an evaluated price shown to both.
- **Dealer:** a fixed rational algorithm, identical in every arm; its quote is rounded to 0.001.
- **Client:** `ornith1.5:35b` on SoCLaaS, temperature 0.5, thinking off, up to 2 retries per malformed reply. Chosen by a selection rule fixed before any candidate was run (`a08cd64`, `1b1a385`). Prompts frozen at `ebb20db` (tag `prompts-frozen`).
- **The band:** "Reported purchases by other dealers of similar notes yesterday (10 trades) imply a typical dealer cost of X; this dealer's own cost is within [low, high] about two times in three." The stated range is accurate by construction.

## Design

A 2 × 2 of disclosure × belief elicitation, each arm on scenarios 1–50 with 2 replicates (100 episodes per arm), all in `logs/main_ornith1.5-35b.jsonl`:

| | Elicited: beliefs asked before and after the quote; a reason with the decision | Not elicited: quote and decision in one turn |
|---|---|---|
| No band | U (run) | U_noelicit (run) |
| Band | B (to be run) | B_noelicit (to be run) |

Arms with the same elicitation use the same API seeds for each scenario and replicate, so each band/no-band pair differs only in the band paragraph.

## Primary hypothesis

**H1:** the band raises the share of the surplus the client captures when it is only bargaining: `client_capture` in **B_noelicit** exceeds **U_noelicit**.

- `client_capture` = (limit − price) / (limit − cost) if a deal is made, else 0.
- **Test:** for each scenario, average the two replicates in each arm and take the difference; average over the 50 scenarios. 95% confidence interval from a bootstrap over scenarios, 10,000 resamples, generator seed 0.
- **Decision rule:** H1 is supported if the lower bound of the 95% CI is above 0.
- **Reference:** the rational client facing the same dealer gains +0.111 on these 50 scenarios.

## Secondary analyses (stated in advance; none can replace H1)

- **S1:** B − U (the elicited arms), by the same method.
- **S2:** the interaction (B − U) − (B_noelicit − U_noelicit): does being asked about beliefs change the band's effect?
- **S3:** efficiency, `E_client` = Σ over scenarios of the LLM's gain in client surplus ÷ Σ of the rational value of disclosure, for both comparisons.
- **S4:** no-deal rates, band vs no band, by an exact McNemar test on episodes paired by scenario and replicate; counter rates; rule violations.

## Anchoring predictions

The band's typical cost `avg` is recorded for every scenario, though shown only in the band arms, so distances to it can be compared across arms.

- **A1 (tested):** with the band, counters move towards `avg`. The mean |counter − avg| (in g, per stated counter, averaged within scenario) is smaller in B_noelicit than U_noelicit, and smaller in B than U. Same paired method and 95% CI; supported if the CI's upper bound is below 0.
- **A2 (descriptive):** how the band is used. Reported per arm: the share of counters within 0.1g of `avg`; the rank correlation between counter and the client's own limit; and the most common counter levels. A Bayesian client's counter would depend on the band's width and its own limit; counters on `avg` regardless of the limit indicate anchoring on the number.

## Beliefs (elicited arms; descriptive)

Stated beliefs are made non-decreasing (isotonic fit) and compared with the benchmark belief at the 9 prices: the prior in U, the prior updated with the scenario's band in B.

- Belief error (mean squared difference, pre-quote), B − U.
- Uptake: the distance from the stated belief's median to `avg`, B − U. The band is registered if the CI's upper bound is below 0.
- Planning gap: |counter − rational counter under the correct belief for that arm|, B − U.
- Four-cell reading from the two CIs (belief error, planning gap): substitution, anchoring, planning failure, or inert.
- Quote inference: the mean change in stated probability from before to after the quote.

These are read as descriptive because of what the U arms already showed (below).

## Known before writing (U arms, Step 8)

- U: mean capture 0.356; U_noelicit: 0.405. The client countered in 99 and 98 of 100 episodes, on average about 1.3g below the evaluated price, with about half of counters rejected. The rational client captures 0.621 on these scenarios.
- Check (a) **failed**: stated beliefs do not predict counters (Spearman ρ = +0.08, p = 0.22). Check (b) passed (U − U_noelicit = −0.050, 95% CI [−0.168, +0.066]). Check (c) passed: the expected 95% CI half-width for a band effect is about 0.054.
- Counters cluster on numbers the client is given: the briefing's "typical dealer's cost" (1.0g below the evaluated price) and its 80% range edge (1.9g), and, when beliefs are asked, the belief question's prices (1.25g, 1.5g, 2.0g). The band adds a new number, `avg`, which A1 and A2 test directly.

## Exclusions and data handling

- Episodes whose decision is malformed after retries are excluded from outcome measures and counted per arm.
- Never used: plumbing and pilot runs, payoff-probe (`U_probe`) episodes, and the 90 mislabelled episodes in `logs/invalid_U_noelicit_81e0da9.jsonl`.
- The runner stops if the served model name changes; any such run is reported.
- No episodes are rerun or dropped because of their results.

## Held-out confirmation

After the main analysis, U_noelicit and B_noelicit are run once on scenarios 1001–1020 (never inspected), and H1 is re-estimated with the same code, unchanged. Reported separately.

## Deviations from the original plan, made before this pre-registration

1. The pre-chosen client model (`qwen3.6:35b`) failed its viability pilot; the fallback rule chose `ornith1.5:35b`.
2. Counters are rounded down to the 0.001 tick, not to the nearest tick (found in that pilot; it did not change its result).
3. A secondary model was planned and then dropped; the design became the 2 × 2 above, and check (b) was redefined, before any check was run or result seen.
4. A bug made 90 control episodes elicited; they were set aside and rerun after a fix.
5. The primary comparison was moved from the elicited arms (B − U) to the no-elicit arms (B_noelicit − U_noelicit) after the U-side data and before any band data, because the elicited arms' counters are partly pulled by the belief question's prices.

Any further deviation will be reported with its reason.
