# Analysis reference

Everything needed to analyse the experiment in one place: where the data is, what each arm is for, what a log line contains, how each measure is defined, and the benchmark numbers to compare against. The history of decisions is in `experiment_log.md`; the full design is in `CLAUDE.md`.

---

## Data

| File | Contents | Used in the analysis? |
|---|---|---|
| `logs/main_ornith1.5-35b.jsonl` | All four arms of the main experiment (U and U_noelicit now; B and B_noelicit after Step 9) | **Yes** |
| `logs/heldout_ornith1.5-35b.jsonl` | Step 11 held-out confirmation (seeds 1001–1020), once run | Reported separately |
| `logs/invalid_U_noelicit_81e0da9.jsonl` | 90 episodes mislabelled by the `NO_ELICIT` bug | **Never** |
| `logs/pilot_*.jsonl`, `logs/plumbing_qwen3.5-9b.jsonl` | Model selection and plumbing runs (incl. `U_probe`) | **Never** |

**Model:** `ornith1.5:35b` (Ornith-1.5-35B-A3B) on SoCLaaS; temperature 0.5; thinking off; up to 2 retries per malformed reply. **Prompts frozen** at `ebb20db` (tag `prompts-frozen`).

---

## The four arms

All on seeds 1–50 × 2 replicates (100 episodes each), facing the same fixed rational dealer with the same rounded quote.

| | Elicited (beliefs before and after the quote; reason with the decision) | Not elicited (quote and decision in one turn) |
|---|---|---|
| **No band** | `U` | `U_noelicit` |
| **Band** | `B` | `B_noelicit` |

| Comparison | Role |
|---|---|
| **`B_noelicit − U_noelicit`** | **Primary** (H1): the band's effect on a client that is only bargaining |
| `B − U` | Main secondary; the arms the belief analysis uses |
| `(B − U) − (B_noelicit − U_noelicit)` | Interaction: does being asked about beliefs change the band's effect? |
| `U − U_noelicit` | Does asking about beliefs change play without the band? (Step 8 check (b)) |

**API seeds:** each scenario and replicate has a base seed `seed × 1000 + replicate × 100`. Arms with the same elicitation use the same seeds (U with B, U_noelicit with B_noelicit), so each pair differs only in the band (common random numbers).

---

## What one log line contains

One JSON object per episode:

| Field | Meaning |
|---|---|
| `key.seed`, `key.condition`, `key.replicate` | Which episode |
| `run` | Model requested and `served`, provider, temperature, reasoning mode, config hash, `git_commit`, `git_dirty`, numpy and Python versions |
| `scenario.m`, `.c`, `.r` | Evaluated price, dealer's cost, client's limit |
| `band.avg`, `.low`, `.high`, `.s` | The band (in every line for reference); `band.shown` is true only in B and B_noelicit |
| `band.d`, `band.prints` | This dealer's deviation and the 10 prints: audit only, never shown to the client |
| `quote` | The dealer's quote (rounded to 0.001) |
| `beliefs.pre`, `beliefs.post` | The 9 stated probabilities, lowest price first; `null` in no-elicit arms or if unreadable |
| `decision.action`, `.counter_stated`, `.reason` | What the client said: accept or counter, the counter it stated (rounded down to 0.001), and its reason (elicited arms only) |
| `episode.status`, `.price` | What happened under the rules: `quote_accepted`, `counter_accepted` or `counter_rejected`; the trade price |
| `outcome` | `client_capture`, `client_surplus`, `dealer_surplus`, `no_deal`, `clawback`, `violations`; `null` if the decision was malformed |
| `malformed_decision` | True if the decision failed all its attempts (excluded from outcome measures) |
| `turns` | Per question: every raw reply, rejection reasons, seeds, served name, seconds |
| `messages` | The full conversation the client saw |

A counter at or above the quote counts as accepting the quote, so `decision.counter_stated` can differ from `episode.price`.

---

## Measures

`g` = 0.0338731 price points (about $3,400 on $10mm). Distances below are in `g` from the evaluated price `m` unless stated.

**Outcome (per episode, from `outcome`)**
- `client_capture` = (r − price) / (r − c) if a deal, else 0. **The primary outcome.**
- `client_surplus` = r − price if a deal, else 0. `no_deal` = 1 if no deal.

**Comparing two arms (all comparisons):** for each scenario, average the replicates in each arm, take the difference, then average over scenarios. 95% CI from a bootstrap over scenarios (10,000 resamples). Episodes with a malformed decision are left out.

**Primary decision rule (H1):** supported if the 95% CI of `B_noelicit − U_noelicit` in `client_capture` lies above 0.

**Efficiency:** `E_client` = Σ over scenarios of [mean LLM `client_surplus` in the band arm − in the no-band arm] ÷ Σ over scenarios of the rational value of disclosure. Ratio of sums, so scenarios with almost no value can't distort it. Computed for both comparisons.

**No-deal rates:** McNemar test on episodes paired by scenario and replicate (exact binomial test on the pairs where only one arm failed to trade).

**Anchoring (all arms; the band's `avg` is in every log line, though shown only in B arms)**
- **A1, distance to the band:** |counter − `avg`|, in g, per counter. Prediction: smaller in B than U, and in B_noelicit than U_noelicit.
- **A2, how the band is used:** the share of counters within ±0.1g of `avg`; and the rank correlation between counter and limit (both in g from `m`). A Bayesian client's counter depends on its limit and the band's width; a counter that sits on `avg` regardless of the limit is anchoring.
- Also reported: the counter levels most often used, compared with the briefing's typical cost (1.0g below `m`), its 80% range (0.53g and 1.9g below) and the belief-question prices (0.25g to 3.0g below).

**Beliefs (elicited arms only; descriptive, since Step 8 found stated beliefs do not drive counters)**
- **Stated belief as a CDF:** the 9 probabilities made non-decreasing (isotonic fit), joined by straight lines, anchored at 0 at m − 6g and 1 at m.
- **Benchmark belief:** in U, the prior; in B, the prior updated with that scenario's band (`benchmarks.band_posterior`).
- **Belief error:** mean squared difference between the stated (made non-decreasing) and benchmark probabilities at the 9 prices. Pre-quote belief for the main comparison.
- **Uptake (Step 9 check):** the distance from the stated belief's median to `avg` is smaller in B than in U.
- **Planning gap:** |counter − rational counter|, where the rational counter uses the correct belief for that arm (prior in U, band posterior in B) and the true limit.
- **Four-cell table (B vs U):** whether belief error falls (CI below 0) crossed with whether the planning gap falls.
- **Quote inference:** the change in stated belief from before to after the quote, averaged over the 9 prices. The rational dealer's quote carries little information (it explains 18% of the variation in cost), so a large shift is over-reading.

**Manipulation check:** the dealer's quote is identical in all four arms for every scenario.

---

## Benchmark numbers (rational client vs rational dealer, quote rounded to 0.001)

| | Seeds 1–50 (the main scenarios) | Seeds 1–1000 (design figures) |
|---|---|---|
| Rational capture without band (U*) | 0.621 | 0.652 (unrounded quote) |
| Rational capture with band (B*) | 0.732 | 0.756 (unrounded quote) |
| **Predicted effect on capture (B* − U*)** | **0.111** | |
| VoD, as a share of total surplus (ratio of sums) | 15.8% | **11.35%** |
| Mean VoD per scenario | 0.394g | |
| Quote accepted, U* / B* | 92% / 22% | 83% / 38% (unrounded quote) |
| No deal, U* / B* | 4% / 8% | 6.5% / 6.5% (unrounded quote) |

Seeds 1–50 happen to have more value from the band than the 1,000-scenario average.

---

## Step 8 results so far (ornith1.5:35b)

| | U | U_noelicit |
|---|---|---|
| Episodes | 100 | 100 |
| Mean `client_capture` | 0.356 | 0.405 |
| Countered | 99 | 98 |
| Counter accepted / rejected | 46 / 53 | 45 / 50 |
| Mean counter | 1.34g below m | 1.30g below m |
| Most common counter levels | 1.0g (35), 1.5g (23), 1.25g (20) | 1.0g (46), 1.9g (31) |

Checks: (a) fail (ρ = +0.08; stated beliefs don't drive counters), (b) pass (U − U_noelicit = −0.050, 95% CI [−0.168, +0.066]), (c) pass (expected half-width 0.054 vs predicted effect 0.111).

---

## Code

| File | What it does |
|---|---|
| `analysis.py` | All measures above (reads logs only; never calls the API) |
| `baseline_checks.py` | Step 8 checks |
| `disclosure_results.py` | Step 10: the comparisons, anchoring and belief measures, and the report (primary, secondary, anchoring, beliefs). Kept separate from `analysis.py` so the Step 8 code stays as it was when the checks ran |
| `pilot_summary.py` | Model selection checks (pilot logs only) |
