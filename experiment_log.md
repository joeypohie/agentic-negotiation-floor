# Experiment log

A dated record of every LLM run: what was run, why, with which code, where the episodes are, and what came out. Times are UTC, as recorded in the episode logs (Singapore time is UTC+8). Episode logs live in `logs/` and are not committed; each line records the git commit and whether the working tree had uncommitted changes (`git_dirty`).

Fixed settings for every run unless stated: SoCLaaS API, temperature 0.5, thinking off (`reasoning_effort="none"`), up to 2 retries per malformed reply, dealer = fixed rational dealer with its quote rounded to 0.001.

---

## 2026-10-16 — Settings screen (model selection rule, step 1)

**Script:** `check_models.py`. **Models:** `qwen3.6:35b`, `qwen3.6:27b`, `gemma4:26b`, `ornith1.5:35b`.

**Result:** all four accept temperature 0.5, a seed and JSON mode. Only `reasoning_effort="none"` turns thinking off; `extra_body={"think": False}` and `/no_think` had no effect. With the full configuration, 3 of 3 calls per model returned correct JSON with no thinking.

| Model | Served as | Thinks by default | Seconds per call (short prompt) |
|---|---|---|---|
| `qwen3.6:35b` | `qwen3.6:35b` | No | 0.2–0.5 |
| `qwen3.6:27b` | `qwen3.8:27b` | Yes | 0.3–0.5 |
| `gemma4:26b` | `gemma4:26b` | Yes | about 1.5 (wraps JSON in code fences) |
| `ornith1.5:35b` | `ornith1.5:35b` | Yes | about 0.2 |

`qwen3.6:27b` is routed to `qwen3.8:27b`; re-checked on 2026-10-23, still the case.

**Decisions the same day:** the client is told the dealer accepts any counter at or above its cost; the dealer's quote is rounded to the nearest 0.001 in LLM and rational episodes alike (VoD becomes +11.35%); the client's `reason` comes after its action; condition labels stay U and B.

---

## 2026-10-21 — Live connection check

**Script:** `check_client.py`. **Model:** `qwen3.5:9b` (plumbing model). **Scenario:** seed 901, condition U, one episode's client turns. **Commit:** `063c75c`.

**Result:** all three turns parsed on the first attempt; served name `qwen3.5:9b` throughout; about 4–5 seconds per call. The model played poorly, as expected of a small model: its belief after the quote collapsed onto a single price, and it "countered" above the quote (which counts as accepting).

---

## 2026-10-21 to 2026-10-23 — Plumbing run (Step 6 gate)

**Command (final form):**
`python run.py --model qwen3.5:9b --conditions U,B,U_noelicit,U_probe --seeds 901-905 --reps 1 --log logs/plumbing_qwen3.5-9b.jsonl`
(an earlier run had logged seeds 901–907 in U, B and U_noelicit into the same file).

**Log:** `logs/plumbing_qwen3.5-9b.jsonl`: 26 episodes (U 7, B 7, U_noelicit 7, U_probe 5). **Commit:** `063c75c`, with uncommitted changes (`git_dirty: true`).

**Result: plumbing gate passed.** 0 malformed of 64 client turns (gate: under 2%). Six turns needed a retry and all recovered; five of the six were the model replying `"reject"`, which is not an allowed action. Not analysed.

**Change made because of it:** one sentence added to the rules in `render.py`: "There is no option to reject the quote outright: if you do not want to trade at the quote, make a counteroffer."

---

## 2026-10-23 — Selection rule and model decision committed (before any candidate ran)

- **`a08cd64`:** viability gate made precise (five checks in `pilot_summary.py`), payoff probe `U_probe` added to `run.py`, no-reject sentence.
- **`1b1a385`:** `qwen3.6:35b` pre-chosen as primary (largest candidate, no default thinking, served as requested, fast), confirmed by its pilot; fallback to the original rule over the other candidates if it fails. `qwen3.5:9b` secondary, as a capability comparison, not piloted.

---

## 2026-10-23 06:18–06:21 — Pilot: `qwen3.6:35b`

**Command:** `python run.py --model qwen3.6:35b --conditions U,U_probe --seeds 501-520 --reps 1 --log logs/pilot_qwen3.6-35b.jsonl`

**Log:** 40 episodes (U 20, U_probe 20). **Commit:** `1b1a385`, clean. Served `qwen3.6:35b`. Median 3.9 s per episode.

**Result: fails the viability gate on check 2.**

| Check | Result |
|---|---|
| 1 Readable replies | pass: 0.0% malformed of 120 turns |
| 2 Sensible counters | **fail: 67% of 39** |
| 3 Refuses losing trades | pass: 0 of 20 probe quotes taken |
| 4 Counters track limit | pass: ρ = +0.63 over 20 probe counters |
| 5 Coherent beliefs | pass: 100% of 80 |

All 13 non-sensible counters were probe episodes; all 19 counters in U were sensible. Six were the model countering exactly at its 4-decimal limit, pushed just above it by the reader rounding to the nearest 0.001 (our error); one was its own rounding of its limit; six were real errors, countering at the briefing's "typical dealer's cost" when that was above its limit. Counting the seven as sensible gives 85%, still a fail.

**Correction (`a3184e1`):** counters are now rounded **down** to the 0.001 tick, so a counter can never exceed the price the client stated. Applied to every later run; it does not change this result.

---

## 2026-10-23 06:34–07:02 — Fallback pilots

**Commands:** as above, with `--model qwen3.8:27b`, `gemma4:26b`, `ornith1.5:35b` and logs `logs/pilot_qwen3.8-27b.jsonl`, `logs/pilot_gemma4-26b.jsonl`, `logs/pilot_ornith1.5-35b.jsonl`. Each 40 episodes (U 20, U_probe 20). **Commit:** `a3184e1`, clean. Each served exactly the model requested.

| Model | Gate | 1 Malformed | 2 Sensible counters | 3 Probe quotes taken | 4 ρ | 5 Coherent beliefs | Median s/episode |
|---|---|---|---|---|---|---|---|
| `qwen3.8:27b` | fail | 33.3% | 57% of 35 | 0 of 20 | +0.58 | 100% of 40 | 26.8 |
| `gemma4:26b` | pass | 0.0% | 100% of 34 | 0 of 20 | +0.99 | 100% of 80 | 10.7 |
| `ornith1.5:35b` | pass | 0.0% | 92% of 40 | 0 of 20 | +0.86 | 99% of 80 | 3.0 |

`qwen3.8:27b` wrote out step-by-step maths before the JSON in every pre-quote belief turn and ran out of tokens, so all 40 of those turns were malformed.

**Choice (rule step 3):** `gemma4:26b` and `ornith1.5:35b` both pass with 0.0% malformed, a tie within 1 point, so the faster model is chosen: **`ornith1.5:35b` is the primary client model.** Outcome measures were not looked at.

**Commit `ebb20db`:** `config.yaml` model section set to `ornith1.5:35b`; prompts frozen, git tag `prompts-frozen`.

---

## 2026-10-23 08:22–08:27 — Step 8: U run, and the first control episodes

**Commands:**
`python run.py --model ornith1.5:35b --conditions U --seeds 1-50 --reps 2 --log logs/main_ornith1.5-35b.jsonl`
`python run.py --model ornith1.5:35b --conditions U_noelicit --seeds 1-10 --reps 1 --log logs/main_ornith1.5-35b.jsonl`

**Log:** `logs/main_ornith1.5-35b.jsonl`: U 100 episodes (seeds 1–50 × 2), U_noelicit 10 (seeds 1–10). **Commit:** `085f7df` (baseline check definitions committed before any U data), clean. Served `ornith1.5:35b`. 0 malformed of 300 U turns. Median 3.0 s per episode.

**Checks:** not run at this point.

---

## 2026-10-23 — Design changed before any check was run or result seen

**Commit `81e0da9`:** one model only (`qwen3.5:9b` dropped); a 2 × 2 of disclosure × elicitation (`U`, `B`, `U_noelicit`, `B_noelicit`), each on seeds 1–50 × 2; baseline check (b) redefined as a paired comparison of U and U_noelicit over 50 scenarios.

---

## 2026-10-23 09:01–09:06 — Control episodes completed, invalid (bug)

**Command:** `python run.py --model ornith1.5:35b --conditions U_noelicit --seeds 1-50 --reps 2 --log logs/main_ornith1.5-35b.jsonl` at `81e0da9`.

**Problem:** `run.py` tested `condition == "NO_ELICIT"` (a string) instead of `condition in NO_ELICIT`, so the 90 new "U_noelicit" episodes were run with the belief questions: near-copies of U, with the same API seeds. Found when check (b) came out almost exactly zero and the turn counts did not match.

**Action:** the 90 episodes were moved, not deleted, to `logs/invalid_U_noelicit_81e0da9.jsonl`. **Commit `139dfbb`:** fixed to `condition in NO_ELICIT`, and `baseline_checks.py` now stops if any no-elicit episode has belief turns. Checked with one fake episode per condition before rerunning.

---

## 2026-10-23 09:19–09:20 — Control episodes rerun; Step 8 checks

**Command:** the same U_noelicit command, at `139dfbb`. **Log:** U_noelicit now 100 genuine episodes (10 at `085f7df`, 90 at `139dfbb`), each with only the decision turn. 0 malformed.

**Command:** `python baseline_checks.py logs/main_ornith1.5-35b.jsonl`

| Check | Result |
|---|---|
| (a) elicitation not decorative | **fail**: ρ = +0.08, one-sided p = 0.218, over 100 U counters |
| (b) asking doesn't change play | pass: U − U_noelicit = −0.050, 95% CI [−0.168, +0.066] (U 0.356, U_noelicit 0.405); a wide interval |
| (c) power | pass: expected B − U half-width 0.054 vs predicted effect 0.111 |

**Diagnosis:** the client counters in almost every episode, far below what its stated beliefs imply (1.34g below `m` against 0.47g), and ignores its own limit. Without elicitation it counters mostly at the briefing's "typical dealer's cost" (1.0g below `m`, 46 of 98) or the edge of the stated 80% range (1.9g, 31). With elicitation, prices from the belief question's grid (1.25g, 1.5g, 2.0g) also become counter levels (51 of 100), which almost never appear without it. 87 of 100 U reasons cite the median or typical cost. About half of counters are rejected; capture is 0.36–0.41 against the rational client's 0.65.

**Decisions:**
- Proceed to pre-registration and B, reporting check (a) as a finding, rather than changing the frozen prompts or the model.
- **Primary comparison changed to `B_noelicit − U_noelicit`**, because the elicited arms' counters are partly pulled by the question grid. The elicited B − U becomes the main secondary. Made after the U-side data, before any B data; to be disclosed.
- Possible extension: the same design with the prior given only as gaps, without the "typical dealer's cost is about X" price.

---

## 2026-10-23 17:44 SGT — Pre-registration committed

**`21b2f76`:** `disclosure_results.py` (Step 10 analysis code). **`cfe5245`:** `prereg.md`. Both committed before any band data; the log held only U and U_noelicit at the time.

---

## 2026-10-23 09:53–09:58 — Step 9: B and B_noelicit; Step 10 report

**Command:** `python run.py --model ornith1.5:35b --conditions B,B_noelicit --seeds 1-50 --reps 2 --log logs/main_ornith1.5-35b.jsonl`

**Log:** B 100 episodes (belief turns and decision), B_noelicit 100 (decision only). **Commit:** `cfe5245`, clean. Served `ornith1.5:35b`. 0 malformed decisions in all four arms. Manipulation check: the dealer's quote is identical in all four arms for every scenario.

**Command:** `python disclosure_results.py logs/main_ornith1.5-35b.jsonl`

| Arm | Capture | No deal | Counters (accepted / rejected) | Mean counter |
|---|---|---|---|---|
| U | 0.356 | 53% | 100 (46 / 53) | 1.34g below m |
| B | 0.353 | 55% | 100 (44 / 55) | 1.27g below m |
| U_noelicit | 0.405 | 50% | 98 (45 / 50) | 1.30g below m |
| B_noelicit | 0.332 | 58% | 99 (41 / 58) | 1.34g below m |

**Primary (H1), B_noelicit − U_noelicit in capture: −0.073, 95% CI [−0.190, +0.038]. H1 not supported.** The rational client gains +0.111 on the same scenarios, which lies outside the interval: the LLM gained at most about a third of the rational benefit, and the point estimate is negative.

**Secondary:** B − U (elicited) −0.002 [−0.102, +0.097]; interaction (B − U) − (B_noelicit − U_noelicit) +0.071 [−0.062, +0.206]; U − U_noelicit −0.050 [−0.168, +0.066]. `E_client` −0.54 (no-elicit), −0.07 (elicited). No deal, band vs no band: 19 vs 11 discordant pairs without elicitation (McNemar p = 0.20), 11 vs 9 with (p = 0.82).

**Anchoring:** A1 supported in both comparisons: |counter − avg| falls by 0.158g [−0.235, −0.081] without elicitation (49 scenarios with counters in both arms) and by 0.257g [−0.326, −0.193] with it. A2: only 16% (B) and 19% (B_noelicit) of counters are within 0.1g of `avg` (4% and 15% in U, U_noelicit); the most common counter is still the briefing's typical cost (1.0g below m) in every arm; counters still ignore the client's limit (ρ between −0.18 and −0.03).

**Beliefs (elicited arms):** uptake yes, the stated belief's median moves 0.232g towards `avg` [−0.337, −0.137]; belief error not improved, +0.012 [−0.004, +0.030]; planning gap narrows by 0.245g [−0.331, −0.150]; four-cell reading **anchoring** (behaviour moves towards the rational counter without more accurate beliefs). Quote inference: after the quote, stated probabilities fall by 0.214 (U) and 0.221 (B) on average, a large shift from a quote that carries little information (over-reading), unchanged by the band.

**Reading:** the client treats the numbers it is given as targets. The band pulls its counters partly towards the disclosed typical cost, but it does not weigh the band's uncertainty or its own limit, so its outcome does not improve.

**Precision:** the primary CI is about ±0.11 wide, twice the ±0.054 the Step 8 power check predicted; that check counted replicate noise only and ignored how the band's effect varies between scenarios (as its output warned).

**Next:** Step 11 held-out confirmation on seeds 1001–1020 with `disclosure_results.py` unchanged. All four arms are run (1 replicate), because the script expects all four; the pre-registration names the no-elicit arms, and running U and B there too adds data without changing the code or the primary test.

---

## 2026-10-23 10:02–10:04 — Step 11: held-out confirmation

**Command:** `python run.py --model ornith1.5:35b --conditions U,B,U_noelicit,B_noelicit --seeds 1001-1020 --reps 1 --log logs/heldout_ornith1.5-35b.jsonl`

**Log:** `logs/heldout_ornith1.5-35b.jsonl`: 80 episodes (20 per arm), seeds 1001–1020, 1 replicate. **Commit:** `cfe5245`, clean; analysis code unchanged since `21b2f76`. Served `ornith1.5:35b`. 0 malformed. Manipulation check OK; no-elicit episodes had only the decision turn.

**Command:** `python disclosure_results.py logs/heldout_ornith1.5-35b.jsonl`

| Arm | Capture | No deal | Counters (accepted / rejected) | Mean counter |
|---|---|---|---|---|
| U | 0.278 | 70% | 20 (6 / 14) | 1.56g below m |
| B | 0.323 | 65% | 20 (7 / 13) | 1.32g below m |
| U_noelicit | 0.406 | 50% | 20 (10 / 10) | 1.25g below m |
| B_noelicit | 0.479 | 45% | 19 (10 / 9) | 1.21g below m |

**Primary (H1), B_noelicit − U_noelicit: +0.073, 95% CI [−0.063, +0.222]. Not supported.** The rational gain on these 20 scenarios is only +0.033 and lies inside the interval, so the held-out set cannot test whether the LLM falls short of the rational benefit.

**Secondary:** B − U +0.045 [−0.164, +0.254]; interaction −0.029 [−0.297, +0.259]; U − U_noelicit −0.128 [−0.297, +0.015]. `E_client` +1.42 (no-elicit) and +1.74 (elicited), not meaningful here: the rational gain in the denominator is tiny. No-deal differences: 1 vs 2 and 2 vs 3 discordant pairs (McNemar p = 1.0).

**Anchoring:** A1 supported without elicitation (−0.299g [−0.481, −0.131]); borderline with it (−0.159g [−0.313, +0.006]). A2: 25% (B) and 26% (B_noelicit) of counters within 0.1g of `avg`, against 10% (U) and 15% (U_noelicit); the briefing's typical cost (1.0g below m) is the most common counter in both band arms.

**Beliefs:** belief error worse with the band (+0.055 [+0.026, +0.090]); uptake not clear (−0.013g [−0.189, +0.170]); planning gap narrows (−0.452g [−0.622, −0.279]); four-cell reading **anchoring**. Quote inference: −0.133 (U), −0.235 (B).

**Comparison with the main experiment:** H1 not supported in either; the primary estimate changes sign (−0.073 main, +0.073 held-out), consistent with an effect near zero. Anchoring towards `avg` (A1, no-elicit) and the four-cell reading "anchoring" replicate, as does over-reading of the quote. Uptake in beliefs does not replicate clearly. With 20 scenarios and one replicate, the held-out intervals are wide.

---

## Open items

- ~~Find out what `ornith1.5:35b` is.~~ Ornith-1.5-35B-A3B: open-weight mixture-of-experts reasoning and coding model by Ornith AI, 35B parameters with about 3B active per token; recently released. Run with thinking off.
- ~~U and B in sequence or interleaved?~~ Sequential (2026-10-23): U, then pre-registration, then B.
- ~~Build the analysis code the Step 8 gates need.~~ `analysis.py` and `baseline_checks.py`, committed at `085f7df`.
- ~~Write and commit `prereg.md`.~~ Analysis code `disclosure_results.py` committed at `21b2f76`; `prereg.md` committed at `cfe5245` (2026-10-23 17:44 SGT), both before any band data.
- ~~Run B and B_noelicit, then the report.~~ Done at `cfe5245`: H1 not supported (−0.073, 95% CI [−0.190, +0.038]); A1 supported; four-cell reading anchoring.
- ~~Step 11 held-out.~~ Done at `cfe5245`: H1 not supported (+0.073, 95% CI [−0.063, +0.222]); anchoring towards `avg` and the four-cell reading replicate.
- Write-up and presentation.

---

## Template for the next entry

```
## YYYY-MM-DD HH:MM–HH:MM — <what> (<step>)

**Command:** `python run.py --model ... --conditions ... --seeds ... --reps ... --log logs/....jsonl`
**Log:** <file>: <n> episodes (<by condition>). **Commit:** <hash>, clean / dirty. Served <name>. Median <s> per episode.
**Result:** <gate checks or summary>
**Changes made because of it:** <none / what and commit>
```
