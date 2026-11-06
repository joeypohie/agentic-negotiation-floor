# Can a regulator protect an AI trading agent by giving it more data?

**Short answer: not this one.** I gave an LLM trading agent the same market data that makes a rational trader about 11% of the surplus better off. The LLM gained nothing. It didn't ignore the data. It copied the numbers straight into its bids instead of reasoning with them.

This is my solo project for CS6101 *Rational Approaches to Cooperative Intelligence* at the National University of Singapore (MComp in AI, 2026). Before this I worked in FX risk management at a bank, and the project comes from that world. It asks what happens when an AI agent, not a person, has to bargain for a price in an RFQ.

📄 [Report (PDF)](report/CS6101%20Report_vF.pdf) · 🖥️ [Slides (PDF)](report/Presentation%20Slides_vF.pdf) · 🔍 [One negotiation, start to finish](notebooks/one_negotiation.ipynb)

**If you only open one thing**, make it [the one-negotiation notebook](notebooks/one_negotiation.ipynb). It follows a single trade through every condition, using the AI's actual prompts and replies. It shows what the AI was told, what it said it believed, what it bid and why, and why the same data that helped a rational client cost this one the trade.

---

## The question

AI agents are starting to negotiate prices for people. The agent on one side could be far more capable than the one on the other. One common way to protect the weaker side is to **give it information**. US regulators did this when they began publishing corporate bond trades (TRACE), and investors' trading costs fell.

So I asked:

1. If you give an LLM agent a summary of other dealers' trades, does it get a **better price**?
2. Does it use that information **the way a rational (Bayesian) trader would**, or does it just **anchor** on the numbers?

## The setup, in trading terms

- **Trade:** a client buys $10mm of an off-the-run 10-year US Treasury note from one dealer, by voice/chat RFQ.
- **Three moves:** the dealer quotes a price. The client either lifts it or counters lower. The dealer then accepts the counter if it covers the dealer's cost. Otherwise there's no trade. Quotes are firm (no last look).
- **Private information:** the dealer knows its own cost. The client knows the most it is willing to pay. Neither sees the other's number.
- **The disclosure ("the band"):** in some runs the client also sees other dealers' recent purchase levels, applied to today's evaluated price, with a stated range. This models FINRA publishing off-the-run Treasury trades, which today it doesn't.
- **Realistic scale:** price gaps come from realised 10-year yield volatility (Fed GSW curve). The typical dealer margin works out to about 0.43bp, or roughly 1/32nd. That's within the real range for this trade.
- **The client:** an open-weight 35B LLM (Ornith-1.5-35B), with reasoning off.
- **The benchmark:** a rational client that does the Bayesian maths exactly, facing the same dealer.

## What I found

| | Without the band | With the band | Change |
|---|---|---|---|
| Rational client's share of the surplus | 62% | 73% | **+11 points** |
| LLM client's share of the surplus | 41% | 33% | −7 points (95% CI −19 to +4), not significant |

![Client share of surplus, with and without the band](report/figures/fig2_results.png)

- **It anchors on whatever number it is shown.** Without the band, 77 of 98 counteroffers landed exactly on one of two prices from its briefing. With the band, a quarter of its counters sat on the band's lowest price. The dealer's cost is below that price only about 16% of the time, so 77% of those counters were rejected.
- **It weighs evidence the wrong way round.** The dealer's quote reveals almost nothing about its cost, yet the LLM moved its estimate a lot after seeing it (0.57 units, against 0.08 for a Bayesian). The band reveals a lot, yet the LLM moved only half as far as a Bayesian would.
- **Its stated beliefs barely drove its bids.** When asked, it gave sensible probabilities, but they didn't predict where it actually bid. It also ignored its own limit: a client willing to pay more didn't bid any differently.
- **The result held on 20 fresh scenarios** that I set aside at the start and never looked at during development.

**Why this matters for an eFX desk:** as LLM agents start handling RFQs and client chat, the same weaknesses carry straight over. How a reference price is shown to an agent, whether as a single level or as a range, can change how it trades as much as whether it is shown at all. Before you trust an agent with a price, test it against a benchmark where you know the right answer.

---

## How I worked (the part I'd most like you to see)

**1. I built the maths before calling any AI model, and it killed my first design.**
My original plan was the opposite question: does a dealer gain from seeing a client's past trading flow? Before writing any LLM code, I computed what a perfectly rational dealer would gain. The answer was about 0%. Even knowing the client's limit *exactly* was worth only about 2%. The reason is structural: in this protocol the client moves last, so its counter is almost always accepted, and its behaviour reveals almost nothing. The same calculation showed that information flowing *to the client* was worth about 11%. So I switched the study around before spending any time or compute on the dead end.

**2. I compared like with like.**
I first planned an LLM on both sides of the trade. I replaced the dealer with a fixed rational algorithm, so the LLM client and the rational client face the *same* counterparty, and the gap between them measures only the client.

**3. I set every pass/fail rule before seeing the numbers.**
The project went in steps, each with a test (a "gate") written before it ran. There are 213 automated tests. For example, one test checks that the client's prompt with the band is word for word the prompt without it, plus the band paragraph and nothing else. Another checks that the dealer quotes identically in every condition.

**4. I chose the AI model by a rule fixed in advance, and followed it when it went against me.**
My first-choice model failed a basic competence check: only 67% of its counteroffers were below the quote and within its own limit, against a 90% bar. The pre-written fallback rule picked the replacement. I didn't choose by which model gave the most interesting results.

**5. I found my own bug and reported it.**
A one-line bug in the runner (comparing to a string instead of checking membership in a list) meant 90 "no-questions" episodes were actually run *with* questions. I moved the bad data aside rather than deleting it, fixed and tested the code, reran the episodes, and recorded it all in the [experiment log](notebooks/experiment_log.ipynb).

**6. I pre-registered the main test and reported a null result.**
The main hypothesis and its analysis code were committed to git before any data with the band existed ([prereg.md](prereg.md)). The hypothesis wasn't supported, and the report says so plainly. Then I re-ran everything on held-out scenarios to check.

**7. I kept a dated record of every run.**
The [experiment log](notebooks/experiment_log.ipynb) records what was run, when, on which git commit, and what came out, including the parts that went wrong. Below each entry, a code cell reruns that step's analysis script, so you can check the written numbers against the script's own output.

---

## What's in this repo

| File | What it does |
|---|---|
| `generator.py` | Draws each scenario: evaluated price, dealer cost, client limit, and the band |
| `env.py` | The game: moves, payoffs, and rule violations |
| `rational.py` | The rational client and dealer (best counter, walk-away price, dealer's optimal quote) |
| `benchmarks.py` | The Bayesian update of the client's belief about the dealer's cost, given the band |
| `render.py` | Writes the client's prompt, identical across conditions except for the band paragraph |
| `agents.py` | Calls the LLM, parses its JSON replies, retries malformed ones |
| `run.py` | Runs the experiment; one log line per negotiation; resumes after a dropped connection |
| `disclosure_results.py`, `analysis.py` | The pre-registered analysis (bootstrap confidence intervals over scenarios) |
| `tests/` | 213 tests, one file per stage |
| `logs/` | Every negotiation the AI played: prompts, raw replies, outcome, and the git commit it ran on (one JSON line each) |
| `report/` | The paper (LaTeX), slides, and `figures.ipynb`, which rebuilds every figure |

**Notebooks** (in `notebooks/`, saved with their outputs so they read without running):

| Notebook | What it shows |
|---|---|
| [`one_negotiation.ipynb`](notebooks/one_negotiation.ipynb) | One trade through every condition, next to the rational client |
| [`rational_benchmark.ipynb`](notebooks/rational_benchmark.ipynb) | What the band is worth to a perfect client, how little the quote reveals, and the numbers that killed my first design |
| [`calibration.ipynb`](notebooks/calibration.ipynb) | Setting the price scale from Fed yield-curve data, checked against two other sources |
| [`experiment_log.ipynb`](notebooks/experiment_log.ipynb) | Dated record of every run and decision, with each analysis rerun below its entry |

### Running it

```bash
conda env create -f environment.yml
conda activate cosi
pytest                # 213 tests, about 10 seconds
```

Everything except the AI calls runs from a fresh clone: the tests, the notebooks, the analysis and the report figures all read the committed logs. The AI calls used NUS's internal model API, which only works on the university network. The calibration notebook also needs internet access to download the yield data.

## Limitations

The game is deliberately simple, to isolate the effect of information. There's one dealer, so no competing quotes. There's one counteroffer, not a back-and-forth. Walking away is worth nothing. Costs and limits are synthetic, and only the price scale comes from market data. I tested one model, with reasoning off. The findings describe this model in this game. They aren't a claim about every LLM or about real markets.

---

*Joey Poh · National University of Singapore · 2026*
