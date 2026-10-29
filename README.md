# stale-deal-radar

Can a system find bad pipeline news before the CRM does, and make the fix one tap for the seller?

CRM hygiene is an incentive problem. The seller pays the data-entry cost and leadership collects the forecast, so fields decay. This repo tests three ways to rank deals for a nudge, under a hard budget of **two nudges per seller per week**, on a simulated B2B sales org with a held-out final quarter. It also writes the weekly forecast note and checks every number in it against SQL.

Built with [Claude Code](https://claude.com/claude-code): Claude wrote the simulator, models, verifier and this write-up, and Claude (through the `claude` CLI) wrote the model-generated forecast notes. Two independent Codex reviews checked the code and the claims, and their findings are fixed below. Jeremiah Weise set the question and reviews the output.

## What it found

All results are on the held-out quarter, scored after the models were frozen (coefficients hash `3ef7a6ac25f4`).

1. **The standard stale-deal rule measures the seller, not the deal.**
   - Ranking by days since last logged activity scores **AUC 0.54**, barely better than chance, because in this simulation logged activity follows the seller's habits. That is an assumption built into the simulator, not a finding about real sellers.
   - Example: take the highest-dollar deal the two-nudge policy flags in week 73 whose close date is still weeks out. Its seller logged activity that week, while the buyer sent one email in two weeks and cancelled two meetings.
2. **Most bad news is already in the CRM fields.**
   - A logistic model on stage, close-date proximity, pushes, age and amount scores **AUC 0.88**. At the same budget its nudges are right **88%** of the time, against 45% for the activity rule.
   - Close-date proximity dominates. Build the plain CRM model first.
   - At the same budget the CRM model's nudges are slightly more precise than the radar's (0.879 against 0.870), even though the radar ranks better overall. Whether the radar's extra coverage is worth that, and the cost of the buyer data, is a product decision, not a modeling one.
3. **Buyer-side signals help where the CRM looks healthy.**

   | Slice (CRM fields → radar) | World 7 | World 11 | World 23 |
   |---|---|---|---|
   | Deals whose CRM close date is 3+ weeks out, AUC | 0.80 → **0.83** | 0.82 → **0.84** | 0.80 → **0.83** |
   | Deals the buyer has abandoned that the CRM still shows open, AUC | 0.78 → **0.84** | 0.80 → **0.85** | 0.76 → **0.82** |
   | Abandoned deals flagged under the same budget | 126 → **146** of 203 | 135 → **166** of 242 | 125 → **145** of 238 |

   - I chose these two slices after seeing world 7's results. Worlds 11 and 23 played no part in choosing them and show the same lift.
   - In a strict null world (buyer signals removed from every feature the radar adds), the radar's AUC equals the CRM model's on both slices in all three worlds.
4. **The saved forecast notes check out against SQL, and the checker catches planted errors.**
   - Four weeks of notes, from a template and from Claude: 15 to 17 numbers each, all matched to a fact computed in SQL.
   - Claude's notes must cite a fact id after every number. Planted errors in them: the checker caught **480 of 480** (wrong numbers, flipped directions, wrong deal ids).
   - The uncited template notes: 157 of 160 wrong numbers caught (the three misses happened to equal another fact), and 160 of 160 for both directions and deal ids.
   - The first verifier skipped long integers and minus signs and misread adjacent changes of opposite sign. A second review found that it also passed a Unicode minus, a minus separated by a space, and spelled-out amounts ("one trillion dollars"). Codex found all of these, and each now has a regression test. The checker now rejects any quantity it cannot parse instead of skipping it.
   - These are planted errors in four weeks of notes, not a measured error rate on real model output.
5. **Drafted updates beat silence, on the deals that did slip.** There were 352 close-date pushes with a radar flag in the four weeks before them (counting only flags after the deal's previous push). The drafted date is simple: the CRM close date plus the training quarters' median push of 5 weeks. Measured against the next recorded close date, it was off by **2 weeks** (median). Leaving the CRM unchanged was off by **5 weeks**.
   - This counts only flagged deals that went on to slip. It does not count the cost of suggesting a new date on flagged deals that didn't slip, or whether a seller would accept the suggestion.

An example drafted update (same deal as item 1):

> D5182 ($232K, Negotiate, CRM close week 76): 1 buyer email in two weeks, 2 meetings cancelled in four, 4 buyer stakeholders engaged; last logged activity 0 weeks ago. Suggested update: move close to week 81. [Confirm] [Edit] [Not now]

An example Claude-written note (week 73), every bracketed fact verified:

> Commit for quarter 6 [F13] fell $429K [F2] this week to $6.8M [F1], and best case dropped $1.2M [F4] to $8.5M [F3]. Won quarter to date stands at $12.3M [F5]. Also this week, 39 [F8] deals worth $1.8M [F9] were marked lost and 21 [F6] deals worth $1.0M [F7] slipped out of the quarter. The largest slip was D5044 [F14] at $201K [F15], which moved from week 73 [F16] to week 79 [F17]. The biggest risk for the call is that radar flags 22 [F10] commit deals worth $944K [F11]. Pressure-test those before accepting the $6.8M [F1] commit, because that is where the remaining commit risk sits.

## Full-label results

The label is bad news within four weeks: a close-date push or a lost mark.

| Ranker | AUC | Precision at 2 nudges/seller/week | Bad-news events flagged in the 4 weeks before |
|---|---|---|---|
| Days since logged activity | 0.54 | 0.45 | 44% |
| CRM-fields model | 0.88 | 0.88 | 57% |
| Radar | 0.90 | 0.87 | 61% |

Held-out quarter: 9,597 deal-weeks. The last column counts the 650 bad-news events whose whole four-week warning window falls inside the test weeks; events near either edge of the quarter would otherwise get fewer chances to be flagged. On the full label the radar's lift is small (+0.017, +0.011 and +0.015 AUC across worlds 7, 11 and 23), because most bad news is a close date arriving.

| Buyer-signal strength | 1.0 | 0.5 | 0.25 | 0.0 (strict null) |
|---|---|---|---|---|
| CRM-fields AUC | 0.882 | 0.890 | 0.890 | 0.894 |
| Radar AUC | 0.899 | 0.896 | 0.894 | 0.894 |

## How it works

- **`radar/simulate.py`:** 80 sellers, about 5,600 deals over six quarters.
  - Each deal has a latent health the CRM never sees.
  - Buyer emails, reply latency, meetings, cancellations and stakeholders track that health; `buyer_signal_strength` scales how strongly. Seller emails mostly track the seller's habit.
  - Sellers log wins at once and log losses and pushes late, at a seller-specific diligence.
- **`radar/detect.py`:** features, the label, three rankers, and the budget evaluation.
  - Train on weeks 13 to 60; test on weeks 65 to 73. The four-week gap keeps training labels out of the test window.
  - Model coefficients are written to `report/frozen_models.json` before any test scoring.
- **`radar/notes.py`:** loads only the CRM view (no truth columns) into SQLite and computes each week's facts in SQL. Writes a template note and, with `--llm`, a Claude note that must cite a fact id after every number.
- **`radar/verify.py`:** checks every number, deal id and direction word against the facts, rejects quantities it cannot parse (spelled-out amounts, ambiguous dashes), plus the planted-error generator.

## What this does not show

- **It is a simulation that encodes its own hypothesis:** buyer engagement carries the slip signal, and seller activity mostly doesn't. Real data has to test that first. The strict null shows the method reports no lift when the signal is absent. It cannot show that the signal exists in a real CRM.
- **The slices were chosen after the first run.** The full-label result turned out to be dominated by close dates arriving. Worlds 11 and 23 replicate the slices, but they come from the same simulator, so they can't rule out the simulator's own assumptions.
- **Abandoned deals get marked lost quickly here** (median 0 weeks after the buyer walks away), so the radar flags them only about a week before the CRM does. Real CRMs usually carry dead deals longer, which would make that slice matter more. That is a guess, not a result.
- **The verifier checks numbers, deal ids and the direction word nearest each change, not interpretation.** "That is where the remaining commit risk sits" is a judgment, and nothing checks it. It is a prototype, not a guarantee over arbitrary prose. One known gap: it lets the word "one" through, because it is usually a pronoun.
- **Confirm, Edit and Not now are printed proposals.** Nothing here connects to a CRM, and no seller has used it.

## With real CRM data, in order

1. Fit the CRM-fields model first. It may be most of the value.
2. Test whether buyer-side signals add lift beyond it, on deals the CRM shows as healthy. That is the real version of `buyer_signal_strength`.
3. Treat the nudge budget and the drafted update as the incentive design. Measure confirm, edit and dismiss rates per seller, and treat a falling confirm rate as lost trust.
4. To deprecate fields, measure which ones any report, model or forecast actually reads.
5. Measure seller hours as time spent per pipeline update, before and after.

## Reproduce

```bash
uv venv .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python run.py                 # template notes only; under a minute
.venv/bin/python run.py --llm           # also has the local `claude` CLI write notes, then verifies them
.venv/bin/python run.py --reuse-notes   # re-verifies the Claude notes already saved in report/results.json
.venv/bin/python -m pytest -q tests
```

Results land in `report/results.json`. Claude-written notes vary from run to run, so `--reuse-notes` re-checks the exact notes reported here.
