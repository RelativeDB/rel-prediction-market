# Study 4 — minute-level jumps, and whether a headline explains them

Prediction markets sometimes reprice hard in a few minutes. This study finds
those minutes, checks whether news arrived just before them, and then asks the
model to predict which headline will move which market.

```bash
python -m jumps.find_jumps --days 3 --markets 600 --reuse-news
python -m jumps.impact --floor 0.55 --min-move 0.02 --sigmas 4 --context-cells 512
```

## Finding jumps

346 active markets, 3 days of **1-minute** bars (1,288,943 of them), against
GDELT's quarter-hour headline stream. A jump is a 15-minute move of at least
2–3¢ *and* at least 4–6× that market's own median absolute move, deduplicated
with a 60-minute cooldown.

**1,174 jumps.** Some look exactly like the story you would want:

```
07-24 16:36   0.795 -> 0.590  (-0.205, typical move 0.005)
   market: Israel x Iran ceasefire continues through...?
   news  : [6 min before, cos 0.57] Iran rejects temporary ceasefire until
           Strait of Hormuz demands met
```

## The control that kills the naive version

| | small universe (83 markets) | deep universe (346 markets) |
|---|---:|---:|
| jumps found | 530 | 1,174 |
| news within 30 min **of a jump** | 70.6% | 46.8% |
| news within 30 min **of a random minute** | 71.9% | 48.6% |
| **lift** | **−1.4%** | **−1.7%** |

Matching a headline to a jump is easy and meaningless. With ~60k on-topic
headlines in the window, *something* is always within half an hour of
*anything*. The presence of news carries no information about jumps — and this
replicated at 2.2× the sample.

Which is what makes the next question the right one: not *is there news*, but
*which* news matters.

## Predicting impact

Population: (headline, market) pairs matched by embedding similarity ≥ 0.55.
Target: did the market move past the jump threshold in the next 15 minutes?
Two arms, identical but for whether the headline text is visible.

| | dev (829 events, 29 movers) | holdout (681 events, 30 movers) |
|---|---:|---:|
| always moves (p=0.5) | 0.500 | 0.500 |
| similarity to market | 0.484 | 0.564 |
| RT-J, headline text | 0.556 | 0.494 |
| RT-J, headline muted | **0.585** | 0.461 |
| **gain from the headline** | −0.030 [−0.062, −0.005] | +0.033 [−0.001, +0.072] |

The text hurts on dev, helps on holdout, and the model is at chance on the
held-out half. **This is underpowered, not negative**: 59 positives cannot
separate a real ±0.03 effect from zero.

## Why — the examples say it plainly

The single most decisive headline in the set was ranked near the *bottom*:

```
p(move)=0.41 (muted 0.41)   0.875 -> 0.915 (+0.040)
   MARKET: Will Trump meet with Netanyahu by July 31, 2026?
   NEWS  : Netanyahu to visit Washington for White House meeting with Trump
```

The price moved, the headline all but resolves the question, and both arms
returned the identical 0.41 — the model did not read it.

Meanwhile two of the three highest-ranked true positives were this:

```
p(move)=0.87   MARKET: Will the highest temperature in Wellington be 10°C…
               NEWS  : What will the weather be like in Mid Cheshire this weekend?
```

Wellington, New Zealand matched to Cheshire, England: topically similar,
causally unrelated. And the same Netanyahu story, filed by a different outlet
one minute later, was scored 0.87 on an occasion when nothing moved.

So the model is ranking on **market identity and recent activity** — "these
markets are lively" — not on headline content. Across nearly every example the
text and muted arms sit within 0.02 of each other.

## What would fix it

Not more link engineering. The failure is in the event set:

1. **Deduplicate by story, not by string.** One event republished twelve times
   is twelve chances to be wrong and one to be right.
2. **Require the headline to name the market's entities**, so "weather this
   weekend" cannot attach to a Wellington temperature market.
3. **Give the model the first occurrence**, not the twelfth — by the time a
   story has been filed a dozen times, the price has already moved.
4. **More days of news.** Three days yields 59 positives at a similarity floor
   strict enough to avoid junk matches; four more days would roughly double it,
   which is exactly the resolution the current splits are arguing about.
