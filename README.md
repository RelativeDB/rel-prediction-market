# rel-prediction-market

Can a frozen relational transformer forecast prediction markets — resolution
outcomes, short-horizon price moves, the impact of a single headline — when the
news, the order tape and two exchanges are wired together as one relational
database?

**No. The market price wins, and not narrowly.** On 1,365 held-out contracts the
crowd's own price scores 0.942 AUROC; the model scores 0.775. Everything below is
the evidence, the infrastructure that produced it, and the one exploitable
structure the work did turn up — which belongs to the market, not the model.

Built on [RelativeDB / RelQL](https://github.com/RelativeDB/RelQL) —
[relativedb.com](https://relativedb.com) · [RelQL docs](https://relql.com/docs/)

Every prediction is one query:

```sql
PREDICT markets.resolved_yes
FROM markets
WHERE markets.market_id IN :ids
RETURN PROBABILITY
```

No feature engineering, no fitted head, and outside the fine-tuning section no
gradient ever taken.

---

## The database

Two exchanges, nine tables at its widest:

```
events ──< markets ──< price_ticks
   │           │
   │           └──< market_subjects >── subjects ──< post_subjects >── comments
   │
   └──< news_mentions >── news_articles
```

| table | rows | one row is |
|---|---:|---|
| `markets` | 2,605 | a settled binary contract (Polymarket + Kalshi) |
| `price_ticks` | 230,925 | one market-hour: VWAP, dollar flow, distinct takers, built from YES-token fills |
| `news_articles` | 31,923 | a headline with named entities and tone |
| `news_mentions` | 63,823 | this headline is about this event, capped per event per week |
| `chatter` | 707,660 | a Reddit or Hacker News post |

Two invariants make it a backtest rather than a demonstration:

**Every row carries the moment it became knowable**, and the engine's temporal
bound drops the rest. Verified at the 2025-12-10 anchor: 24,154 of 25,094
articles visible, none from the future, none without a timestamp; 27,631
mentions reachable through the event→mention→article path, none from the future.

**A peer's outcome is masked until it settles.** A contract closing next week
already exists in the table this week with its result in a column — a timestamp
filter does not catch that, because the row is old and the outcome is not. The
retrievers strip `resolved_yes` from any peer unsettled at the anchor. Verified
at the same anchor: 1,440 visible outcome cells, every one settled; 139 open
contracts, zero outcomes exposed.

Splits are fixed by settlement date before any scoring: **dev** settles
2025-11-01 to 12-05, **holdout** 12-06 to 12-31.

---

## Forecasting resolution: the model loses to the price

Two to three days before each contract settles, predict the outcome. The
benchmark is the market price at that same moment — a real forecast from people
with money at stake.

| holdout, 1,365 contracts | acc | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| **market price (24h VWAP)** | 0.881 | **0.942** | **0.082** | **0.256** |
| RT-J (markets + tape + news) | 0.824 | 0.775 | 0.205 | 0.603 |
| RT-J (news ablated) | 0.815 | 0.768 | 0.199 | 0.596 |
| price × RT-J (logit mean) | 0.883 | 0.943 | 0.090 | 0.306 |
| always NO (p=0.5) | 0.235 | 0.500 | 0.250 | 0.693 |

The model beats chance comfortably — a frozen checkpoint that has never seen a
prediction market extracts real signal about which contracts resolve YES. It
extracts nothing the price does not already contain: **blending it into the price
gains +0.000 AUROC [−0.004, +0.005]**.

Dev shows a +0.012 [+0.006, +0.018] blend gain that does not survive the
holdout. Dev also shows the model beating the price on *uncertain* contracts
(0.732 vs 0.659 where the price sits between 0.2 and 0.8); holdout does not
reproduce it.

**News contributes +0.008 AUROC [−0.027, +0.041]** on holdout, +0.016
[−0.014, +0.047] on dev. Consistently signed, never distinguishable from zero.

---

## Text does not move short horizons

Three separate tests, each built to give language its best chance:

**Chatter → 6-hour price direction.** 707,660 Reddit and Hacker News posts,
attached to markets two ways — shared distinctive vocabulary, then MiniLM
similarity against the market question with a shared-subject bridge and hourly
rollups. Muting the post text changes nothing: **+0.001 [−0.008, +0.009]** on
holdout with the better graph. Restricting to posts whose embedding matches the
market unambiguously (cosine ≥ 0.5) gives +0.010 on dev and −0.005 on holdout.

**Headline → 15-minute impact.** 1,510 (headline, market) events. Muting the
headline: **−0.030 [−0.062, −0.005]** on dev, +0.033 [−0.001, +0.072] on
holdout. The model is at chance on the held-out half.

**Presence of news → jumps.** 1,174 jumps found across 346 markets and 1.29M
minute bars, defined as a 15-minute move ≥2–3¢ and ≥4–6× that market's typical
move. A matching headline appeared within 30 minutes of **46.8%** of jumps — and
of **48.6%** of randomly chosen minutes. **Lift −1.7%**, replicated at two
sample sizes.

With ~60k on-topic headlines in a window, something is always within half an
hour of anything. Individual cases look compelling — the Israel/Iran ceasefire
market falling 0.795 → 0.590 six minutes after *"Iran rejects temporary
ceasefire until Strait of Hormuz demands met"* — and the control says that
appearance carries no information.

---

## Strike ladders: an architectural limit

A daily strike ladder ("will WTI settle above $K?") is a CDF over one number.
Reconstructing the same question from 503 sessions of WTI history gives 14,160
labelled rows — 40× the data any other table here has.

| 1,200 strikes over 40 held-out days | AUROC | Brier |
|---|---:|---:|
| **martingale** (normal at prior close, σ = trailing realized vol) | **0.917** | **0.1070** |
| xgboost (12,960 training rows, 9 features) | 0.917 | 0.1123 |
| RT-J | 0.636 | 0.2317 |
| constant at base rate | — | 0.2252 |

**RT-J scores worse than a constant.** The diagnosis is exact:

| strike distance from prior close | n | RT-J mean | realized |
|---|---:|---:|---:|
| −6% to −4% | 120 | 0.457 | 0.958 |
| −2% to 0% | 120 | 0.459 | 0.583 |
| +2% to +4% | 120 | 0.448 | 0.133 |
| +7% to +12% | 280 | 0.439 | 0.014 |

Realized outcomes sweep 94 points; the model's output moves 1.8. Its full range
across 1,200 predictions is 0.342–0.637, and its correlation with strike
distance is −0.256 against the outcome's −0.691.

This is not a calibration problem. Isotonic regression fitted *in sample* — the
ceiling of any post-hoc tuning — moves Brier only 0.2317 → 0.2130 and AUROC
0.636 → 0.644, because a monotone map cannot reorder. Nor is it a data problem:
XGBoost on identical rows reaches 0.917, so the information is present and
learnable. The task is a threshold comparison on a continuous variable, and the
model's numeric channel does not recover a sharp decision boundary from it.

XGBoost tying the closed form is its own result. 82% of its gain sits in the two
features that *are* the martingale (distance ÷ σ, and distance in percent);
momentum, realized vol, the Brent spread, the RBOB crack and weekday together
add ~18% of gain and **zero** AUROC. Over this period WTI has no exploitable
drift beyond the martingale.

---

## Fine-tuning: a small general lift, no domain effect

Trained on 541 Kalshi contracts about Trump, war and oil settling before
2025-12-06; tested on 188 later contracts in those categories, with 400
out-of-domain contracts as a control.

| | in-domain (188) | control (400) |
|---|---:|---:|
| market price | 0.866 | 0.875 |
| zero-shot | 0.750 | 0.718 |
| fine-tuned, lr 1e-5 | 0.819 → +0.069 [+0.002, +0.142] | 0.700 → −0.018 [−0.077, +0.042] |
| fine-tuned, lr 1e-6 | 0.783 → +0.033 [−0.009, +0.076] | 0.766 → **+0.048 [+0.019, +0.079]** |

The learning rates disagree about which arm improves, which is what 188
contracts and 86 positives buy. The domain hypothesis is unsupported; a small
general improvement in resolution forecasting survives at the lower rate, and
still loses to the price by 8 points.

Training loss rose in both runs (0.902 → 1.40 and → 1.31). The leading
explanation is per-batch label normalization: with `batch_size=4` on a binary
target, an all-YES batch has near-zero label variance and the normalized target
diverges. `--ft-batch` defaults to 16 for that reason; the explanation is
untested, so those losses are not evidence about model quality.

---

## What is exploitable: the market's own bias

The one durable edge found here is not the model's. It is favourite–longshot
bias, measured on 1,170 settled gas-ladder strikes with their closing prices.

| closing price | n | realized YES | edge vs price |
|---|---:|---:|---:|
| 0.00–0.02 | 235 | 0.0000 | −0.010 |
| 0.02–0.05 | 167 | 0.0000 | −0.027 |
| 0.05–0.10 | 56 | 0.0000 | −0.065 |
| 0.10–0.20 | 62 | 0.0323 | −0.103 |
| 0.90–0.94 | 30 | 1.0000 | +0.085 |
| 0.96–0.975 | 65 | 1.0000 | +0.034 |
| 0.975–0.99 | 98 | 1.0000 | +0.020 |

**458 contracts priced under 10¢: two resolved YES. 163 priced 0.96–0.99: all
163 resolved YES**, a 95% lower bound of 0.9818 on the true rate against a
fee-inclusive breakeven of 0.9720.

Three constraints keep this from being a strategy:

1. Those are *closing* prices — the final trades land within an hour of
   settlement, when most uncertainty has resolved. An entry hours earlier faces
   strictly more risk than this sample measures.
2. 163 contracts came from 68 days. A whole ladder settles on one number, so a
   day is one bet, not thirty.
3. At 97¢ the payoff is 32:1 against. At a true 99% rate that is ~2.5 losses per
   250 days, each erasing ~32 wins — the return is dominated by a tail that 68
   days cannot measure.

For pricing the ladders themselves, the closed-form martingale beat everything
tried, is better calibrated than XGBoost, and needs no training step.

---

## Data

~21 GB pulled, all public.

| source | what came out of it |
|---|---|
| [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) | 1.8M markets; 115M fills scanned from a 1.03B-row, 37.5 GB tape |
| [TrevorJS/kalshi-trades](https://huggingface.co/datasets/TrevorJS/kalshi-trades) | Kalshi markets and fills, 2023-03 to 2026-01 |
| [GDELT 2.0 GKG](http://data.gdeltproject.org/gdeltv2/) | 2.17M headlines with entities and tone, Aug–Dec 2025 |
| [mitanshugoel/reddit-2025](https://huggingface.co/datasets/mitanshugoel/reddit-2025) | 120k topical comments filtered from 21.5M scanned |
| [open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) | 587k posts and comments |
| Kalshi API | 69 settled gas ladders, 1,170 closing prices, 1-minute market bars |
| Yahoo chart API | 503 sessions of WTI, Brent and RBOB settles |
| [pmxt](https://pmxt.dev) | live markets and candles across venues |

Two fetch techniques worth reusing:

**Row-group slicing.** The Polymarket fills file is time-sorted and every parquet
row group carries min/max statistics, so a window is selected from the footer —
113 of 1,027 row groups, 4.5 GB on the wire instead of 37.5 GB.

**Range-request prefixes.** A month of Reddit comments is 37 GB of zstd, written
in timestamp order, and the Hub honours HTTP range requests — a 2.5 GB prefix
decodes to a contiguous slice from the first of the month.

GDELT's API throttles an unauthenticated caller into `429`s within a few
requests; its file archive does not.

---

## Running it

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# resolution at scale, two exchanges
python -m scale.pull_polymarket --start 2025-10-15 --end 2026-01-01
python -m scale.pull_kalshi     --min-volume 25000
python -m scale.pull_gdelt      --start 2025-10-25 --end 2026-01-01 --every 2
python -m scale.semantic        --top-k 40 --min-similarity 0.42
python -m scale.resolve --split dev --limit 0 --arms news,nonews
python -m scale.analyze --split holdout

# chatter over 6 hours
python -m comments.pull_chatter --month 2025-11 --reddit-gb 2.5
python -m comments.rich_task    --horizon 6 --per-split 900

# minute-level jumps and headline impact
python -m jumps.find_jumps --days 3 --markets 600 --reuse-news
python -m jumps.impact     --floor 0.55 --context-cells 512

# domain fine-tuning
python -m finetune.domain --train-cap 700 --epochs 2 --ft-batch 16

# strike ladders: gas (real ladders) and WTI (synthesized from futures)
python gas/pull.py --days 70 && python gas/model.py
python wti/dataset.py && python wti/predict.py && python wti/boost.py
```

**Python ≥ 3.10 on macOS arm64**, where the `relativedb` wheel bundles the
native engine; fine-tuning additionally needs Apple MPS. Elsewhere, build
`librt_c` from the [RelQL repo](https://github.com/RelativeDB/RelQL) and point
`RELATIVEDB_RT_LIB` at it. First run downloads the RT-J checkpoints (~350 MB)
and the MiniLM encoder.

Every expensive step is cached and resumable: predictions per arm and universe,
embeddings per text hash, scoring in checkpointed chunks.

Code and results only — no data is committed. Everything is rebuilt by the
commands above.

```
scale/       two-exchange resolution study
comments/    chatter study (Reddit + Hacker News)
jumps/       minute-level jump detection and headline impact
finetune/    domain fine-tuning
gas/         AAA gas-price ladder: pull, fit, walk-forward backtest
wti/         WTI ladder: synthesized dataset, RT-J, XGBoost
fetch.py db.py predict.py   the small live example
```
