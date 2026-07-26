# rel-prediction-market

Four experiments asking whether a **frozen relational transformer** can
forecast prediction markets — resolution outcomes, short-horizon price moves,
and the impact of individual headlines — with the news, the order tape and two
exchanges wired together as one relational database.

Built on [RelativeDB / RelQL](https://github.com/RelativeDB/RelQL) —
[relativedb.com](https://relativedb.com) · [RelQL docs](https://relql.com/docs/)

Every prediction here is one query. This is the whole interface:

```sql
PREDICT markets.resolved_yes
FROM markets
WHERE markets.market_id IN :ids
RETURN PROBABILITY
```

No feature engineering, no head fitting, and (except in study 4) no gradient
ever taken. The tables are raw records, the links say how they relate, and
RT-J discovers the subgraph it wants.

---

## What was found

| # | question | answer |
|---|---|---|
| 1 | Can it call a 6-hour price move on 45 live markets? | Inconclusive — n too small |
| 2 | Can it beat the market's own price at resolution, 2–3 days out? | **No** (holdout: −0.007 AUROC [−0.019, +0.005]) |
| 2b | Do news tables help the model itself? | **Small and consistently positive, not established** (+0.030 [−0.009, +0.068] held out) |
| 3 | Does Reddit/HN chatter predict the next 6 hours? | **No** — on either a lexical or a semantic graph |
| 4 | Does a jump have news behind it? | **No** — 46.8% of jumps vs 48.6% of random minutes |
| 4b | Can it predict which headline moves a market? | Underpowered — 59 positives can't separate ±0.03 |
| 5 | Does fine-tuning on Trump/war/oil help future markets in those categories? | *pending — code in `finetune/`, run not yet complete* |

Every claim here is negative or underpowered. The closest thing to a positive
is 2b — deleting the news tables costs the model accuracy in both splits, by
+0.023 to +0.030 AUROC — but neither interval clears zero, so it is a
consistent direction rather than an established effect. An earlier version of
this study reported +0.052 with an interval that did clear zero; that number
came from a news table whose coverage collapsed in the test window, and it did
not survive the fix (see [study 2](studies/02-resolution-at-scale.md)).

Full numbers, caveats and prediction-level examples live in [`studies/`](studies/).

---

## The database

Nine tables at its widest, across two exchanges:

```
events ──< markets ──< price_ticks
   │           │
   │           └──< market_subjects >── subjects ──< post_subjects >── comments
   │
   └──< news_mentions >── news_articles
```

| table | rows | one row is |
|---|---:|---|
| `markets` | 2,671 | a settled binary contract (Polymarket + Kalshi) |
| `price_ticks` | 262,563 | one market-hour: VWAP, dollar flow, distinct takers |
| `news_articles` | 25,094 | a headline with entities and tone |
| `news_mentions` | 56,505 | this headline is about this event (18k lexical, 38k semantic) |
| `chatter` | 707,660 | a Reddit or Hacker News post |

### Two rules that make it a backtest

**Everything is stamped.** Every row carries the moment it became knowable;
the engine's temporal bound drops the rest.

**Peer outcomes are masked until they settle.** A contract closing next week
already exists in the table this week with its result in a column — a
timestamp filter does not catch that, because the row is old and the outcome is
not. The retrievers strip `resolved_yes` from any peer that had not settled at
the anchor. Verified directly: at the 2025-12-10 anchor, 1,440 visible outcome
cells, every one settled; 139 open contracts, zero outcomes exposed.

---

## Data

~19 GB pulled, all public, none scraped.

| source | what came out of it |
|---|---|
| [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) | 1.8M markets; 95M fills from a 1.03B-row, 37.5 GB tape |
| [TrevorJS/kalshi-trades](https://huggingface.co/datasets/TrevorJS/kalshi-trades) | Kalshi markets and fills — a second exchange, same labels |
| [GDELT 2.0 GKG](http://data.gdeltproject.org/gdeltv2/) | 933k headlines with entities and tone |
| [mitanshugoel/reddit-2025](https://huggingface.co/datasets/mitanshugoel/reddit-2025) | 120k topical comments filtered from 21.5M |
| [open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) | 587k posts and comments |
| [pmxt](https://pmxt.dev) | live markets, 1-minute candles |

Two tricks worth stealing:

- **Row-group slicing.** The Polymarket fills file is sorted by time and every
  parquet row group carries min/max statistics, so a two-month window is 92 of
  1,027 row groups — **3.7 GB on the wire instead of 37.5 GB**.
- **Range-request prefixes.** A month of Reddit is 37 GB of zstd, but the dump
  is written in timestamp order and the Hub honours HTTP range requests, so a
  2.5 GB prefix decodes to a contiguous slice from the 1st of the month.

GDELT's *API* throttles an unauthenticated caller into `429`s within a few
requests; its *file archive* does not.

---

## Running it

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# study 1 — live markets, 6-hour direction
./run.sh

# study 2 — resolution at scale (two exchanges, 2,671 contracts)
python -m scale.pull_polymarket --start 2025-10-15 --end 2026-01-01
python -m scale.pull_kalshi     --min-volume 25000
python -m scale.pull_gdelt      --start 2025-10-25 --end 2026-01-01 --every 2
python -m scale.semantic        --top-k 40 --min-similarity 0.42
python -m scale.resolve --split dev --limit 0 --arms news,nonews
python -m scale.analyze --split holdout

# study 3 — does chatter move markets?
python -m comments.pull_chatter --month 2025-11 --reddit-gb 2.5
python -m comments.rich_task    --horizon 6 --per-split 900

# study 4 — minute-level jumps and headline impact
python -m jumps.find_jumps --days 3 --markets 600 --reuse-news
python -m jumps.impact     --floor 0.55 --context-cells 512

# study 5 — fine-tune on Trump / war / oil, test on later markets
python -m finetune.domain --train-cap 700 --epochs 2
```

Requirements: **Python ≥ 3.10** on **macOS arm64**, where the `relativedb`
wheel bundles the native engine (fine-tuning additionally needs Apple MPS).
Elsewhere, build `librt_c` from the [RelQL repo](https://github.com/RelativeDB/RelQL)
and point `RELATIVEDB_RT_LIB` at it. First run downloads the RT-J checkpoints
(~350 MB) and the MiniLM encoder.

Every expensive step is cached and checkpointed: predictions are written per
arm, embeddings per text hash, scoring in resumable chunks. Re-analysis costs
seconds, and a run that dies at 80% keeps its 80%.

---

## What ships here

Code and write-ups only — **no data**. The pulls, the derived tables, the
embedding cache and the scored predictions all stay local; every one of them
is rebuilt by the commands above, and every number they produced is in
[`studies/`](studies/).

```
studies/          the four write-ups, with numbers and caveats
scale/            two-exchange resolution study
comments/         chatter study (Reddit + Hacker News)
jumps/            minute-level jump detection and headline impact
finetune/         domain fine-tuning
fetch.py db.py predict.py   study 1, the small live example
```

---

## A note on the results

Most of what is written up here is negative, and the negative parts are the
load-bearing ones. A dev-split result of **+0.024 AUROC over the market price**
looked strong, had a plausible mechanism, and evaporated on a holdout that was
fixed before any scoring — the apparent edge traced to one stratum where the
baseline itself was at chance. Without that split it would have shipped as a
finding.

Two data defects are documented rather than quietly fixed, because they change
how the numbers should be read: the Polymarket price series pools both outcome
tokens (so a "price" can flip between p and 1−p), and thread grouping in study
3 silently did nothing. Both are called out where they affect a conclusion.

---

## What it looks like up close

Aggregate metrics hide what a model is actually doing. These are real
predictions from the runs above.

### When it disagreed with the crowd and won

Resolution, 2–3 days before settlement (study 2, holdout):

| crowd's price | RT-J | outcome | market |
|---:|---:|---|---|
| 0.92 | 0.33 | **NO** | ECB rate cut in 2025? |
| 0.96 | 0.38 | **NO** | Xi Jinping out in 2025? |
| 0.73 | 0.21 | **NO** | Will the US officially declare war on Iran in 2025? |
| 0.17 | 0.72 | **YES** | Will Trump say "TikTok" before Dec 8, 2025? |
| 0.29 | 0.80 | **YES** | Will Trump say "Argentina" before Dec 8, 2025? |

And where the crowd was right and the model was badly wrong:

| crowd's price | RT-J | outcome | market |
|---:|---:|---|---|
| 0.07 | 0.96 | NO | Who will be the next Head Coach of the Michigan Football Team? |
| 0.02 | 0.85 | NO | #2 US Netflix Show on Dec 29, 2025? |
| 0.86 | 0.22 | YES | Epstein blackmail evidence released in 2025? |

Across 1,392 contracts the crowd won overall — which is the finding.

### A news-driven jump, exactly as advertised

```
07-24 16:36   0.795 -> 0.590   (-0.205, this market's typical 15-min move: 0.005)
   market: Israel x Iran ceasefire continues through...?
   news  : [6 minutes earlier, cos 0.57]
           "Iran rejects temporary ceasefire until Strait of Hormuz demands met"
```

These exist and are easy to find — 1,174 of them in three days. What does *not*
work is using the presence of news to find them: 46.8% of jumps have a matching
headline within 30 minutes, and so do 48.6% of randomly chosen minutes.

### The headline that mattered most, ranked near the bottom

```
p(move)=0.41  (muted 0.41)   0.875 -> 0.915  (+0.040)
   MARKET: Will Trump meet with Netanyahu by July 31, 2026?
   NEWS  : "Netanyahu to visit Washington for White House meeting with Trump"
```

The price moved, the headline nearly resolves the question, and the arm that
could see the text returned the same number as the arm that could not. Fifteen
minutes earlier the *same story* from a different outlet scored 0.87 — on an
occasion when nothing moved at all.

### Meanwhile, at the top of the confidence list

```
p(move)=0.87   MARKET: Will the highest temperature in Wellington be 10°C on July 26?
               NEWS  : "What will the weather be like in Mid Cheshire this weekend?"
```

Wellington, New Zealand; Cheshire, England. Topically similar, causally
unrelated — and two of the three highest-ranked true positives were this same
pairing. The model looked right for reasons that were pure coincidence.

### Matching failures worth laughing at

From the chatter study, where posts were attached to markets by shared
distinctive words:

| market | post that matched it |
|---|---|
| Will Jesus Christ return in 2025? | r/wallstreetbets: *"…jesus christ can he shut up for one weekend?"* |
| NYSE marketwide circuit breaker in 2025? | *"Transformers don't 'trip'. Circuit breakers do."* |
| Will ChatGPT reach 1b monthly active users in 2025? | *"AOL had an amazing warcraft 2 community. There was an online games service in the 90s called Engage…"* |

An expletive, an electrical component, and 1990s dial-up nostalgia. Embedding
similarity fixed some of this — it found real paraphrase matches like
*"Predator: Badlands Rotten Tomatoes score?"* ← *"Predator: Badlands Debuts With
86% Score"* (cos 0.91) — but the ablation says the text still contributed
nothing to a six-hour price move.
