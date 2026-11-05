# Watch-time recommender on industrial impression logs

A two-stage recommender for a streaming home screen, built on
[ContentWise Impressions](https://github.com/ContentWise/contentwise-impressions) (CW10M):
real interaction logs from a streaming service, including the recommendation lists users were
shown, the slot each title sat in, and how far into each title they watched (`vision_factor`).

The metric, the layout and the architecture were decided before any model code. The design notes
are on the [`design-notes`](../../tree/design-notes) branch.

## Problem

Optimize expected watching per home-screen list, not clicks: a click the user abandons after a
minute is not a win. A secondary metric tracks discovery, meaning watching of series that are new to
the user.

## Pipeline

```
history ──► candidates (100/user) ──► LightGBM LambdaRank ──► feed: continue-watching + explore
            30 from the user's history                        sections, explore share ρ
            70 from EASE (new to the user)
```

- **Labels:** watch fraction per user-series, rolled up from episodes (`dataprep/folds.py`),
  graded into 0–4 for LambdaRank.
- **Features (13):** recency (days since last watch), user's watch depth and episode count,
  popularity over 1 week / 2 weeks / all time, trend, title age, film vs series, EASE score.
- **Baselines:** recent popularity, continue-watching by recency, ItemKNN, RP3β, EASE.

## Evaluation

- **Position-weighted value:** each list is scored as Σₖ θₖ · Wₖ, where θₖ is the examination
  weight of slot k, estimated from the logged impressions as click rate per slot ÷ exposures,
  normalized to slot 0 (`dataprep/loaders.py::load_theta`). `norm_value` divides by the best
  possible list for that user.
- **Discovery:** share of that value coming from series new to the user.
- **No look-ahead:** weekly rolling folds (train on week t−2, early-stop on t−1, test on t). Every
  feature is computed from data before the cutoff, and `check_no_lookahead` asserts that no
  training timestamp crosses it (`dataprep/splits.py`).

## Results (held-out week 13, 12,000 users, top-20 list)

| model | value | norm. value | discovery |
|---|---|---|---|
| continue-watching by recency | 1.108 | 0.409 | 2.5% |
| LightGBM ranker, popularity features only | 0.523 | 0.187 | 20.7% |
| **LightGBM ranker, all features** | **1.243** | **0.452** | **10.7%** |
| + two-tower as a second candidate source | 1.243 | 0.452 | 11.1% |

- The full ranker beats the recency baseline by **12%** on position-weighted value and quadruples
  discovery.
- Popularity alone halves value: personalization carries the result.
- Recency (`days_since`) carries about half of the ranker's gain importance.
- Adding a two-tower retrieval model as a second candidate source changed nothing: candidate
  recall stayed at 64.5%.

This is one test week. More weeks are needed before calling the gap stable.

## Layout

```
dataprep/    loading, weekly folds, ranking splits, leakage check
models/      baselines, ItemKNN, RP3β, EASE, two-tower, LightGBM ranker
evaluation/  NDCG, position-weighted scorer, feed construction
scripts/     experiment entry points (results land in results/)
notebooks/   data exploration
```

## Running

```
pip install -r requirements.txt
# download ContentWise Impressions into data/ (see the dataset repo)
python scripts/run_model1.py 13
```

The dataset is licensed CC BY-NC-SA 4.0 and is not included here.
