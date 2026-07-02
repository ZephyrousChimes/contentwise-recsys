# Methodology: two-stage recommender on ContentWise Impressions

Research done 2026-09-27, before any code. Every step below is tied to (a) a source that did it
this way and why, or (b) a fact about this dataset that forces it. Sources are listed at the end
as [S#].

---

## 0. What serious practitioners actually do, in one page

Read across the industrial papers and the evaluation-methodology literature, the same pattern
keeps coming up:

1. **Design the problem before solving it.** Booking.com calls this *Problem Construction* and
   says the best setup "is often not the one that comes to mind immediately" [S7]. YouTube says the
   choice of surrogate problem "has an outsized importance ... but is very difficult to measure
   offline" [S1]. Hidasi & Czapp: "step zero is task definition, since the offline evaluation is
   designed around the task, not vice versa" [S10].
2. **Reproduce the thing you are replacing before you improve it.** YouTube's depth-0 network
   "performed very similarly to the predecessor system" (matrix factorization) before any depth or
   features were added [S1]. Airbnb's first NN that shipped was one hidden layer, "booking neutral
   against the GBDT". Its value was proving that the pipeline was correct, not any gain [S3].
3. **Users lead, model follows.** Airbnb abandoned the "{download paper → implement → A/B test}"
   loop after deeper nets, residuals, batch norm, wide&deep and attention all came back neutral.
   The replacement rule: "first quantify a user problem. Model tweaks come afterwards, and in
   response to the user problem" [S4]. Every one of their wins (two-tower, cold-start estimator,
   position dropout) started from a **segment where the metric was measurably worse**.
4. **When there is no clear user problem, they say so, and narrow the search on purpose.** Airbnb
   on position bias: "there was no strong reason to believe positional bias was the sole culprit
   ... we were shooting in the dark. But ... we decided to go after the biggest gaps in modeling
   that explained the observations. And a literature survey was crucial in identifying where major
   gaps might be lurking" [S4]. Booking.com: build a minimal model to test the hypothesis. If the
   result is inconclusive, adjust so the hypothesis gets "a fair chance". If it is negative, reject
   the concept [S7].
5. **Complexify one thing at a time, and test the data for the property a method assumes.**
   Karpathy: "don't be a hero", "complexify only one at a time" [S13]. Hidasi & Czapp ran the
   *same* model with and without sequence modelling to check whether the data had any sequential
   signal at all. On rating datasets it largely didn't [S10].
6. **Baselines are tuned as hard as the model, or the comparison is void.** Only 7 of 18
   top-venue neural recommenders were reproducible, and 6 of those 7 lost to kNN/graph heuristics
   [S14]. A carefully set-up vanilla MF beat five years of published ML-10M results [S15]. With
   untuned baselines "every method ... can be reported to be outperforming the state-of-the-art"
   [S17].
7. **Offline metrics are a health check, not the verdict.** At Booking.com, offline gain against
   business gain across 23 model pairs: Pearson −0.1, CI (−0.45, 0.27) [S7]. YouTube: "live A/B
   results are not always correlated with offline experiments" [S1]. We have no A/B test, so we
   claim only what offline data can support (§6).

The rest of this document applies those seven rules to ContentWise.

---

## 1. Facts about the data that constrain the design (from the dataset paper [S20])

| Fact | Consequence |
|---|---|
| **Impressions list `series_id`, not `item_id`.** Rows show series; the 145k items are mostly episodes (85% of items, 87% of interactions). | **The recommendation unit is the series (28,881).** Recommending episode 7 vs episode 8 isn't a real product decision. This also makes EASE feasible (below). |
| **Impressions *without* a direct link (23.3M rows) have no timestamp**, only `user_id`. | They can't be placed on the global timeline. Any use of them in a temporal protocol leaks the future. The reference code folds them into one timeless user×item matrix [S21]. That is part of why published "impression" results aren't temporal. **Excluded from the main protocol**; at most one clearly labelled leaky ablation. |
| **Impressions *with* a direct link (307k slates)** join to interactions via `recommendation_id`. | Show time can be estimated as the earliest linked interaction timestamp, so these slates *can* go on the timeline. This is where "shown-and-skipped" negatives come from. |
| Average user: **248 interactions, only 22 via a logged impression**. Items: 72 vs 25. | ~90% of viewing comes from search, browse or OTT-added rows that aren't logged. Retrieval must learn from *all* views (YouTube did the same: "from all YouTube watches ... rather than just watches on the recommendations we produce" [S1]). The reranker only has impression-linked data, which is small, so it has to be a GBDT and not a deep net (Airbnb: "confined to smaller scales, models like GBDT are arguably at par" [S3]). |
| Each row is produced by **one algorithm; different rows by different algorithms**, some non-personalized. Position is 2-D: `row_position` × index in list. | Position bias is two-dimensional [S22]. The fact that several rankers produced the rows is exactly what intervention harvesting needs to estimate position bias without randomization [S19]. |
| A view writes a **start row and an end row** with `vision_factor` ∈ [0,1]. Some users with broken vision factors were removed. | Views must be deduplicated to one record per viewing, keeping the final `vision_factor`. |
| Accounts are **shared households**. | Multi-interest users. Park it as a hypothesis; don't design for it until data shows it (§4). |
| Dates shifted to Jan 7 – Apr 15 2019 (day-of-week preserved); **some days have no data**. | Weekly windows stay aligned to weekdays. Missing days must be listed, and any fold containing them flagged. |
| Long tails: 4.05% of series carry 80% of interactions. | Tail series have few interactions to learn an ID embedding from. That is the condition under which Airbnb's listing-ID embeddings overfit [S3]. Measure the train/test gap by popularity bucket before relying on ID embeddings. |

**Machine budget (checked):** 14 GB RAM (~4 GB free at the time), 16 cores, **5.1 GB free disk
(96% full)**. Full-catalog series-level EASE needs a 28.9k² float32 Gram (3.3 GB) plus a copy to
invert, which is too tight. Restricting to series with ≥ k training interactions is motivated
anyway (tail series carry no CF signal), and at 10–15k series it drops to 0.4–0.9 GB. **Disk must
be freed before downloading the data.** LightGBM isn't installed yet.

---

## 2. Problem construction (decided before any model)

### 2.1 The prediction task
At a cutoff time T (start of week w+1), for each user with history before T, **rank all series
available as of T**. Ground truth: the series the user positively views during [T, T+7d).

### 2.2 Repeat vs. explore: split, never pool
Continuing a series you are already watching dominates TV-episode data and is nearly trivial to
predict. Li et al. show that repeat samples "often act as easy shortcuts". Pooled accuracy with a
significance test "does not fully represent a model's recommendation accuracy". Neural models
"suffer from an inherent repetitive bias" [S18]. YouTube also notes that episodic series are
"watched sequentially", which makes co-watch asymmetric [S1]. So:

- **Explore task (headline):** ground truth = test-week series the user never touched before T.
  Previously-seen series are filtered out of the ranking. This is the discovery product.
- **Repeat task (reported separately):** ground truth = test-week series already in the user's
  history. Baseline: "continue watching", i.e. most recently watched first.
- **Check before committing:** measure the repeat share of test-week positives. If explore
  positives are too sparse per user-week to evaluate, widen the test window to 14 days and record
  why.

### 2.3 The positive label
One `vision_factor` threshold, chosen from its **distribution** (look for bimodality: sampled vs
committed) and fixed before modelling. The positive unit is series-level: the user reached the
threshold on ≥1 item of the series in the window. `access`/`detail` interactions aren't positives.
They're a candidate *feature* ("looked but didn't watch").

Weighting positives by watch amount (YouTube's weighted logistic regression [S1]) is a later
hypothesis, not a default. Airbnb's cautionary result: a long-view auxiliary task raised long views
"by a large margin. But bookings remained neutral" [S3]. Booking.com calls this proxy
over-optimization [S7].

### 2.4 Two evaluation modes, because the stages answer different questions
Hidasi & Czapp distinguish *behavior prediction* (predict everything the user does, organic
included) from *interaction prediction* (predict responses to what the recommender showed; closed
feedback loop) [S10].

- **Retrieval and the full pipeline → behavior prediction**: full-catalog ranking, Recall@K /
  NDCG@K on the test week.
- **Reranker → both**: (a) pipeline NDCG@20 over retrieved candidates (behavior); (b) ranking
  quality *inside logged slates* (interaction). (b) is biased by the production ranker's choices
  (Zinkevich #35, #36 [S6]), so it counts only as supporting evidence.

---

## 3. Evaluation protocol (fixed before any model is trained)

1. **Global timeline, rolling weekly folds.** Train on [start, T_w), test on [T_w, T_w + 7d),
   advance a week, repeat. Zinkevich #33: "measure performance of a model on the data gathered
   after the data you trained the model on" [S6]. Ignoring the global timeline leaks future data
   and makes model orderings "unpredictable" [S11]. Only 4 of 88 surveyed papers used sliding
   windows [S12]. YouTube's logQ paper evaluates the same way: train sequentially, average offline
   results over **7 consecutive days** "to account for a weekly pattern" [S2].
2. **Nested in time: tune on early folds, report on later folds.** About 14 weeks exist. Draft:
   weeks 1–5 burn-in, weeks 6–8 validation (all tuning happens here), weeks 9–14 test (touched
   once per final config). Hyperparameters are re-tuned if the training window size changes
   (Hidasi & Czapp tuned "separately for each training window size and algorithm" [S10]).
3. **Full-catalog ranking. No sampled negatives in evaluation.** Sampled metrics "do not persist
   relative statements ... not even in expectation" and collapse toward AUC [S16]. Hidasi & Czapp
   show sampling can flip model order [S10].
4. **Candidate catalog = series seen before T.** Report the share of test positives that are
   cold-start series. CF can't recommend them, and hiding this inflates everyone equally.
5. **Metrics:** NDCG@20, Recall@20 (headline); Recall@200 for retrieval (candidate-pool ceiling);
   catalog coverage and popularity-of-recommended as sanity checks for popularity collapse.
6. **Statistics:** per-user paired differences between two systems in the same fold. Paired
   t-test or bootstrap CI. **Not** sign/Wilcoxon, whose false-positive rates grow with the large
   n of recsys evaluation [S23]. Report effect size with p. When one table compares many systems,
   control FDR (Benjamini–Hochberg) [S23]. Neural models: ≥3 seeds, and report the spread.
7. **Segment table for every run** (this is the engine behind §4): metric by user activity
   quintile, series popularity bucket, repeat/explore, item type (film vs series), and cold-start
   flag.
8. **Our numbers aren't comparable to published ones.** The dataset paper's random 70/10/20 split
   is leaky. Later papers use leave-last-out with no global timeline and pool repeat and explore.
   We state this and don't chase their figures.

---

## 4. How every modelling step gets motivated

Each experiment is a card in `experiments/NNN-name.md`, written **before** running:

```
Observation   – a measured number from a segment table or data check (not a paper)
Hypothesis    – what is causing it
Change        – the smallest model/data change that tests the hypothesis
Prediction    – which metric/segment should move, and roughly how much
Kill rule     – what result rejects the hypothesis
Result        – filled after; includes seeds, CI
Decision      – keep / reject / "not a fair test yet" (what to fix, one retry max)
```

**When there's no observation to motivate the next step.** This is the "lack of motivation" case,
and it's handled by a fixed procedure, not by grabbing a paper:

1. **Segment scan first.** Find the segment with the largest gap to overall NDCG, or to the best
   baseline *on that segment* (Airbnb found cold start via a −6% new-listing gap [S4]).
2. **Gap audit against successful systems.** List what YouTube/Airbnb/logQ systems do that ours
   doesn't. Rank by how well each gap explains the observed segment deficits (Airbnb's fallback
   [S4]).
3. **Property test before method.** Before adopting a method, test that the data has the property
   it exploits. Examples: shuffle history order to see if sequence matters [S10]; check that a
   series actually appears at varied positions before modelling position bias (Airbnb needed this
   variation [S4]); check the train/test gap by popularity bucket before trusting ID embeddings
   [S3].
4. **Bounded minimal test.** Smallest version, fixed budget, one retry if the test was unfair,
   then reject. Negative results are logged, not deleted.
5. **Stop rule.** When three consecutive well-posed hypotheses are rejected, the model is at its
   plateau for the available information. Zinkevich #41: look for "qualitatively new sources of
   information" rather than more architecture [S6]. Write that up as the result.

---

## 5. Build order (each rung must beat or match the one before, or we find out why)

### Phase A: data audit (each check feeds a named decision above)
- Interaction dedup (start/end view rows) → one viewing per row. Integrity tests like the ones the
  dataset ships [S20].
- `vision_factor` distribution → positive threshold (§2.3).
- Repeat share of weekly positives → explore/repeat split feasibility (§2.2).
- Missing days, weekly volume → fold boundaries (§3.1).
- `recommendation_id` → linked interaction time span. Check that slates can be dated; if one id
  spans days, stop and rethink.
- Per-series position variance across slates/rows → whether position debiasing is identifiable.
- Series interaction counts → EASE catalog cut `k`, ID-embedding risk.
- Airbnb's smoothness check: plot every engineered feature; spikes are usually bugs [S3].

### Phase B: evaluation harness + dumb baselines (Karpathy step 2; Zinkevich #4–5)
Harness tested independently of any model (Zinkevich #5): a unit test with a hand-built toy log
where the right metric values are known; an "oracle" recommender must score 1.0; a
random recommender scores ≈ K/|catalog|.
- **B0 recency-windowed popularity**, window tuned. Sun shows that defining popularity by
  recency, the way real charts do, raises its accuracy by at least 70% on MovieLens compared with the usual all-training-data popularity
  [S12]. A weak popularity baseline is the most common way results get inflated.
- **B1 continue-watching** (repeat task only).

### Phase C: strong, tuned classical baselines (equal tuning budget)
- **ItemKNN (cosine), RP3β**: the best published honest results on this dataset are graph and kNN
  methods. Port about 30 lines from the reference code.
- **EASE** at series level with catalog cut `k` [S24]: one closed form, one hyperparameter.
- Same Optuna trial budget for every method, tuned on validation folds only [S15][S17].

### Phase D: two-tower retrieval, built up one change at a time
- **D0**: ID-only history (mean of watched-series embeddings) → linear → dot product, in-batch
  softmax. This is YouTube's "depth 0" rung and is expected to land near the MF/EASE family [S1].
  If it's far below EASE, it's a **bug or tuning problem, not a finding**. Debug before going on.
- **D1**: + logQ correction (item frequency from the *training window only*). Prediction:
  popularity-of-recommended falls, tail recall rises [S2]. Plus L2 normalization + temperature τ
  (τ tuned) [S2].
- **D2 onward**, each only via an experiment card: next-watch labels with rolled-back history
  instead of random holdout [S1]; example-age feature [S1]; fixed examples per user so heavy
  households don't dominate the loss [S1]; depth.
- Hard constraint: **D-series must beat tuned EASE on explore NDCG@20 on the test folds, or the
  write-up says it doesn't.** (Yggdrasil's lesson: the two-tower only matched popularity until
  audited.)

### Phase E: GBDT reranker, only if the data says ordering is the bottleneck
- **Gate:** retrieval Recall@200 is high but NDCG@20 is low relative to it. If the relevant items
  aren't in the pool, a reranker can't help, so work on retrieval instead. (Same rule as the
  earlier Kion notes.)
- Training data: dated, linked slates. Positive = watched past threshold. Negative = shown in the
  same slate and not interacted with. Grouped by slate, LambdaMART objective (Airbnb's first NN
  breakthrough was this pairwise, ΔNDCG-weighted objective [S3]).
- Features, starting with the ones YouTube calls most important: "the user's previous interaction
  with the item itself and other similar items" [S1]. Retrieval score and rank; recent
  popularity; user–series affinity from history; number of previous impressions of this series to
  this user (YouTube's "churn" feature [S1]).
- Position: include `row_position` and in-row index **at training only**, set to a constant at
  scoring. Test the identifiability check from Phase A first. Pick the dropout rate the way Airbnb
  did: sweep it and plot NDCG_rel against (NDCG_rel+bias − NDCG_rel) [S4]. The alternative is the
  shallow bias tower, trained jointly with a 10% position-feature dropout [S5].
- Report both evaluation modes (§2.4).

### Phase F: write-up
Every rung's result, including the ones that lost, with CIs. Offline-only caveat stated in the
first paragraph, not the last.

---

## 6. What this project can and cannot claim
- **Can:** under a leakage-free rolling protocol with tuned baselines, method X ranks next-week
  discoveries better or worse than Y by Δ (CI), for these segments.
- **Cannot:** that it would lift engagement online (Booking.com's −0.1 correlation [S7]), or that
  the reranker's in-slate gains transfer to new slates (closed loop, Zinkevich #35 [S6]).
- **Cannot:** compare to the CIKM'20 or later published scores (different protocols, §3.8).

---

## Sources
- [S1] Covington, Adams, Sargin. *Deep Neural Networks for YouTube Recommendations*. RecSys 2016. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- [S2] Yi et al. *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*. RecSys 2019. https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf
- [S3] Haldar et al. *Applying Deep Learning to Airbnb Search*. KDD 2019. https://arxiv.org/abs/1810.09591
- [S4] Haldar et al. *Improving Deep Learning for Airbnb Search*. KDD 2020. https://arxiv.org/abs/2002.05515
- [S5] Zhao et al. *Recommending What Video to Watch Next: A Multitask Ranking System*. RecSys 2019. https://dl.acm.org/doi/10.1145/3298689.3346997
- [S6] Zinkevich. *Rules of Machine Learning*. https://developers.google.com/machine-learning/guides/rules-of-ml
- [S7] Bernardi, Mavridis, Estevez. *150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com*. KDD 2019. https://dl.acm.org/doi/10.1145/3292500.3330744
- [S10] Hidasi, Czapp. *Widespread Flaws in Offline Evaluation of Recommender Systems*. RecSys 2023. https://arxiv.org/abs/2307.14951
- [S11] Ji, Sun, Zhang, Li. *A Critical Study on Data Leakage in Recommender System Offline Evaluation*. TOIS 2023. https://arxiv.org/abs/2010.11060
- [S12] Sun. *Take a Fresh Look at Recommender Systems from an Evaluation Standpoint*. SIGIR 2023. https://arxiv.org/abs/2210.04149
- [S13] Karpathy. *A Recipe for Training Neural Networks*. 2019. https://karpathy.github.io/2019/04/25/recipe/
- [S14] Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress?* RecSys 2019. https://arxiv.org/abs/1907.06902
- [S15] Rendle, Zhang, Koren. *On the Difficulty of Evaluating Baselines*. 2019. https://arxiv.org/abs/1905.01395 (also Rendle et al., *NCF vs MF Revisited*, RecSys 2020, https://arxiv.org/abs/2005.09683: dot product beats learned MLP similarity, which is why the towers score by dot product)
- [S16] Krichene, Rendle. *On Sampled Metrics for Item Recommendation*. KDD 2020. https://www.ijcai.org/proceedings/2021/0651.pdf
- [S17] Shehzad, Jannach. *Everyone's a Winner! On Hyperparameter Tuning of Recommendation Models*. RecSys 2023. https://dl.acm.org/doi/10.1145/3604915.3609488
- [S18] Li, Vardasbi, Yates, de Rijke. *Repetition and Exploration in Sequential Recommendation*. SIGIR 2023. https://dl.acm.org/doi/10.1145/3539618.3591914
- [S19] Agarwal et al. *Estimating Position Bias without Intrusive Interventions*. WSDM 2019. https://arxiv.org/abs/1812.05161
- [S20] Pérez Maurera et al. *ContentWise Impressions: An Industrial Dataset with Impressions Included*. CIKM 2020. https://arxiv.org/abs/2008.01212
- [S21] Reference code, `Utils/dataset.py` (`_generate_urm_impressions`): local clone at `/home/kronos/ml/contentwise-impressions`
- [S22] Felicioni, Ferrari Dacrema, Cremonesi. *A Methodology for the Offline Evaluation of Recommender Systems in a User Interface with Multiple Carousels*. 2021. https://arxiv.org/abs/2105.06275
- [S23] Ihemelandu, Ekstrand. *Inference at Scale: Significance Testing for Large Search and Recommendation Experiments*. SIGIR 2023. https://arxiv.org/abs/2305.02461 (and *Multiple Testing for IR and RecSys Experiments*, ECIR 2024)
- [S24] Steck. *Embarrassingly Shallow Autoencoders for Sparse Data*. WWW 2019. https://arxiv.org/abs/1905.03375
