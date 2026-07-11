# Task list

Each notebook answers one question and ends with one decision, logged in DECISIONS.md.
Every notebook opens with two cells written **before** running anything:
1. **Question**: what are we trying to find out?
2. **Prediction**: what do we expect, and what would we do if it's wrong?

Notebooks live in `notebooks/`. Data: `data/ContentWiseImpressions/data/ContentWiseImpressions/CW10M/`.

## Look at the data
- [ ] **01 Data integrity.** Nulls, codes and ranges, duplicates, joins, list lengths, time range.
      → Is the data usable as-is; what gets fixed or dropped.
- [ ] **02 First look.** Activity per user; walk a random and the heaviest user's timeline.
      → Write the story of how each table was produced.
- [ ] **03 Views.** Single rows or start/end pairs? `vision_factor` distribution; accounts stuck at 1.0;
      episodes per user-series.
      → The rule that turns view rows into W, the watch fraction used everywhere. (D7)
- [ ] **04 Time.** Daily volume, missing days.
      → Weekly train/test windows.
- [ ] **05 Repeat vs new.** Share of each week's watching that is series new to the user.
      → Starting explore share (D4); discovery-rate formula (D2).
- [ ] **06 Impressions.** Can linked lists be dated? Clicks by screen position. Does a series appear
      at varied positions?
      → Whether the ranker and position-bias handling are feasible.
- [ ] **07 Does history predict the future?** Split at T. Do users with similar pre-T history watch
      the same *new* series after T, more than random users of similar activity, and beyond
      popularity? One plot: overlap lift by activity bucket.
      → Whether collaborative filtering is justified (and how hard popularity will be to beat).
- [ ] **08 Churn.** Gaps between activity; users near the end of the data.
      → Churn definition. Test: repetition → churn, watching → retention.

## Finish the spec
- [ ] **Loss exercise** (on paper): one regression vs P(click) × E[watch | click].
- [ ] **D1 formula**, guardrails, out-of-scope list.

## Build
- [ ] **09 Evaluation + baselines.** Weekly rolling evaluation; recent popularity; continue-watching.
- [ ] **10 Classical models.** ItemKNN, RP3β, EASE.
- [ ] **11 Two-tower retrieval.** Built one change at a time.
- [ ] **12 Ranker.** Click model + watch model (only if 11 shows ordering is the bottleneck).
- [ ] **13 Policy layer + write-up.**

- [ ] **14 Demo.** Cluster series and users (check personas are stable: first half vs second
      half). Coloured placeholder cards per cluster; clicking updates recommendations live. Show
      cross-cluster picks, not just same-colour ones. Optional: same method on MovieLens with posters.

Parked: D6 paid promotion.
