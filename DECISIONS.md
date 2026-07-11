# Decision log

Each entry: what was decided, why, what was rejected, and what is still open.
Status is one of: OPEN (still being argued), DECIDED, REVISED (superseded by a later entry).

---

## D1. Primary metric: based on watch completion — OPEN (2026-09-27)

**Business goal (from the PM):** leadership tracks total hours watched and monthly
cancellations. Cancellations rose last quarter. Support tickets say "the home screen always shows
me the same stuff".

**How we got here**
1. The PM's "better" was several unmeasurable wishes: more watching, fewer cancellations, less
   repetition, more of the catalogue surfaced, more pay-per-view purchases, per-person profiles.
2. First choice was watch time, as the value leadership already tracks.
3. **The data has no duration column.** `series_length` counts episodes, not minutes. The only
   watch signal is `vision_factor` ∈ [0, 1]: how far into one item the user got. Watch time can
   only be approximated.
4. So the primary metric will be built on **watch percentage (`vision_factor`)**.

**Choices still open. The metric isn't defined until these are answered.**
- **Denominator.** Candidates:
  - (a) mean over clicked items only. Rejected direction: the model controls the denominator, so
    fewer, safer clicks can score higher.
  - (b) per impression, with unclicked = 0. Rewards both click and watch-through. Only computable
    on logged lists, and only click-bearing lists are dated.
  - (c) per user-week sum. Closest to "usage", but not attributable to a list.
- **Comparability across item types.** A fraction of a 22-minute episode ≠ a fraction of a 2-hour
  film. Either accept an episode bias explicitly or compare within `item_type`.
- **Episode → series.** The recommendation unit is `series_id`, while `vision_factor` is per
  episode. Need an aggregation rule.

**Open data question (for the audit):** a view logs a start row and an end row. Does pause/resume
create separate views?

**Next step:** write the formula (numerator, denominator, averaging unit, source columns) and
state which of the trade-offs above it accepts.

---

## D2. Secondary metric: novelty — REVISED 2026-09-28 (see below)

Proposed by: data scientist. The formal definition is pending (see discussion).

Clarified 2026-09-28: D2 is the *metric* (how novelty is measured). How the system *produces*
novelty is a separate decision (D3). Still needed for D2: (1) which novelty: new-to-user,
new-vs-previous-screen, or rare-across-service; (2) how it's read against the primary metric:
a constraint (primary may drop at most X%) or a reported trade-off curve.

Revision 2026-09-28:
- **Definition chosen: new to the user**, decided per `series_id` (a new episode of a series being
  watched is not new). "Watched before" uses the same threshold as D1. Known limitation: "user"
  is a shared household account.
- **Rejected:** including the model's predicted liking in the novelty definition. A metric must be
  computable from user behaviour alone; otherwise the model grades itself. Liking is measured by
  the primary metric (D1).
- **Consequence of D4:** a fixed explore share makes "share of feed that is novel" true by
  construction, so it is a *setting*, not a metric. Novelty is measured by outcomes instead:
  - **Discovery rate:** share of a user's watched series in the week that are new to them.
  - **Explore-section performance:** D1's primary metric computed within the explore section only.
- **Open:** formula for discovery rate (averaging unit: per user-week?).

---

## D3. Serving policy: sampled lists with logit boosts — PROPOSED (2026-09-28)

**Idea (data scientist):** don't show a fixed top-K. Sample the list from softmax(logits / T).
The content team can boost titles by adjusting their logits.

**Why it's promising**
- Rotation across visits targets the per-user repetition complaint directly. Temperature T is the
  dial.
- Adding log(b) to a logit multiplies that title's unnormalized weight by exactly b, so "boost 2×"
  is well defined. Boosts live in the policy layer: no retraining when business priorities change.
- Known display probabilities make later off-policy evaluation (inverse propensity scoring)
  possible. The existing logs lack propensities, which is the main blocker to honest offline
  evaluation today.

**Corrections to the first version**
- Sample from top-N candidates, not the full 28k-series catalogue. Tail mass is large.
- A list needs sampling without replacement (Plackett–Luce; Gumbel-top-k trick), not a single
  softmax draw.
- Log the boost and the final display probability. Otherwise boosted titles look popular in the
  next training set (feedback loop).
- T has a cost in the primary metric. Choosing it depends on D2's constraint/trade-off rule.

**Open:** N, how T is chosen, who owns the boost list, and whether boosts get a cap.

---

## D4. Home-screen layout: "continue watching" + "explore" sections — PROPOSED (2026-09-28)

**Idea (data scientist):** the feed has a continue-watching section and an explore section. The
explore share is one setting that the SDE team can change.

**Why:** turns a trade-off between objectives into a layout decision that product and engineering
can own. It also separates repeat viewing (easy to predict) from discovery (hard) in evaluation,
so repeat viewing can't inflate the discovery numbers.

**Related serving rule (belongs with D3):** new titles the user responds to stay on screen; titles
shown repeatedly without a click fade out (previous-impression count as a ranking feature).

**Open:** initial explore share. Set it from data: measure the current share of weekly watching
that is new series, and its spread across users (bingers vs browsers). Global for now; per-user is
a later hypothesis.

---

## D5. Where novelty lives in the architecture — DECIDED (2026-09-28)

**Question (data scientist):** a novelty ranker first, then a likability ranker?

**Decision:** no novelty model. Three layers, as in YouTube and Netflix-style home screens:
1. **Candidate generation (recall):** several sources. The explore section's sources apply the
   "not watched before" filter (D2's definition). This is a lookup, not a prediction.
2. **Ranking (precision):** one learned model for the D1 objective,
   P(click) × E[watch fraction | click].
3. **Policy / re-ranking:** MMR diversity (λ), sampling for rotation (T, D3), content boosts (D3),
   fade-out of ignored titles, section sizes (D4). All tunable without retraining.

**Rejected:** a cascade with a graded-novelty ranker before the likability ranker. Each cascade
stage decides what later stages can see, so ranking by "different from your taste" first
discards what the user would like before the likability model sees it. Rule: early stages are
judged on recall of the *final* objective. Hard yes/no filters are fine anywhere.

**Why this shape survives changing requirements:** the learned model does one stable job. Business
priorities are settings (λ, T, boosts, explore share), not retraining.

**Continue-watching section:** candidates are in-progress series. Baseline = most recently watched
first. A learned model must beat it before it's used.

---

## D7. The spec as one optimization program — REVISED 2026-09-29 (see end)

**Outcomes of showing series s at slot k to user u at time t:** C ∈ {0,1} (started a view from
this list), W ∈ [0,1] (how far, rolled up to series; rule from notebook 03). Value Y = C·W.

**Primary (closes D1's denominator):** expected watch per impression,
E[Y] = P(C=1)·E[W | C=1]; unclicked impressions count as 0.

**Model (position-based click model):** P(C=1 | u,s,k) = θ_k · r(u,s).
θ_k = P(slot k is examined), a property of the screen; r = P(click | examined), learned.
w(u,s) = E[W | C=1, u,s], learned.

**Program:**
    maximize_L  Σ_k θ_k · r(u,s_k) · w(u,s_k)
    s.t.  s_k distinct; explore slots s_k ∉ H_u(t) (D2); #explore = ⌈ρK⌉ (D4); s_k ∈ candidates(u,t)
Since θ_k is non-increasing in k, the optimum sorts each section by f = r·w (rearrangement
inequality). θ matters for training (debiasing), not for ranking.

**Learning:** r with log loss (calibrated probability), position as a training-only feature.
w in [0,1] on clicked impressions, squared error or cross-entropy on the fraction (both give the
conditional mean). One model vs two for f: open (loss exercise).

**Secondary:** discovery rate = per active user-week, share of series watched past θ that are new
to the user; averaged over users with ≥1 such watch.

**Guardrails (no worse than baseline):** continue-watching section performance; weekly catalogue
coverage; share of impressions to the top 1% of series.

**Offline measurement (stand-ins, stated as such):** (a) ranking quality of f inside logged lists;
(b) next-week NDCG with gain W over all watching, for retrieval.

**Open:** θ, episode→series rule (nb 03); ρ (nb 05); one vs two models (exercise).

Revision 2026-09-29 (DS): spec simplified.
- **Primary:** expected watch per impression. Rationale (DS): users pay because they watch; watch
  fraction is the proxy for contentment.
- **Secondary:** discovery rate. Rationale (DS): the goal is that the user *tries* something new; a
  new title shown but ignored doesn't address the complaint. Principle: measure the outcome
  (novelty experienced), not the lever (novelty displayed).
- **Guardrail (only one):** the primary metric must not drop more than X% while discovery is
  pushed. X is a business choice.
- **Dropped from guardrails:** continue-watching performance (covered by the overall guardrail);
  catalogue coverage and top-1% concentration are now diagnostics only (plotted, not enforced).
- Status: DECIDED, except θ ("watched" threshold, nb 03) and X.

Revision 2026-09-29 (DS): **θ dropped; everything continuous.**
- Primary evaluated with graded NDCG, gain = watch fraction W (linear gain), not a 0/1 "watched".
- Discovery rate = Σ W over series new to the user ÷ Σ W over all series, per user-week.
- "New to the user" = no prior view of the series at all.
- Rationale (DS): the metric is continuous by nature; a binary cutoff adds an arbitrary parameter.
  No absolute durations were logged, so relative watch fraction is what we have.
- X stays a tunable setting (business / A-B choice).
- Remaining open: the rule that turns view rows into W (nb 03).
