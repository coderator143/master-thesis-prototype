# ML model, approach, and technologies (new direction)

> **This page describes the planned Phase 3+ direction, not what's built
> yet.** For what v1 actually does today (no ML at all), see
> [`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md).
> Status of each phase lives in
> [`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md), including
> the pending supervisor-confirmation note — this direction is a real
> departure from what the already-submitted draft says the prototype would
> do (*"instead of training large multimodal models..."*), and that's worth
> keeping visible rather than writing over.

## ML model used: a trained classifier, and that's a deliberate change

Unlike v1, the new direction does train a model: a **Random Forest** (or
Gradient Boosting — XGBoost/LightGBM) classifier, trained on the variables
extracted in Phase 1, predicting accident risk. This is a genuine ML model
with learned parameters, not a formula — see Phase 3/4 in
[`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md) for the full
mechanics (including the unresolved question of what the training target
even is, since VRU-Accident videos don't come with a risk label — resolved
there via a dense-caption-derived severity proxy).

**Why Random Forest / Gradient Boosting specifically**, over other options:

- Both work well on small-to-medium tabular data (dozens to low hundreds of
  rows) — appropriate for the ~50–100 video first pass, unlike deep
  learning approaches that need much more data.
- Both expose **feature importance** directly, without extra work — this is
  the thing that replaces `DEFAULT_WEIGHTS` from v1.
- Neither needs a GPU — realistic to train and iterate on a laptop.
- Both handle a mix of categorical and continuous variables (weather,
  speed, distance, etc.) without much preprocessing.

## Where do the "weights" come from now? Two different meanings, don't conflate them

This is worth being precise about, since it's an easy thing to get wrong in
the final report:

1. **Model parameters** — for a linear model these would be literal
   weights; Random Forest doesn't have this in the same sense, it learns a
   collection of decision trees instead.
2. **Feature importance** — a score per input variable (e.g. speed 0.34,
   distance 0.27, visibility 0.16, weather 0.08...) measuring how much that
   variable contributed to the model's predictions across the training
   data. **This is what replaces the manually-set weights.** It's stronger
   than v1's approach in one specific sense — it's derived from data rather
   than guessed — but it answers a different question than v1's weights
   did, and that difference matters (next section).

## The caveat that has to be in the report: feature importance is not a causal weight

Feature importance says "this variable helped the model predict risk in
this training data." It does **not** say "changing this variable causes
risk to change by this much," which is what v1's weights were trying to
approximate (badly, but at least honestly framed as a guess) and what the
thesis's own causal framing is ultimately about. A variable can have high
feature importance because of confounding — e.g. if bad weather in the
training data happens to correlate with a road type that's independently
riskier, the model can't tell those apart, but a human reading a feature-
importance bar chart might assume it can.

This is exactly the limitation the thesis's Related Works chapter raises
about other ML-based accident systems — training a classifier and treating
its output as causal doesn't avoid that critique just because the
classifier here is simpler (a Random Forest instead of a deep network). The
honest framing for the report: **feature importance is used as an
interpretable, data-driven proxy for variable relevance — not as validated
causal effect sizes.** The DAG (Phase 5) is what carries the actual causal
claims, and it stays separate and manually authored for exactly this
reason. See [`../learning/00-key-concepts.md`](../learning/00-key-concepts.md)
for more on the distinction.

## The approach used

In one sentence: **a trained correlational classifier for the risk
prediction/KPI, plus a separate hand-built causal DAG for the causal
claims and the recommendation logic** — deliberately not one system trying
to do both jobs. The simulator (Phase 6) re-runs the classifier on edited
inputs; the recommendation engine (Phase 7) uses `variable_metadata.csv`'s
controllability flags plus the classifier to search for realistic
interventions, and can lean on the DAG's structure (once built) to avoid
searching nonsensically (e.g. trying to "fix" a variable with no plausible
causal path to risk).

## Technologies needed (new dependencies vs. v1)

v1's stack was deliberately minimal (Python, pandas, matplotlib). The new
direction genuinely needs more, since it's doing real CV and ML work:

| Technology | Used for | New vs. v1? |
|---|---|---|
| Python 3.13 | Still the only language | Same |
| pandas | Still the tabular data layer (`scene_dataset.csv`, `variable_metadata.csv`) | Same |
| matplotlib | Still charts (risk distributions, feature-importance bar charts) | Same |
| `ultralytics` (YOLO) | Object detection — vehicles, pedestrians, cyclists (Phase 1) | New |
| A tracker (ByteTrack or DeepSORT) | Associating detections across frames to get motion/speed proxies (Phase 1) | New |
| OpenCV | Frame extraction, optical flow, brightness/visibility analysis (Phase 1) | New |
| scikit-learn (Random Forest) and/or `xgboost` | Training the classifier and extracting feature importances (Phase 3) | New |
| A small graph structure (plain dicts, or `networkx` if the traversal logic in Phase 7 gets non-trivial) | Representing the causal DAG (Phase 5) | New |

Deliberately still not planned: any deep learning training beyond the
classifier itself (no fine-tuning a vision-language model — the thesis's
own scope explicitly excludes that), no web framework unless Phase 6's
simulator ends up needing a real UI rather than a script/notebook, no
database (CSV is still enough at this scale).

## How it all ties together with the thesis

The thesis's central claim was **interpretable, intervention-based
reasoning as an alternative to black-box ML prediction.** The new direction
doesn't abandon that claim, but it does complicate it: part of the system
(the classifier) *is* now closer to the black-box category the thesis
originally critiqued, and the interpretability claim now rests specifically
on keeping the DAG separate, hand-built, and clearly labeled as the source
of any causal (not just predictive) claims — plus being explicit in the
report about the feature-importance-vs-causal-weight distinction above.
That reframing is exactly why this direction needs to be discussed with the
supervisor rather than treated as a pure implementation upgrade — it changes
the thesis's argument, not just its code. See
[`00-approach-overview.md`](00-approach-overview.md) for the full
chapter-by-chapter mapping.
