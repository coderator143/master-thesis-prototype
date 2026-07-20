# Approach used in this prototype

This page is the short answer to "what did you actually build (or plan to
build), and how does it map back to the thesis?" The other pages in this
folder go deeper:
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md)
covers v1's formula, [`02-ml-model-and-technologies.md`](02-ml-model-and-technologies.md)
covers the ML model, causal DAG, and tech stack for the new direction, and
[`03-dataset-source.md`](03-dataset-source.md) covers the real dataset this
all runs on.

> **This page describes two things at once, on purpose:** what v1 already
> is, and what the new 7-phase direction is planned to become. They're kept
> in the same doc because the new direction genuinely builds on/replaces
> v1, and the contrast is the most useful way to explain the pivot. See
> [`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md) for status
> and the pending supervisor-confirmation note.

## v1, in one line

A person watches a short accident video, describes it with four simple
labels, a fixed hand-picked formula turns those labels into a 0–1 risk
score, and the prototype re-runs that formula after changing one label to
show "what would have happened if this one thing were different." Fully
working, no ML anywhere.

## The new direction, in one line

Extract scene variables from video with computer vision (not by hand), train
a real classifier to predict risk from them (not a hand-picked formula),
keep a separate hand-drawn causal DAG for the actual causal reasoning (the
classifier only knows correlations), and build a simulator + recommendation
engine on top that only ever suggests changing variables that are genuinely
controllable.

## Why the pivot, and what it costs

v1's whole pitch was **interpretability over accuracy**: every number can be
traced by hand to a human decision, at the cost of a much lower accuracy
ceiling than a trained model could reach. The new direction trades some of
that back — a trained classifier's internal reasoning is not directly
inspectable the way a formula is — in exchange for weights (feature
importances) that are actually derived from data instead of guessed, and a
system that can, in principle, get more accurate as more videos are added.

That's a legitimate trade to make, but it changes what the thesis is
claiming, which is why [`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md)
flags it as needing supervisor confirmation rather than treating it as a
pure implementation detail. The DAG (Phase 5) is what keeps *some*
hand-inspectable causal reasoning in the system even after the classifier
arrives — see the methodological caveat in
[`../00-overview.md`](../00-overview.md) about not overstating what the
classifier + simulator combination actually proves.

## How the pieces map onto the thesis chapters

| Thesis chapter / concept | v1 (built) | New direction (planned) |
|---|---|---|
| Ch. 3 — Table 3.1 (scene variables) | `SPEED_SCORE`/`VISIBILITY_SCORE`/`PROXIMITY_SCORE`/`WEATHER_SCORE` dicts in `risk_model.py` | **Done** — `data/scene_dataset.csv`, produced by `extract_features.py` + `ground_truth.py` (see `03-dataset-source.md`) |
| Ch. 3 — Fig 3.1 (causal DAG) | `DEFAULT_WEIGHTS` in `risk_model.py`, reconciled from the figure's inconsistent numbers (see `01-risk-model-and-intervention-logic.md`) | Phase 5's DAG — a real multi-hop graph, not a flat weighted sum, and explicitly *not* used to compute the KPI |
| Ch. 4 — Dataset Description | `videos/` (3 clips) + `data/scene_variables.csv` (manual guesses) | The full 1,000-video dataset with real GT — `03-dataset-source.md`; 103 videos processed so far |
| Ch. 5.1 — Scene Variable Extraction | Manual annotation by watching each video | **Done** — Phase 1: YOLO+ByteTrack+optical-flow CV extraction, GT lookup for weather/road type |
| Ch. 5.2 — Causal Graph Construction | Baked into `calculate_risk()`'s fixed structure | Phase 5 — an explicit, separate graph structure |
| Ch. 5.3 — Accident-Risk KPI Estimation | `risk_model.calculate_risk()` | Phase 3 (train) + Phase 4 (predict) |
| Ch. 5.4 — Intervention-Based Simulation | `risk_model.simulate_intervention()` | Phase 6 (simulator) + Phase 7 (recommendation engine) |
| Ch. 6 — Prototype Implementation | `prototype.py` end to end | All 7 phases, once built |
| Ch. 7 — Limitations and Future Work | N/A | [`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md) |

## What's genuinely new versus what's "filling in the blanks"

For v1, the thesis draft described the *shape* of the system in prose but
left concrete decisions unmade (exact weights, score mappings, what
"intervention" means in code) — those are documented in
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md).

The new direction is a bigger jump: it's not filling in a blank the draft
left, it's proposing a different architecture than the draft describes at
all. That distinction matters for how it gets written up — it should read
as "here's how the approach evolved and why," not as if it were the
original plan all along.
