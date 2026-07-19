# Approach used in this prototype

This page is the short answer to "what did you actually build, and how does
it map back to the thesis?" The other two pages in this folder go deeper:
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md)
covers the formula in detail, and
[`02-ml-model-and-technologies.md`](02-ml-model-and-technologies.md) covers
what (if any) ML is involved and the tech stack.

## The one-line summary

A person watches a short accident video, describes it with four simple
labels, a fixed formula turns those labels into a 0–1 risk score, and the
prototype re-runs that formula after changing one label to show "what would
have happened if this one thing were different." That's the entire system.

## Why this approach, specifically

The thesis is explicit that it is **not** trying to build the most accurate
possible accident predictor — plenty of existing work already does that with
deep learning (see the Related Works summary in
[`../learning/01-dataset-and-related-work.md`](../learning/01-dataset-and-related-work.md)).
Instead, the goal is **interpretability**: every number in this prototype
can be traced back, by hand, to a human decision. That's a deliberate
trade-off — lower accuracy ceiling, in exchange for a system where you can
always answer "why did the score change?" in one sentence. A rule-based,
hand-weighted formula is the simplest possible way to make that trade-off
real and demonstrable in a prototype, rather than a large trained model
whose internal reasoning can't be inspected.

## How the pieces map onto the thesis chapters

| Thesis chapter / concept | What it becomes in this repo |
|---|---|
| Ch. 3 Problem Statement — Table 3.1 (the 4 scene variables) | The `SPEED_SCORE`, `VISIBILITY_SCORE`, `PROXIMITY_SCORE`, `WEATHER_SCORE` dictionaries in `risk_model.py` |
| Ch. 3 Problem Statement — Fig 3.1 (the causal DAG) | The `DEFAULT_WEIGHTS` dictionary in `risk_model.py` (see [`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md) for how the DAG's weights were reconciled into these numbers) |
| Ch. 4 Dataset Description | The `videos/` folder (3 real VRU-Accident clips) and `data/scene_variables.csv` (the manual annotations) |
| Ch. 5.1 Scene Variable Extraction | Manually filling in `data/scene_variables.csv` after watching each video |
| Ch. 5.2 Causal Graph Construction | The fixed structure of `calculate_risk()` — four variables feeding into one score, matching the DAG's shape |
| Ch. 5.3 Accident-Risk KPI Estimation | `risk_model.calculate_risk()` and `risk_model.interpret_risk()` |
| Ch. 5.4 Intervention-Based Simulation | `risk_model.simulate_intervention()`, invoked per-video in `prototype.py`'s `INTERVENTIONS` dictionary |
| Ch. 6 Prototype Implementation / Fig 6.1 | `prototype.py` end to end, plus the chart-generation functions (extending Fig 6.1's 3-variable sketch to the full 4-variable model) |
| Ch. 7 Limitations and Future Work | [`../02-next-steps-and-future-work.md`](../02-next-steps-and-future-work.md) |

## What's genuinely new versus what's "filling in the blanks"

The thesis draft describes the *shape* of this system in prose, but leaves
several concrete decisions unmade (exact weight numbers that add up
correctly, what weather's score mapping should be, what an "intervention"
looks like in actual code, how results should be visualized). Building the
prototype meant making those decisions concretely and consistently — that's
documented, with reasoning, in
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md)
so it's easy to explain and defend later.
