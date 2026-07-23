# What is this project?

This is the prototype for the thesis *"A Prototype System for Simulating the
Effect of Visual Scene Variables on Accident-Risk KPIs using Simple Causal
Models"* (supervisor: Jakob Suchan). This page explains the whole idea in
plain language — no jargon, no academic phrasing.

> **Status note (2026-07-20):** the project is mid-pivot. What's described
> below as "the plan" is a substantially bigger architecture than the one in
> the already-submitted Step-1 draft, and **has not yet been confirmed with
> the supervisor.** The old, already-working version (a hand-built weighted
> formula, no ML) still exists and still runs — see
> [`02-how-to-run-the-v1-prototype.md`](02-how-to-run-the-v1-prototype.md).
> Treat this page as "where we're headed," not "what's done." The
> phase-by-phase status lives in
> [`01-phases-and-roadmap.md`](01-phases-and-roadmap.md).

## The problem, in one paragraph

There's a lot of AI that can look at a dashcam video and say "this looks like
a dangerous scene" or "this looks like an accident." That's useful, but it
doesn't answer a more useful question: **if something about the scene had
been different, would it have been safer?** For example: would slowing the
car down have helped? Would better lighting have helped? Most existing
systems are trained to *detect* danger, not to *explain what would change
it.* That's the gap this thesis is about.

## The plan, in one paragraph

Watch real accident videos and use computer vision to pull out concrete
scene facts (how fast the vehicle was going, how far the pedestrian was,
how many pedestrians, the weather, how crowded the traffic was, the road
type, whether there was an obstacle). Store those facts in a table. Train a
machine-learning classifier on that table to predict accident risk — this
is a genuine, trained ML model, not a hand-picked formula, and it's the KPI.
Separately, hand-draw a causal diagram (a DAG) encoding domain knowledge
about which variables plausibly cause which — this stays manually authored,
not learned, because the classifier only knows correlations, not causes. A
simulator lets you edit a scene's variables and instantly re-run the trained
classifier to see the risk change. A recommendation engine then searches
over the variables that are actually *controllable* (you can't recommend
"make the weather sunny") to find realistic interventions that would have
lowered the risk.

## Why this is a genuine pivot from the submitted draft

The Step-1 draft explicitly says: *"Instead of training large multimodal
models, the proposed system focuses on manually defining interpretable
variables."* The new plan does train a model (a Random Forest or similar).
That's not a small tweak — it changes what kind of thesis this is, from
"demonstrate causal-style reasoning without any ML" to "combine a trained
ML predictor with a separate hand-built causal structure." Both are
legitimate research directions, but they're different claims, and the
second one needs the supervisor's sign-off since it goes beyond what was
already submitted and accepted. See
[`01-phases-and-roadmap.md`](01-phases-and-roadmap.md) for the full
reasoning and what's pending.

## The seven phases, at a glance

1. **Visual feature extraction** — CV models (YOLO, a tracker, optical
   flow, a segmentation model, brightness analysis) pull structured
   variables out of each video.
2. **Structured storage** — every video's variables go into one CSV;
   whether a variable is controllable (for later use) lives in a *separate*
   `variable_metadata.csv`, since controllability doesn't change per video.
3. **Train a classifier** — a Random Forest/XGBoost model learns to predict
   risk from the extracted variables. Its *feature importances* become the
   data-driven equivalent of the "weights" the old formula hand-picked.
4. **Risk prediction** — the trained classifier's output (a risk
   probability/class) *is* the accident-risk KPI.
5. **Causal DAG** — a separate, hand-drawn diagram of assumed cause →
   effect relationships. Not learned. Used for reasoning about structure,
   not for computing the score.
6. **Interactive simulator** — edit a scene's variables, re-run the same
   trained classifier, see the new score. No new model involved.
7. **Recommendation engine** — searches over controllable variables only
   (per `variable_metadata.csv`) to suggest realistic interventions that
   would lower the predicted risk.

Details, status, and the open questions for each phase are in
[`01-phases-and-roadmap.md`](01-phases-and-roadmap.md).

## An important methodological caveat, worth understanding early

The classifier (Phases 3–4) and the causal DAG (Phase 5) answer different
questions, and it matters that they don't get blurred together:

- The classifier learns **correlations** in the training data. Its feature
  importances say "this variable was useful for predicting risk in this
  dataset" — not "this variable causes risk." Re-running it with an edited
  input (Phase 6) tells you what the model *would have predicted*, which is
  only a trustworthy stand-in for "what would actually have happened" if
  the correlations it learned happen to reflect real causal mechanisms —
  which isn't guaranteed, and is exactly the limitation the thesis's own
  Related Works chapter raises about black-box ML systems.
- The DAG is the actual causal claim — and it's manually authored, not
  learned, same as before.

This is worth stating plainly in the final report rather than glossing
over: the simulator is doing correlational re-prediction, dressed in a
causal-sounding interface. That's a reasonable, common simplification for a
prototype (real causal ML — e.g. structural causal models fit to do-
calculus — is a much bigger undertaking), but it should be named as a
limitation, not implied to be more rigorous than it is. See
[`learning/00-key-concepts.md`](learning/00-key-concepts.md) for more on
this distinction.

## The dataset — bigger than we thought

A full local copy of the actual VRU-Accident benchmark repository exists at
`/Users/warshaw65/Desktop/VRU-Accident/` (outside this project folder) —
**1,000 real videos** with real ground-truth annotations (weather & light,
location, road type, accident type, accident reason, prevention method) and
a rich text description per video. See
[`approach/03-dataset-source.md`](approach/03-dataset-source.md) for the
full details and how this project will use it — in short: real GT is used
directly for the variables it already covers, computer vision is only
needed for the variables it doesn't (speed, pedestrian distance/count,
traffic density), and the first end-to-end pass targets ~50–100 videos
before scaling up.

## How the pieces of this repo fit together

| File / folder | What it is |
|---|---|
| `v1_legacy/` (`risk_model.py`, `prototype.py`, `data/scene_variables.csv`, `outputs/`) | **v1** — the original hand-built weighted formula, kept in its own folder, separate from v2. Still works, kept as the baseline this thesis started from. See [`02-how-to-run-the-v1-prototype.md`](02-how-to-run-the-v1-prototype.md). |
| `ground_truth.py`, `extract_features.py`, `data/scene_dataset.csv` | **Phase 1 (done)** — the CV extraction pipeline and its output: real scene variables for all 103 local videos. See [`01-phases-and-roadmap.md`](01-phases-and-roadmap.md). |
| `scripts/sample_train_videos.py`, `data/train_video_manifest.csv` | How the 100 training videos were chosen from the full dataset and copied in. |
| `outputs/` | v1's generated result tables and charts, plus `outputs/debug_frames/` (Phase 1 validation screenshots). |
| `videos/test/<source>/`, `videos/train/<source>/` | The video clips available locally (3 in `test/`, 100 in `train/`). |
| `docs/` | You are here. |

`docs/` has two subfolders:

- [`learning/`](learning/00-key-concepts.md) — background concepts (causal
  reasoning vs. correlation, the dataset, related work) for understanding
  *why* the project is built this way.
- [`approach/`](approach/00-approach-overview.md) — what's actually built
  (and what's planned): the model(s), the tech stack, the dataset source,
  and how it all maps back to the thesis.

Start with [`01-phases-and-roadmap.md`](01-phases-and-roadmap.md) for the
current plan and status.
