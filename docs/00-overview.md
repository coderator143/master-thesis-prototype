# What is this project?

This is the prototype for the thesis *"A Prototype System for Simulating the
Effect of Visual Scene Variables on Accident-Risk KPIs using Simple Causal
Models"* (supervisor: Jakob Suchan). This page explains the whole idea in
plain language — no causal-inference jargon, no academic phrasing.

## The problem, in one paragraph

There's a lot of AI that can look at a dashcam video and say "this looks like
a dangerous scene" or "this looks like an accident." That's useful, but it
doesn't answer a more useful question: **if something about the scene had
been different, would it have been safer?** For example: would slowing the
car down have helped? Would better lighting have helped? Most existing
systems are trained to *detect* danger, not to *explain what would change
it.* That's the gap this thesis is about.

## The idea

Instead of training a big machine-learning model, this project does
something much simpler and easier to explain:

1. **Look at a few real accident videos** (from a public research dataset
   called VRU-Accident, which is full of real dashcam footage of accidents
   involving pedestrians and cyclists — "VRU" = Vulnerable Road User).
2. **Describe each video with four simple labels**, chosen by a person
   watching the video:
   - How fast was the vehicle going? (low / medium / high)
   - How clear was visibility? (high / medium / low)
   - How close was the vehicle to the pedestrian? (far / medium / close)
   - What was the weather like? (clear / rainy / night)
3. **Turn those labels into numbers** and combine them with a simple
   formula to get a single "risk score" between 0 and 1 — think of it like a
   traffic-light rating: green (low risk), yellow (moderate), red (high).
4. **Ask "what if?"** — change one label (e.g. "what if speed had been low
   instead of high?") and recalculate the score. If the score drops, that
   variable mattered. This is called an "intervention," and it's the core
   trick of the whole project.

## Why this counts as "causal," even though it's simple

A real causal model, done properly, is discovered from lots of data using
statistics. This project doesn't do that — it's explicitly **not** trying to
discover which variables actually cause accidents. Instead, a person
*manually decides* (based on common sense and domain knowledge) that speed,
visibility, proximity, and weather probably matter, draws that relationship
as a simple diagram (a "causal graph"), and then simulates "what if I change
this one thing" on top of it. It's a lightweight, interpretable stand-in for
a full causal model — good enough to demonstrate the *concept* of
intervention-based reasoning without the complexity of proper causal
discovery or training a large model. The thesis is upfront about this
limitation — it's a deliberate scope decision, not an oversight.

## The dataset

The full VRU-Accident benchmark has around 1,000 real dashcam accident
videos with rich annotations (captions, causes, weather/lighting info,
prevention suggestions). **This prototype only uses 3 of those videos**
(`VRU_9`, `VRU_10`, `VRU_14`) to keep things manageable for a first working
version — see `data/scene_variables.csv` for how each one has been
described. Expanding to more videos is listed as a next step in
[`02-next-steps-and-future-work.md`](02-next-steps-and-future-work.md).

## What's explicitly out of scope

The thesis is deliberately narrow, on purpose:

- No automatic causal discovery (the causal graph is drawn by hand).
- No training of deep learning / large multimodal models.
- No fully automated video analysis — variables are annotated by a person
  watching the video, not extracted automatically.
- No claim that the risk score is scientifically precise — it's a tool for
  *comparing* scenarios (before vs. after an intervention), not a certified
  accident-prediction system.

## How the pieces of this repo fit together

| File / folder | What it is |
|---|---|
| `risk_model.py` | The actual "causal model" — the formula, the weights, the intervention logic. No file reading/writing, just the math. |
| `prototype.py` | Runs everything end to end: reads the annotated videos, computes risk scores, runs example interventions, saves results and charts. |
| `data/scene_variables.csv` | The manually annotated variables for each video (this is the "ground truth" input). |
| `outputs/` | Everything `prototype.py` generates: result tables and charts. |
| `videos/` | The 3 real accident video clips being analyzed. |
| `docs/` | You are here — plain-language explanations, reusable for the final written thesis report. |

`docs/` itself has two subfolders:

- [`learning/`](learning/00-key-concepts.md) — background concepts (causal
  reasoning, the dataset, related work) for actually understanding *why*
  the project is built this way.
- [`approach/`](approach/00-approach-overview.md) — what was actually
  built: the risk model mechanics, what (if any) ML is used, the tech
  stack, and how it all maps back to specific thesis chapters.

See [`approach/01-risk-model-and-intervention-logic.md`](approach/01-risk-model-and-intervention-logic.md)
for the actual formula and numbers, and
[`01-how-to-run-the-prototype.md`](01-how-to-run-the-prototype.md) to run it
yourself.
