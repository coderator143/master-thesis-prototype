# Thesis Prototype: Accident-Risk KPI Simulator

Prototype for the M.Sc. thesis *"A Prototype System for Simulating the
Effect of Visual Scene Variables on Accident-Risk KPIs using Simple Causal
Models"* (supervisor: Jakob Suchan).

Given a small set of manually observed scene variables for a traffic
accident video (speed, visibility, pedestrian proximity, weather), this
prototype computes an accident-risk score and simulates "what if" changes
to one variable at a time (e.g. "what if speed had been lower?").

## Start here

Read [`docs/00-overview.md`](docs/00-overview.md) for a plain-language
explanation of the whole project, then
[`docs/01-how-to-run-the-prototype.md`](docs/01-how-to-run-the-prototype.md)
to set it up and run it.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python prototype.py
```

## Layout

- `risk_model.py` — the risk-scoring and intervention-simulation logic.
- `prototype.py` — runs everything end to end and saves results to `outputs/`.
- `data/scene_variables.csv` — manually annotated variables per video.
- `videos/` — the 3 accident video clips being analyzed.
- `outputs/` — generated result tables and charts.
- `docs/` — plain-language explanation of the project, the model, and next steps.
  - `docs/learning/` — background concepts, the dataset, and related work.
  - `docs/approach/` — what was built: the risk model, ML/tech stack, and how it maps to the thesis.

Note: `videos/` is git-ignored (large files, and more will be added) — it stays local only.
