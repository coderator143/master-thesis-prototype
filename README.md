# Thesis Prototype: Accident-Risk KPI Simulator

Prototype for the M.Sc. thesis *"A Prototype System for Simulating the
Effect of Visual Scene Variables on Accident-Risk KPIs using Simple Causal
Models"* (supervisor: Jakob Suchan).

**Status: mid-pivot.** v1 (below) is a working, hand-built weighted formula
— no ML. The project is moving to a bigger 7-phase architecture (CV feature
extraction → trained classifier → separate hand-built causal DAG →
interactive simulator → recommendation engine), which is a genuine
departure from the already-submitted draft and is **pending confirmation
with the supervisor.** See
[`docs/01-phases-and-roadmap.md`](docs/01-phases-and-roadmap.md) for the
live plan and status of each phase.

v1, working today: given a small set of manually observed scene variables
for a traffic accident video (speed, visibility, pedestrian proximity,
weather), computes an accident-risk score and simulates "what if" changes
to one variable at a time (e.g. "what if speed had been lower?").

## Start here

Read [`docs/00-overview.md`](docs/00-overview.md) for a plain-language
explanation of the whole project (both v1 and the planned direction), then
[`docs/01-phases-and-roadmap.md`](docs/01-phases-and-roadmap.md) for what's
being built next. To run v1 specifically:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python prototype.py
```

(See [`docs/02-how-to-run-the-v1-prototype.md`](docs/02-how-to-run-the-v1-prototype.md)
for details.)

## Layout

- `risk_model.py`, `prototype.py` — **v1**: the risk-scoring and
  intervention-simulation logic, hand-built formula, no ML.
- `data/scene_variables.csv` — v1's manually annotated variables per video
  (has a `split` column: `test` or `train`).
- `videos/test/` — the 3 accident video clips v1 was built and demoed
  against.
- `videos/train/` — where additional videos are added as the dataset grows.
- `outputs/` — v1's generated result tables and charts.
- `docs/` — plain-language explanation of the project, both what's built
  (v1) and what's planned (the new direction).
  - `docs/learning/` — background concepts, the dataset, and related work.
  - `docs/approach/` — what was built and what's planned: the models, the
    real dataset source, the tech stack, and how it all maps to the thesis.

Notes:
- `videos/` (both subfolders) is git-ignored (large files, more will be
  added) — stays local only.
- The *full* VRU-Accident dataset (1,000 videos + real ground-truth
  annotations) is available locally at `/Users/warshaw65/Desktop/VRU-Accident/`
  — a separate folder/repo, not copied into this project. See
  [`docs/approach/03-dataset-source.md`](docs/approach/03-dataset-source.md).
