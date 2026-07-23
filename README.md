# Thesis Prototype: Accident-Risk KPI Simulator

Prototype for the M.Sc. thesis *"A Prototype System for Simulating the
Effect of Visual Scene Variables on Accident-Risk KPIs using Simple Causal
Models"* (supervisor: Jakob Suchan).

**Status:** building the v2 architecture — a 7-phase pipeline (CV feature
extraction → trained classifier → separate hand-built causal DAG →
interactive simulator → recommendation engine), a genuine departure from
the already-submitted draft, **pending confirmation with the supervisor**
before Phase 3 (training) proceeds. **Phase 1 is done.** See
[`docs/01-phases-and-roadmap.md`](docs/01-phases-and-roadmap.md) for the
live plan and status of each phase.

v1 — the original hand-built weighted formula (no ML), working, kept as
the baseline this project started from — now lives in its own
[`v1_legacy/`](v1_legacy/) folder, separate from v2 so the two don't get
tangled together.

## Start here

Read [`docs/00-overview.md`](docs/00-overview.md) for a plain-language
explanation of the whole project (both v1 and v2), then
[`docs/01-phases-and-roadmap.md`](docs/01-phases-and-roadmap.md) for what's
built and what's next.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run v1: `cd v1_legacy && python prototype.py` (see
[`docs/02-how-to-run-the-v1-prototype.md`](docs/02-how-to-run-the-v1-prototype.md)).
To re-run Phase 1's extraction pipeline: `python extract_features.py` from
the repo root.

## Layout

- `v1_legacy/` — v1: the original hand-built formula (`risk_model.py`,
  `prototype.py`), its own data/outputs, no ML. See
  [`v1_legacy/README.md`](v1_legacy/README.md).
- `ground_truth.py`, `extract_features.py` — **v2, Phase 1 (done):** the CV
  extraction pipeline (YOLO detection + tracking, optical flow, brightness
  analysis) and ground-truth JSON loading.
- `mine_risk_labels.py`, `data/variable_metadata.csv` — **v2, Phase 2
  (done):** mines the `risk_label` training target from dense captions;
  the static controllability table for Phase 7.
- `scripts/sample_train_videos.py` — how the 100 training videos were
  chosen from the full dataset and copied into `videos/train/`.
- `data/scene_dataset.csv`, `data/scene_dataset_raw.csv` — Phase 1's
  output (now with `risk_label` added by Phase 2): real scene variables
  for all 103 local videos.
- `data/train_video_manifest.csv` — which videos got sampled into
  `videos/train/`, and from where.
- `videos/test/<source>/`, `videos/train/<source>/` — the video clips
  available locally (3 in `test/`, 100 in `train/`).
- `outputs/debug_frames/` — annotated frames saved while validating Phase
  1's detection/tracking before trusting the full batch run.
- `docs/` — plain-language explanation of the project, both what's built
  and what's planned.
  - `docs/learning/` — background concepts, the dataset, and related work.
  - `docs/approach/` — what was built and what's planned: the models, the
    real dataset source, the tech stack, and how it all maps to the thesis.

Notes:
- `videos/` is git-ignored (large files, more will be added) — stays local
  only.
- The *full* VRU-Accident dataset (1,000 videos + real ground-truth
  annotations) is available locally at `/Users/warshaw65/Desktop/VRU-Accident/`
  — a separate folder/repo, not copied into this project. See
  [`docs/approach/03-dataset-source.md`](docs/approach/03-dataset-source.md).
