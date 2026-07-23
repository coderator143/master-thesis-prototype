# v1 (legacy)

This is the original prototype: a hand-built weighted formula, no ML, no
computer vision. It's superseded by the v2 architecture (CV extraction →
trained classifier → causal DAG → simulator → recommendation engine) being
built at the repo root, but it still works and is kept as the baseline this
project started from.

See the main docs for context and detail:

- [`../docs/00-overview.md`](../docs/00-overview.md) — whole-project overview.
- [`../docs/01-phases-and-roadmap.md`](../docs/01-phases-and-roadmap.md) — the current (v2) plan and status.
- [`../docs/02-how-to-run-the-v1-prototype.md`](../docs/02-how-to-run-the-v1-prototype.md) — how to run this folder specifically.
- [`../docs/approach/01-risk-model-and-intervention-logic.md`](../docs/approach/01-risk-model-and-intervention-logic.md) — how v1's formula works.

## Quick start

Uses the same shared `venv` as the rest of the repo (root `requirements.txt`
already includes everything this needs). To run standalone elsewhere, this
folder's own `requirements.txt` (`pandas`, `matplotlib`) is enough.

```bash
cd v1_legacy
python prototype.py
```

Reads `data/scene_variables.csv` (this folder's own copy, not the root
`data/`), writes results/charts to `outputs/` (this folder's own, not the
root `outputs/`).
