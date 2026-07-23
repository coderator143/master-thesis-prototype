# How to run the v1 prototype

> **This is the original, already-working version** — a hand-built weighted
> formula, no ML. It's being superseded by a bigger 7-phase architecture
> (CV extraction → trained classifier → causal DAG → simulator →
> recommendation engine) — see
> [`01-phases-and-roadmap.md`](01-phases-and-roadmap.md) for the live plan
> and status. This page stays accurate for what's *already built*; it's not
> where new work happens. **The code lives in `v1_legacy/`**, separate from
> the v2 pipeline at the repo root, so the two don't get tangled together.

## One-time setup

Uses the same shared `venv` as the rest of the repo — from the
`thesis-prototype/` root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(`v1_legacy/requirements.txt` also exists, with just the two packages v1
actually needs — `pandas`, `matplotlib` — for running it standalone outside
this repo.)

You'll need to run `source venv/bin/activate` again each time you open a new
terminal window before running the prototype.

## Running it

```bash
cd v1_legacy
python prototype.py
```

This will:

1. Read `v1_legacy/data/scene_variables.csv`.
2. Print a table of the estimated accident-risk score for each of the 3
   videos, and save it to `v1_legacy/outputs/risk_results.csv`.
3. Run one example "what if" intervention per video, print the before/after
   comparison, and save it to `v1_legacy/outputs/intervention_results.csv`.
4. Save chart images (`.png`) into `v1_legacy/outputs/` — one summary chart
   comparing all 3 videos, and one before/after chart per video.

## The most important first task: replace the placeholder data

`v1_legacy/data/scene_variables.csv` currently has **placeholder
guesses**, not real observations — they exist so the code has something to
run against before anyone has actually reviewed the footage. Each row says
`PLACEHOLDER - review after watching video` in the `notes` column as a
reminder.

To make the results real:

1. Watch each video in `videos/test/CAP_DATA/` (`VRU_9.mp4`, `VRU_10.mp4`,
   `VRU_14.mp4`) — the actual video files stay at the repo root, shared
   with v2, since they're the same footage either way.
2. For each one, decide honestly:
   - Was the vehicle going `low`, `medium`, or `high` speed?
   - Was visibility `high`, `medium`, or `low`?
   - Was the vehicle `far`, `medium`, or `close` to the pedestrian?
   - Was the weather `clear`, `rainy`, or `night`?
3. Edit `v1_legacy/data/scene_variables.csv` with a text editor or
   spreadsheet app, update the values, and clear out the `PLACEHOLDER`
   note.
4. Re-run `python prototype.py` (from inside `v1_legacy/`) — the results
   will update automatically.

Note: v2's `data/scene_dataset.csv` (repo root) already has real,
CV-extracted + ground-truth values for these same 3 videos (see
[`01-phases-and-roadmap.md`](01-phases-and-roadmap.md)) — v1's placeholders
were never updated to match, since v1 is no longer where active work
happens. If you want real numbers for these 3 videos, v2's output is the
place to look; only update v1's CSV by hand if you specifically want to
keep comparing the two approaches.

## Where to look at results

- `v1_legacy/outputs/risk_results.csv` — one row per video with its risk
  score and label.
- `v1_legacy/outputs/intervention_results.csv` — the before/after
  comparison for each intervention.
- `v1_legacy/outputs/all_videos_risk_summary.png` — a bar chart comparing
  all 3 videos' risk scores, color-coded green/yellow/red by risk level.
- `v1_legacy/outputs/VRU_9_intervention.png`,
  `v1_legacy/outputs/VRU_10_intervention.png`,
  `v1_legacy/outputs/VRU_14_intervention.png` — one before/after chart per
  video.

## Changing the model

- To try different variable values for a video without touching the CSV,
  run this from inside `v1_legacy/`:
  ```python
  import risk_model
  risk_model.calculate_risk(speed="low", visibility="high", proximity="far", weather="clear")
  ```
- To change which intervention is simulated for each video, edit the
  `INTERVENTIONS` dictionary near the top of `v1_legacy/prototype.py`.
- To try different weights (e.g. for a sensitivity check in the final
  report), pass a custom `weights` dictionary into `calculate_risk()` or
  `simulate_intervention()` instead of relying on `DEFAULT_WEIGHTS`.
