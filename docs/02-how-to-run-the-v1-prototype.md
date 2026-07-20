# How to run the v1 prototype

> **This is the original, already-working version** — a hand-built weighted
> formula, no ML. It's being superseded by a bigger 7-phase architecture
> (CV extraction → trained classifier → causal DAG → simulator →
> recommendation engine) — see
> [`01-phases-and-roadmap.md`](01-phases-and-roadmap.md) for the live plan
> and status. This page stays accurate for what's *already built*; it's not
> where new work happens.

## One-time setup

From the `thesis-prototype/` folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll need to run `source venv/bin/activate` again each time you open a new
terminal window before running the prototype.

## Running it

```bash
python prototype.py
```

This will:

1. Read `data/scene_variables.csv`.
2. Print a table of the estimated accident-risk score for each of the 3
   videos, and save it to `outputs/risk_results.csv`.
3. Run one example "what if" intervention per video, print the before/after
   comparison, and save it to `outputs/intervention_results.csv`.
4. Save chart images (`.png`) into `outputs/` — one summary chart comparing
   all 3 videos, and one before/after chart per video.

## The most important first task: replace the placeholder data

`data/scene_variables.csv` currently has **placeholder guesses**, not real
observations — they exist so the code has something to run against before
anyone has actually reviewed the footage. Each row says
`PLACEHOLDER - review after watching video` in the `notes` column as a
reminder.

To make the results real:

1. Watch each video in `videos/test/` (`VRU_9.mp4`, `VRU_10.mp4`,
   `VRU_14.mp4`).
2. For each one, decide honestly:
   - Was the vehicle going `low`, `medium`, or `high` speed?
   - Was visibility `high`, `medium`, or `low`?
   - Was the vehicle `far`, `medium`, or `close` to the pedestrian?
   - Was the weather `clear`, `rainy`, or `night`?
3. Edit `data/scene_variables.csv` with a text editor or spreadsheet app,
   update the values, and clear out the `PLACEHOLDER` note.
4. Re-run `python prototype.py` — the results will update automatically.

## Adding more videos to `videos/train/`

`videos/` has two subfolders: `test/` holds the 3 videos the prototype was
originally built and demoed against (already wired up in
`data/scene_variables.csv`), and `train/` is where additional videos go as
the dataset grows. To add one:

1. Drop the video file into `videos/train/`.
2. Add a row for it to `data/scene_variables.csv` with `split` set to
   `train` and `filename` set to `train/<your-file>.mp4`.
3. Annotate its speed/visibility/proximity/weather the same way as above.

As of now, `prototype.py` doesn't treat `train` and `test` rows any
differently — it scores every row in the CSV the same way, because nothing
in `risk_model.py` is actually learned from data yet (see
[`approach/02-ml-model-and-technologies.md`](approach/02-ml-model-and-technologies.md),
which is exactly what's now planned in
[`01-phases-and-roadmap.md`](01-phases-and-roadmap.md)).
The split is being kept ready ahead of time for when that changes — e.g. if
weights get calibrated/learned from data instead of hand-set, `train/`
videos would be what that calibration runs against, and `test/` would stay
held out to check the result still makes sense.

## Where to look at results

- `outputs/risk_results.csv` — one row per video with its risk score and
  label. This is the table to screenshot for the report's results section.
- `outputs/intervention_results.csv` — the before/after comparison for each
  intervention.
- `outputs/all_videos_risk_summary.png` — a bar chart comparing all 3
  videos' risk scores, color-coded green/yellow/red by risk level.
- `outputs/VRU_9_intervention.png`, `outputs/VRU_10_intervention.png`,
  `outputs/VRU_14_intervention.png` — one before/after chart per video.

## Changing the model

- To try different variable values for a video without touching the CSV,
  you can call the functions in `risk_model.py` directly from a Python
  shell:
  ```python
  import risk_model
  risk_model.calculate_risk(speed="low", visibility="high", proximity="far", weather="clear")
  ```
- To change which intervention is simulated for each video, edit the
  `INTERVENTIONS` dictionary near the top of `prototype.py`.
- To try different weights (e.g. for a sensitivity check in the final
  report), pass a custom `weights` dictionary into `calculate_risk()` or
  `simulate_intervention()` instead of relying on `DEFAULT_WEIGHTS`.
