# How to run the prototype

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

1. Watch each video in `videos/` (`VRU_9.mp4`, `VRU_10.mp4`, `VRU_14.mp4`).
2. For each one, decide honestly:
   - Was the vehicle going `low`, `medium`, or `high` speed?
   - Was visibility `high`, `medium`, or `low`?
   - Was the vehicle `far`, `medium`, or `close` to the pedestrian?
   - Was the weather `clear`, `rainy`, or `night`?
3. Edit `data/scene_variables.csv` with a text editor or spreadsheet app,
   update the values, and clear out the `PLACEHOLDER` note.
4. Re-run `python prototype.py` — the results will update automatically.

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
