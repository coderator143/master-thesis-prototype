# Phases and roadmap

This is the authoritative plan. Each phase is worked on in order — nothing
in a later phase should get built before the phase(s) it depends on are
actually done, since each one's output is the next one's input.

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Planning: architecture defined, dataset located, docs written | ✅ Done (2026-07-20) |
| 1 | Visual feature extraction | ✅ Done (2026-07-20) — 103/103 videos processed, 0 failures |
| 2 | Structured storage (`scene_dataset.csv` + `variable_metadata.csv`) | ✅ Done (2026-07-23) — `risk_label` mined from dense captions, `variable_metadata.csv` created |
| 3 | Train a classifier | ⬜ Not started — Phase 2 output ready; supervisor confirmation in progress, not blocking |
| 4 | Risk prediction | ⬜ Not started — depends on Phase 3 |
| 5 | Causal DAG | ⬜ Not started — structure specified below, not yet built as code/data |
| 6 | Interactive simulator | ⬜ Not started — depends on Phase 4 |
| 7 | Recommendation engine | ⬜ Not started — depends on Phases 2, 4, 5 |

**Overall status:** Phases 1–2 done, Phases 3–7 not started. v1 (the
hand-built weighted formula, `risk_model.py`/`prototype.py`) lives in its
own `v1_legacy/` folder, separate from the v2 pipeline at the repo root —
see [`02-how-to-run-the-v1-prototype.md`](02-how-to-run-the-v1-prototype.md).
It doesn't get deleted; it's the baseline this new direction supersedes.

**Supervisor confirmation: in progress, not treated as blocking.** This
plan trains a real ML classifier, which the already-submitted Step-1 draft
explicitly said this project would *not* do — a legitimate direction, but
enough of a departure from what was already accepted that it needed
Jakob Suchan's sign-off. Status: the user is sending a confirmation email
today (2026-07-23) and has explicitly directed that implementation proceed
on the full v2 plan now, rather than waiting for a reply. Recorded here for
traceability, in case the final report needs to explain the timeline: Phase
3 onward was built before formal written confirmation was received, on the
user's instruction, with the confirmation request already in flight. If the
supervisor comes back with different guidance, the phases below (Phase 5's
DAG especially, since it's independent of the classifier) are still valid
work either way — only Phase 3's specific framing (a trained classifier as
the KPI source) would need to be revisited.

---

## Phase 1 — Visual feature extraction

**Goal:** turn each video into a row of concrete variables, using computer
vision, not manual guessing.

**Status: done (first pass).** `ground_truth.py` (GT loading/parsing) and
`extract_features.py` (the CV pipeline — YOLOv8n detection + ByteTrack
tracking via `ultralytics`, dense optical flow via OpenCV) were built and
run against all 103 local videos (3 `test/` + 100 `train/`, the latter
sampled via `scripts/sample_train_videos.py`). **All 103 processed
successfully, 0 failures**, in well under a minute on this machine (M2,
MPS backend). Output: `data/scene_dataset_raw.csv` (raw per-video values)
and `data/scene_dataset.csv` (final bucketed schema, matching the table
below minus `risk_label`, which is Phase 3's job).

**How each variable actually got computed** (validated against 3 test
videos + spot-checked debug frames in `outputs/debug_frames/` before the
full run — boxes land on real people/vehicles, weather GT matches the
visual scene, brightness ordering tracks reality):

- `vehicle_speed`, `pedestrian_distance`, `visibility` — tertile-bucketed
  (low/medium/high or far/medium/close) from uncalibrated CV proxies:
  background optical-flow magnitude (objects masked out) for speed, max
  normalized person-box height for distance, brightness+contrast (combined
  as whichever is worse) for visibility.
- `num_pedestrians`, `traffic_density` — kept as raw numbers (distinct
  track IDs, mean per-frame vehicle count), not bucketed, per the
  reasoning in the implementation plan.
- `weather`, `road_type` — pulled directly from the dataset's real
  ground-truth annotations (`ground_truth.py`), not derived by CV.
- `obstacle` — a lightweight near-static-vehicle heuristic (lower
  priority, as planned).

**Known limitations found while validating real output** (worth stating
honestly in the report, not silently smoothing over):

- **`obstacle` over-triggers** — 84/103 videos got `"yes"`, which is too
  high to be a discriminating signal as currently thresholded. Needs
  revisiting (e.g. a stricter displacement threshold, or requiring more
  than 2 detections) before it's trustworthy — treat it as a stub for now,
  same as originally planned as acceptable for a first pass.
- **`num_pedestrians` can be noisy in crowded scenes** — spot-checked the
  highest outlier (32, in a real crowded pedestrian-crossing scene,
  confirmed via debug frame) and found real track-ID churn under the
  sparse 2fps sampling (the same handful of people getting reassigned new
  IDs across frames), which likely inflates the count somewhat above the
  true number. The true value is probably somewhere between the largest
  single-frame count and the distinct-ID count; this implementation uses
  the max of the two, which is a documented, not hidden, approximation.
- **`visibility`'s "low" bucket is oversized on purpose, not a bug** — it's
  the `min()` of two independent tertile rankings (brightness, contrast),
  which mathematically skews toward "low" (worked out to 57/33/13 for
  low/medium/high across the 103 videos, close to the theoretical 5/9,
  3/9, 1/9 split for two independent tertiles combined this way) — because
  a video only needs *either* poor brightness *or* poor contrast to be
  flagged low-visibility. Working as designed, but worth explaining in the
  report rather than presenting the class balance as if it were neutral.
- **Weather/road-type class balance is skewed** in this 103-video sample
  (76/103 "sunny day", 65/103 "arterials" road type) — a real property of
  which videos got randomly sampled, worth keeping in mind for Phase 3
  (a classifier trained on this will see much more "sunny day" than
  anything else).

100 training videos (25 each from CAP_DATA, DADA_2000, DoTA, MANUAL_DATA,
chosen with a fixed random seed, excluding the 3 already in `test/`) were
copied into `videos/train/<source>/` via `scripts/sample_train_videos.py`,
with a manifest at `data/train_video_manifest.csv`.

**What's already available and doesn't need building:** the real dataset
(see [`approach/03-dataset-source.md`](approach/03-dataset-source.md)) has
ground-truth answers for `weather and light`, `location`, `road type`,
`accident type`, `accident reason`, and `prevention method` for all 1,000
videos, plus a dense text caption per video. **Pull these from the
dataset's own JSON files directly — don't re-derive them with CV.**
Re-deriving a value CV would only approximate, when an authoritative label
already exists, is wasted effort and a worse result.

**What actually needs a CV pipeline** (not covered by the existing GT):

| Variable | Approach | Notes |
|---|---|---|
| Vehicle speed | Object tracking (ByteTrack/DeepSORT on YOLO detections), motion magnitude between frames | Getting a real km/h number needs camera calibration/homography, which is a nontrivial CV problem on its own. **Recommendation: don't chase exact km/h.** Derive a relative/qualitative speed proxy (e.g. bounding-box displacement rate, bucketed into low/medium/high) — that's all the downstream classifier needs, and it avoids a large, disproportionate sub-project. |
| Pedestrian distance | Same tracking output, using relative bounding-box position/scale as a proxy, or estimated ground-plane distance if a homography is set up anyway for speed | |
| Number of pedestrians | Direct count from YOLO detections (class = person) | Straightforward |
| Traffic density | Count of vehicle-class detections per frame, averaged | Straightforward |
| Obstacle | Detected static objects near the vehicle's path (e.g. "parked car") | Lower priority — the DAG in Phase 5 only uses it as a mediator for distance |

**Visibility** is a borderline case: the dataset's `weather and light` GT
(e.g. "rainy day", "stormy evening") already implies visibility, but a
direct brightness/contrast measurement from the frames is cheap and doesn't
need a model — worth computing as a supplementary signal even though a
weather-based proxy would work too.

**Output:** for each processed video, one row of: `vehicle_speed`,
`pedestrian_distance`, `num_pedestrians`, `weather` (from GT),
`visibility` (from GT + brightness check), `traffic_density`, `road_type`
(from GT), `obstacle`.

## Phase 2 — Structured storage

**Goal:** get every video's variables into one dataset, cleanly separated
from static metadata that doesn't vary per video.

**Status: done (2026-07-23).**

**`data/scene_dataset.csv`** — Phase 1's output (`video_id, source, split,
vehicle_speed, pedestrian_distance, num_pedestrians, weather, visibility,
traffic_density, road_type, obstacle`), now with three more columns added
by `mine_risk_labels.py`: `risk_score_raw`, `risk_label_source`, and
`risk_label` (the Phase 3 training target — see below for how it's built).
v1's `v1_legacy/data/scene_variables.csv` stays as-is, unrelated to this.

**`data/variable_metadata.csv`** — created, static, one row per variable:

```
variable,controllable
vehicle_speed,yes
pedestrian_distance,partially
visibility,partially
traffic_density,no
weather,no
road_type,no
obstacle,no
num_pedestrians,no
```

This directly feeds Phase 7's recommendation engine — it only searches over
rows marked `yes`/`partially`.

**How `risk_label` actually got built** (`mine_risk_labels.py`): rather than
guessing keyword lists, all 103 real dense captions were surveyed first.
Findings: 64/103 (62%) captions state an explicit impact speed ("...traveling
at approximately 40 km/h...", range 0–70 km/h) — a real, objective severity
signal, used as the primary basis for `risk_score_raw` when present. For the
other 39/103 (`risk_label_source = "cv_fallback"`), the score falls back to
Phase 1's CV-derived `vehicle_speed` bucket (low/medium/high → 0.2/0.5/0.9,
same convention as v1's `SPEED_SCORE`). The score is then adjusted ±0.15 for
sparse but real outcome language surveyed in the captions ("no apparent
evasive action" appears in 16/103; "remains motionless"/"serious"/"critical"
in 14/103 combined; "gets up"/"walks away" in 5/103), and tertile-bucketed
into `risk_label` (low/medium/high) the same way Phase 1 buckets its other
relative proxies.

**Known limitation, stated plainly rather than hidden:** for the 39
`cv_fallback` rows, `risk_label` is partly derived from the same
`vehicle_speed` feature that also goes into Phase 3 as a model input — a
mild leakage risk for that subset (the classifier could partly learn
"vehicle_speed predicts risk_label" close to by construction, for those
rows specifically). `risk_label_source` is kept as a visible column
specifically so this can be accounted for later (e.g. evaluating the
classifier on `text`-sourced rows only, as a cleaner check). This should be
named explicitly in the final report as a first-pass heuristic, not
presented as validated ground truth.

Real bug caught during validation: the initial speed-extraction regex
didn't handle decimal-formatted captions ("approximately 40.0 km/h",
used by some MANUAL_DATA entries) and silently matched just the ".0"
fragment as speed = 0 for 7 videos dataset-wide. Fixed and re-verified
before trusting the output.

## Phase 3 — Train a classifier

**Goal:** learn accident-risk from the Phase 1/2 data, instead of hand-
picking weights.

**Status:** not started. Phase 1/2 output is ready; supervisor confirmation
is in progress (not treated as blocking — see the note above).

**Model:** Random Forest or Gradient Boosting (XGBoost/LightGBM) — good
choices here specifically because both expose feature importances directly,
train fast on small-to-medium tabular data, and don't need GPU.

**Training target:** `risk_label` (low/medium/high), built in Phase 2 above
from dense-caption mining — see there for the method and its documented
leakage caveat for `cv_fallback` rows.

**Scale for the first pass:** the 103 videos already processed (not the
full 1,000) — enough for a Random Forest to find real signal, small enough
to iterate quickly. Scale up once the full Phase 1→4 pipeline runs cleanly
end to end.

**Output:** a trained, saved model, plus a feature-importance table/chart
(this replaces `DEFAULT_WEIGHTS` from v1's `v1_legacy/risk_model.py`).

## Phase 4 — Risk prediction

**Goal:** the trained classifier's output *is* the accident-risk KPI.

**Status:** not started (depends entirely on Phase 3).

`predict(scene)` → a risk class (Low/Medium/High) or a probability (e.g.
0.78). This replaces v1's `calculate_risk()` — same role in the pipeline
(scene variables in, one risk number out), different mechanism (learned
model instead of a fixed formula).

## Phase 5 — Causal DAG

**Goal:** encode the causal assumptions the classifier can't provide,
separately from it.

**Status:** not started. Doesn't depend on any other phase — this can
happen in parallel with Phases 1–4, since it's purely domain knowledge, not
derived from data.

Structure (as specified):

```
Weather → Visibility → Risk
Speed ────────────────→ Risk
Traffic Density → Speed
Pedestrian Distance → Risk
Obstacle → Distance
Road Type → Speed
```

This is a real upgrade over v1's DAG (which was a flat "4 variables all
point straight at Risk" structure) — this one has mediation paths (e.g.
Traffic Density affects Risk *through* Speed, not directly). Worth
representing as actual graph data (e.g. a small adjacency structure or a
`networkx` graph) rather than just a diagram, since Phase 7 needs to reason
about it programmatically (walking from a controllable variable to Risk
through its causal path).

**This DAG does not feed into Phase 4's number.** Its jobs are: (a)
documenting/explaining the assumed causal structure for the report, and (b)
giving Phase 7 a structure to reason over when deciding which interventions
make sense.

## Phase 6 — Interactive simulator

**Goal:** let a user edit a scene's variables and see the risk update live.

**Status:** not started (depends on Phase 4 — needs a trained classifier to
call).

Mechanism, exactly as specified: take the current scene, apply an edit
(e.g. `speed: 48 → 35`), call the *same* trained classifier's `predict()`
again, show the new score. **No new model, no DAG involved in the
recalculation** — it's a UI/re-prediction layer over Phase 4's model. Worth
being explicit about this in the report (see the methodological caveat in
[`00-overview.md`](00-overview.md)): this is correlational re-prediction,
not a causal `do()` operation, even though it's presented as "what if."

## Phase 7 — Recommendation engine

**Goal:** given a high-risk scene, suggest realistic changes that would
lower it — realistic meaning only changes to variables that are actually
controllable.

**Status:** not started (depends on Phases 2 (`variable_metadata.csv`), 4
(the classifier to score candidates), and ideally 5 (the DAG, to reason
about *which* controllable variables plausibly affect risk and how, rather
than searching blindly)).

Mechanism, exactly as specified: for the current scene, try changes to
controllable variables only (checked against `variable_metadata.csv`),
re-score each candidate via Phase 4's classifier, and surface the
combination(s) that reduce risk the most — e.g. "reduce speed to 30 km/h"
and "maintain 8m pedestrian distance," never "make the weather sunny."

---

## Suggested order of work

1. Phase 5 (DAG) can start immediately — no dependencies, low risk, and
   useful even if the ML direction changes after talking to the
   supervisor.
2. Phase 1 + 2 together, on the ~50–100 video subset — extraction and
   storage are useful regardless of what consumes them, so this is safe to
   build before the supervisor conversation resolves.
3. Confirm the ML direction with Jakob Suchan before or during Phase 3 —
   ideally before investing in Phase 3 model-tuning work, since the answer
   changes how the thesis frames what's being contributed.
4. Phase 3 → 4 → 6 → 7 in order, since each genuinely depends on the last.

## What happens to v1 if this all lands

v1 (`v1_legacy/risk_model.py` + `v1_legacy/prototype.py`, the hand-built
weighted formula) stays in the repo either way — it's a legitimate,
already-working baseline, and
"we started with a simple interpretable formula, then moved to a trained
classifier + separate causal DAG, and compared the two" is a stronger
thesis narrative than deleting the first attempt. Whether v1 gets an
explicit head-to-head comparison against the Phase 3 classifier in the
final report is worth deciding once Phase 3 actually produces results.
