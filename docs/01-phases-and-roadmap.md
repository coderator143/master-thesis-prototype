# Phases and roadmap

This is the authoritative plan. Each phase is worked on in order — nothing
in a later phase should get built before the phase(s) it depends on are
actually done, since each one's output is the next one's input.

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Planning: architecture defined, dataset located, docs written | ✅ Done (2026-07-20) |
| 1 | Visual feature extraction | ✅ Done (2026-07-20, expanded 2026-07-23) — 303/303 videos processed, 0 failures, 11 variables |
| 2 | Structured storage (`scene_dataset.csv` + `variable_metadata.csv`) | ✅ Done (2026-07-23) — `risk_label` mined from dense captions, `variable_metadata.csv` created |
| 3 | Train a classifier | ✅ Done (2026-07-23) — trained, but honest eval shows no clear signal beyond baseline once leakage is isolated (39.1% text-only vs. 41.6% baseline) — see Phase 3 |
| 4 | Risk prediction | 🟨 Model saved (`classifier.joblib`) but not yet wired into a `predict(scene)` interface — and per Phase 3's result, needs a decision on the training-label problem before this is worth building on |
| 5 | Causal DAG | ⬜ Not started — structure specified below, not yet built as code/data |
| 6 | Interactive simulator | ⬜ Not started — depends on Phase 4 |
| 7 | Recommendation engine | ⬜ Not started — depends on Phases 2, 4, 5 |

**Overall status:** Phases 1–3 done (Phase 3's honest result needs a
decision before Phase 4 is worth building on — see Phase 3), Phases 4–7
not started. v1 (the
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

**Status: done, expanded (2026-07-23).** `ground_truth.py` (GT
loading/parsing) and `extract_features.py` (the CV pipeline — YOLOv8n
detection + ByteTrack tracking via `ultralytics`, dense optical flow via
OpenCV) were built and run against **303 local videos** (3 `test/` + 300
`train/`, up from an initial 103, expanded before Phase 3 for a healthier
samples-per-feature ratio once more variables were added — see below).
**All 303 processed successfully, 0 failures**, in a couple of minutes on
this machine (M2, MPS backend). Output: `data/scene_dataset_raw.csv` (raw
per-video values) and `data/scene_dataset.csv` (final bucketed schema, 17
columns including `risk_label` from Phase 2/3's mining step).

**11 scene variables** (up from the original 8 — see "Added before Phase 3"
below), plus the `risk_label` training target:

- `vehicle_speed`, `pedestrian_distance`, `closing_risk`, `visibility` —
  tertile-bucketed (low/medium/high, or far/medium/close for distance) from
  uncalibrated CV proxies: background optical-flow magnitude (objects
  masked out) for speed, max normalized person-box height for distance,
  their product for closing_risk, brightness+contrast (combined as
  whichever is worse) for visibility.
- `num_pedestrians`, `traffic_density` — kept as raw numbers (distinct
  track IDs, mean per-frame vehicle count), not bucketed.
- `weather`, `road_type`, `location` — pulled directly from the dataset's
  real ground-truth annotations (`ground_truth.py`), not derived by CV.
- `obstacle`, `braking` — lightweight heuristics from the same tracking/flow
  data already computed for the variables above (no extra CV work).

**Added before Phase 3, at the user's request** (see the discussion around
2026-07-23 — "more variables" was pushed back on as a blanket instinct,
since more columns without more rows raises overfitting risk at small
sample sizes, but three specific additions were agreed as genuinely
low-cost and non-redundant, alongside expanding the video count):

- **`location`** (rural/urban/suburban/...) — was already sitting unused in
  the same GT JSON `weather`/`road_type` are pulled from; zero new
  extraction work, just one more field read.
- **`closing_risk`** — `pedestrian_distance_raw × vehicle_speed_raw`, a
  proxy interaction combining ego speed and how close the nearest
  pedestrian got. Not real seconds-to-collision (neither input is
  calibrated to physical units), but the kind of interaction a Random
  Forest won't reliably reconstruct on its own from the two variables
  sitting in separate columns.
- **`braking`** — whether the optical-flow magnitude (the same series
  `vehicle_speed` is derived from) trends downward across the clip
  (second-half mean ≥15% below first-half mean), as a proxy for the ego
  vehicle decelerating before the collision. **Worth flagging a conceptual
  difference from the others**: this describes the driver's *response*
  during the clip, not a fixed pre-existing scene condition the way
  weather or road type are — closer in kind (though not identical) to the
  `accident_type`/`accident_reason` fields that were deliberately excluded
  as circular. Kept because it's an independently observable physical
  signal (not a retrospective annotator judgment like those fields), but
  the report should be honest that it sits closer to "response" than
  "precondition."

**Known limitations found while validating real output** (worth stating
honestly in the report, not silently smoothing over):

- **`obstacle` over-triggers** — 253/303 (83%) got `"yes"`, consistent with
  the same ~81-84% rate seen on the earlier 103-video sample, confirming
  this is a systematic issue with the threshold, not sampling noise. Needs
  a stricter displacement threshold (or requiring more detections) before
  it's a useful discriminating signal — treat it as a stub for now.
- **`num_pedestrians` can be noisy in crowded scenes** — spot-checked the
  highest outlier in the original 103 (32, a real crowded pedestrian
  crossing, confirmed via debug frame) and found real track-ID churn under
  sparse 2fps sampling likely inflates the count somewhat above the true
  number. Documented approximation (max of distinct-ID count and largest
  single-frame count), not hidden.
- **`visibility`'s "low" bucket is oversized on purpose, not a bug** — it's
  the `min()` of two independent tertile rankings (brightness, contrast),
  which mathematically skews toward "low" (147/112/44 for low/medium/high
  across all 303 videos, close to the theoretical 5/9, 3/9, 1/9 split) —
  because a video only needs *either* poor brightness *or* poor contrast
  to be flagged low-visibility.
- **`location` is heavily imbalanced** — 273/303 (90%) came back "urban",
  with rural (25), suburban (2), highway (2), and mountain (1) barely
  represented. A classifier will have essentially no signal to learn about
  the rare categories; worth treating `location` as close to a binary
  urban/non-urban signal in practice, and saying so in the report rather
  than implying it's a balanced 4+-way categorical feature.
- **Weather/road-type class balance is skewed**, consistent with the
  earlier 103-video finding — this is a property of which real accidents
  exist in the dataset, not sampling bias specifically.
- **`risk_label`'s tertile split isn't perfectly even** (126 low / 101
  medium / 76 high, not ~101 each) — because `risk_score_raw` clusters at
  only three discrete base values (0.2/0.5/0.8) for the 101 `cv_fallback`
  rows before adjustment, which prevents a clean three-way quantile split
  when there are many tied values. Not a bug, but worth knowing when
  interpreting Phase 3's training-label balance.

300 training videos (75 each from CAP_DATA, DADA_2000, DoTA, MANUAL_DATA —
up from 25 each — chosen with a fixed random seed, excluding the 3 already
in `test/`) were copied into `videos/train/<source>/` via
`scripts/sample_train_videos.py`, which is incremental/idempotent (re-run
after raising the per-source target and it only tops up what's missing,
never re-copies or invalidates already-processed videos). Manifest at
`data/train_video_manifest.csv`.

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

(This table is the original plan, 8 variables. `location`, `closing_risk`,
and `braking` were added afterward, before Phase 3 — see "Added before
Phase 3" above for what they are and why.)

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
traffic_density, road_type, obstacle`, later expanded to 11 variables —
`closing_risk`, `location`, `braking` added before Phase 3, see Phase 1),
now with three more columns added by `mine_risk_labels.py`:
`risk_score_raw`, `risk_label_source`, and `risk_label` (the Phase 3
training target — see below for how it's built). v1's
`v1_legacy/data/scene_variables.csv` stays as-is, unrelated to this.

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
location,no
braking,yes
```

**Corrected after review (2026-07-23):** two changes from the first version
of this table, both user catches:

- **`braking` is `yes`, not `no`.** Original reasoning was that it
  describes a specific past clip's driver response rather than a scene
  precondition, so it seemed odd to list as a lever the same way speed is.
  But that reasoning missed the actual point of Phase 7: "brake / reduce
  speed in time" is a completely standard, real, actionable safety
  recommendation — arguably a more direct one than the scene-average
  `vehicle_speed`. Corrected.
- **`closing_risk` is dropped from this table entirely**, not just marked
  `no`. It's a derived interaction (`pedestrian_distance_raw ×
  vehicle_speed_raw`), not an independent scene variable — the earlier
  version's instinct that it wasn't a real intervention target was right,
  but a derived variable doesn't belong in a table of *independent*
  variables and their controllability at all. It still stays as a Phase 3
  input feature in `data/scene_dataset.csv` (harmless there, and may help
  a small Random Forest capture the speed×proximity interaction a bit more
  directly), it's just not something Phase 7 reasons about as a
  controllability question — that question only makes sense for
  `vehicle_speed` and `pedestrian_distance`, which it's already asked
  about directly.

This directly feeds Phase 7's recommendation engine — it only searches over
rows marked `yes`/`partially`.

**How `risk_label` actually got built** (`mine_risk_labels.py`): rather than
guessing keyword lists, the initial 103 real dense captions were surveyed
first. Findings: ~62% of captions state an explicit impact speed
("...traveling at approximately 40 km/h...", range 0–70 km/h) — a real,
objective severity signal, used as the primary basis for `risk_score_raw`
when present. For the rest (`risk_label_source = "cv_fallback"`), the score
falls back to Phase 1's CV-derived `vehicle_speed` bucket (low/medium/high
→ 0.2/0.5/0.9, same convention as v1's `SPEED_SCORE`). The score is then
adjusted ±0.15 for sparse but real outcome language found in the survey
("no apparent evasive action", "remains motionless"/"serious"/"critical",
"gets up"/"walks away"), and tertile-bucketed into `risk_label`
(low/medium/high). Re-run as-is (same method, no changes) after the video
count expanded to 303: **202/303 (67%) text-sourced, 101/303
cv_fallback** — consistent with the original survey's ~62%/38% split.
Final distribution: 126 low / 101 medium / 76 high (see the tertile-split
unevenness note under Phase 1 for why it's not closer to 101/101/101).

**Known limitation, stated plainly rather than hidden — updated after Phase
3 measured it directly (see below): this is not a "mild" risk, it's
substantial.** For the 101 `cv_fallback` rows, `risk_label` is partly
derived from the same `vehicle_speed` feature that also goes into Phase 3
as a model input. Phase 3's evaluation measured the actual size of this
effect rather than just flagging it as a possibility: classifier accuracy
on `cv_fallback`-only rows is 84%, on `text`-only rows is 39% (below the
41.6% majority-class baseline). `risk_label_source` is kept as a visible
column specifically so this could be measured. This should be named
explicitly and prominently in the final report as a first-pass heuristic,
not presented as validated ground truth — see Phase 3 for the full
implication.

Real bug caught during validation: the initial speed-extraction regex
didn't handle decimal-formatted captions ("approximately 40.0 km/h",
used by some MANUAL_DATA entries) and silently matched just the ".0"
fragment as speed = 0 for 7 videos dataset-wide. Fixed and re-verified
before trusting the output.

## Phase 3 — Train a classifier

**Goal:** learn accident-risk from the Phase 1/2 data, instead of hand-
picking weights.

**Status: done, and it surfaced an important honest result (2026-07-23).**
`train_classifier.py` trains a `RandomForestClassifier` (500 trees,
`min_samples_leaf=3`, `class_weight="balanced"`) on the 11 scene variables
via a `ColumnTransformer` + `OneHotEncoder(min_frequency=5)` pipeline,
evaluated with pooled 5-fold `cross_val_predict` (one out-of-fold
prediction per row, not an average of 5 small reports), plus a
`RepeatedStratifiedKFold` stability check. Full details/reasoning in the
module docstring and code comments.

**The headline result — read this before citing an accuracy number
anywhere:**

| Evaluation slice | Accuracy | Rows |
|---|---|---|
| Baseline (majority-class guess) | 41.6% | 303 |
| ALL (everything pooled) | 54.1% | 303 |
| `cv_fallback`-only | **84.2%** | 101 |
| `text`-only | **39.1%** | 202 |

Repeated-CV stability: 53.6% ± 6.5% (5-fold × 5 repeats) — the pooled "ALL"
number is stable, not a shuffle artifact.

**What this means, stated plainly:** the 84.2% `cv_fallback` accuracy is
close to circular — for those 101 rows, `risk_label`'s severity score was
literally built from the same `vehicle_speed` bucket that's also a model
input (Phase 2's documented leakage caveat), so the classifier doing well
there mostly confirms the label-construction mechanism, not genuine
learning. The **39.1% `text`-only accuracy is the honest number** — labels
built from real caption text, evaluated on a model that never saw those
labels during that fold's training — and it's *below* the 41.6% baseline.
**On the current evidence, this classifier has not demonstrated
generalizable predictive power beyond guessing the majority class**, once
the construction-leakage effect is isolated. The "ALL" 54.1% figure, if
quoted without the `text`-only breakdown next to it, would overstate what
was actually learned — this needs to be reported with the full table
above, not just the flattering headline number.

**Validated before trusting this conclusion**, not just computed once and
accepted: 5-fold-stable top feature ranking (`vehicle_speed`,
`traffic_density`, `closing_risk` dominate every fold — consistent with
the leakage story, not a contradiction of it), a manual spot-check of
correct/incorrect `text`-only predictions (`outputs/classifier/cv_predictions.csv`
cross-referenced against `data/scene_dataset.csv` — errors look like
genuine model uncertainty, not corrupted data or a pipeline bug), and the
known-noisy features (`obstacle`, `location`) correctly rank near the
bottom of importance rather than suspiciously high.

**Feature importance (final model, fit on all 303 rows, sums to 1.0):**

| Variable | Importance |
|---|---|
| `vehicle_speed` | 0.190 |
| `traffic_density` | 0.187 |
| `closing_risk` | 0.106 |
| `num_pedestrians` | 0.093 |
| `pedestrian_distance` | 0.088 |
| `weather` | 0.075 |
| `visibility` | 0.066 |
| `road_type` | 0.065 |
| `braking` | 0.064 |
| `obstacle` | 0.042 |
| `location` | 0.023 |

`vehicle_speed` and `closing_risk` ranking highest is expected given how
`risk_label` was built (67% of labels from explicit caption speed, the
other 33% from `vehicle_speed` itself) — this table is at least as much a
reflection of the label-construction mechanism as of genuine risk factors,
which is exactly why it shouldn't be presented as "the model discovered
these variables matter" without the caveat above attached.

**What this is genuinely useful for, despite the result:** the thesis's
own argument is that trained classifiers can look like they're learning
something (a headline accuracy, a feature-importance ranking) while
actually reflecting artifacts of how the data was built — this result *is*
a real demonstration of exactly that risk, caught by the same
leakage-aware evaluation methodology this project has used throughout
rather than glossed over. That's a legitimate, honest finding for the
report, not a failed phase.

**Recommended next steps** (not yet implemented, pending direction):
1. A cleaner training target — e.g. restricting training to `text`-only
   rows entirely (202 rows, smaller but leakage-free), or a genuinely
   independent severity signal not derived from any input feature.
2. More `text`-sourced videos specifically (not just more videos overall —
   sampling more from sources/videos where captions state an explicit
   speed) to grow the leakage-free subset.
3. Treat this as the causal-DAG argument in miniature: this is precisely
   why Phase 5's hand-built DAG needs to stay analytically separate from
   this classifier's output, not deferred to it.

**Output (already produced):** `outputs/classifier/classifier.joblib` (the
final fitted pipeline, ready for Phase 4), `feature_importance.csv`,
`cv_predictions.csv`, `evaluation_report.txt` (full classification reports
and confusion matrices for all four slices above).

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
