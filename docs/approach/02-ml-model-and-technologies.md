# ML model, approach, and technologies used

## ML model used: none, and that's intentional

It's worth saying plainly: **this prototype does not train or run any
machine learning model.** There's no neural network, no object detector, no
vision-language model anywhere in `risk_model.py` or `prototype.py`. The
"model" in "risk model" means a simple mathematical formula (a weighted
sum), not a trained ML model.

This isn't a shortcut or a missing piece — it's exactly what the thesis
proposes. Straight from the abstract: *"Instead of training large
multimodal models, the proposed system focuses on manually defining
interpretable variables."* The scene variables (speed, visibility,
proximity, weather) are assigned by a person watching the video, not
predicted by a model. The thesis's whole argument is that you can get
useful, explainable "what if" reasoning without an ML model in the loop at
all — see
[`../learning/00-key-concepts.md`](../learning/00-key-concepts.md) for why
that trade-off is the actual point of the thesis, not a limitation of this
implementation.

### If automated variable extraction were added later (not implemented here)

The thesis's own Related Works chapter and Future Work section point at
what *would* be used if the manual-annotation step were automated — this is
listed here for reference only, none of it exists in this codebase:

- **Object detection / tracking models** (YOLO, Faster R-CNN,
  transformer-based vision architectures) — could estimate vehicle speed
  and pedestrian proximity directly from video frames instead of a person
  eyeballing them.
- **Vision-language / multimodal models** — the original VRU-Accident
  benchmark paper itself uses this class of model for its video-question-
  answering and dense-captioning annotations. This prototype consumes the
  *idea* of that structured information (weather, visibility, etc.) but not
  the model that produced it — those variables are typed in by hand here.
- **Brightness/contrast analysis** — a plausible lightweight (non-ML) way to
  estimate the `visibility` variable automatically, mentioned here because
  it's a much smaller step than full object detection and could be a
  realistic first automation target.

## The approach used

In one sentence: **rule-based interpretable scoring, not statistical or
learned causal discovery.** Four categorical variables are converted to
numbers, combined with fixed, human-chosen weights into a single risk
score, and "interventions" are simulated by changing one variable and
recomputing the score. The full mechanics — the exact formula, the weights,
a worked example, and how interventions work — are in
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md).

This is deliberately the simplest possible version of "causal-style"
reasoning: a real causal model would *discover* which variables matter and
by how much from data; this prototype has a person *assert* it up front.
The thesis is explicit that automatic causal discovery is out of scope.

## Technologies used

The stack is intentionally minimal — a thesis prototype demonstrating a
concept doesn't need a large dependency footprint, and a smaller stack is
easier to explain in a viva:

| Technology | Used for |
|---|---|
| Python 3.13 | The only language used, for everything |
| pandas | Reading `data/scene_variables.csv`, building result tables, writing `outputs/*.csv` |
| matplotlib | Generating the static PNG bar charts saved to `outputs/` |
| CSV files | The entire data storage layer — no database, since the dataset is 3 rows |
| `venv` | Isolating the two dependencies above from the system Python install |
| git | Version-controlling the code and docs (not the video files — see the root `.gitignore`) |

Notably absent, on purpose: any ML/CV framework (PyTorch, TensorFlow,
OpenCV), any web framework, and any database — none of them are needed for
what this prototype actually does, and adding them would be complexity
without a corresponding requirement. If the "automated extraction" future
work from the previous section is ever pursued, that's the point at which
OpenCV or a detection model would become a real dependency, not before.

## How it all ties together with the thesis

The thesis's central claim is that **interpretable, intervention-based
reasoning is a valuable alternative to black-box ML prediction** for
understanding accident risk. Every technology choice above supports that
claim directly: no ML model to keep opaque, a formula simple enough to
compute by hand (see the worked example in
[`01-risk-model-and-intervention-logic.md`](01-risk-model-and-intervention-logic.md)),
and outputs (CSVs, plain bar charts) chosen for how easy they are to read
and explain rather than for visual sophistication. See
[`00-approach-overview.md`](00-approach-overview.md) for the full
chapter-by-chapter mapping between the thesis text and this codebase.
