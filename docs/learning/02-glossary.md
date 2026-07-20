# Glossary (plain-language)

**VRU (Vulnerable Road User)** — a pedestrian, cyclist, or anyone on the road
who isn't protected by a vehicle's frame.

**VRU-Accident dataset** — a public research collection of 1,000 real
dashcam accident videos involving VRUs, with extra written descriptions
attached (what caused the accident, weather, road type, how it could have
been prevented, etc). A full local copy exists outside this project folder
— see [`../approach/03-dataset-source.md`](../approach/03-dataset-source.md).
v1 used just 3 of those videos with hand-guessed values; the new direction
targets ~50–100 videos using the dataset's real annotations.

**Ground truth (GT)** — the correct, verified answer for something, as
opposed to a guess or a model's prediction. The VRU-Accident dataset's
annotations are ground truth (human-verified); a classifier's predictions
are not.

**Scene variable** — one simple, human-readable fact about a video (e.g.
"speed was high"), used as an input to the risk model.

**KPI (Key Performance Indicator)** — a single number used to summarize how
something is doing. Here, the KPI is the accident-risk score (0 to 1).

**Causal graph / DAG (Directed Acyclic Graph)** — a diagram showing which
things are believed to influence which other things, drawn as boxes and
arrows with no loops. In this project, it's hand-drawn (weather, visibility,
speed, and proximity all point toward "accident risk") — not calculated
from data.

**Weighted score / weighted sum** — a way of combining several numbers into
one, where each number gets multiplied by a "weight" (how important it is)
before being added up. Used here to combine the 4 variables into 1 risk
score.

**Intervention** — the act of changing one variable to a different value
(while keeping everything else the same) and seeing what happens to the
outcome. Example: "what if speed had been low instead of high?" This is the
central idea borrowed from causal reasoning, without doing full causal
inference.

**Correlation vs. causation** — correlation is "these two things tend to
happen together"; causation is "one of these things actually makes the
other happen." Most machine-learning accident-detection systems only learn
correlations. This project's whole point is to explore the causation side,
even if only through a simplified, manually-built model.

**Risk band / risk level** — a plain label (Low / Moderate / High Risk)
attached to a risk score range, so numbers are easier to talk about than raw
decimals.

**Prototype** — an early, simplified working version of an idea, built to
demonstrate that it's feasible — not a finished or production-ready system.

**Interpretable** — able to be explained and understood by a person, step
by step, as opposed to a "black box" system whose internal reasoning is
opaque even to its own designers. The main selling point of this project.

**ML model (as opposed to a rule-based model)** — a system that *learns*
its behavior from examples (training data) rather than having it written
down by a person. v1 has no ML model in it — the risk formula is entirely
hand-written. The new direction (Phase 3) does train one. See
[`../approach/02-ml-model-and-technologies.md`](../approach/02-ml-model-and-technologies.md).

**Object detection (YOLO, Faster R-CNN, etc.)** — an ML technique for
finding and labeling things in an image (e.g. "there's a car here, a
pedestrian there"). Not used in v1; used in the new direction's Phase 1 to
extract variables like number of pedestrians and traffic density.

**Object tracking (ByteTrack, DeepSORT)** — following the *same* detected
object across multiple video frames (as opposed to detecting it fresh in
each frame with no memory of it being the same car/person as before).
Needed to estimate motion/speed, since that requires knowing how far the
*same* object moved between frames.

**Optical flow** — a technique for estimating motion between two video
frames at the pixel level (which direction and how fast things in the image
are moving), without needing to first detect specific objects. Used in
Phase 1 as a supplementary motion signal.

**Random Forest / Gradient Boosting (XGBoost, LightGBM)** — types of ML
models that make predictions by combining many simple decision trees. Good
fit for this project's Phase 3 classifier because they work well on small
tabular datasets, need no GPU, and directly expose feature importance.

**Feature importance** — a score a trained model (like a Random Forest) can
report per input variable, showing how much that variable contributed to
its predictions. **Not the same as a causal effect** — see
[`00-key-concepts.md`](00-key-concepts.md#feature-importance-is-not-a-causal-weight-read-this-before-phase-3)
for why that distinction matters here specifically.

**Proxy label / weak supervision** — using an indirect, imperfect signal as
a stand-in training target when the real thing you want to predict isn't
directly available. This project's planned severity label (Phase 3),
mined from keyword patterns in dense captions rather than a real
per-video risk rating, is an example.

**Controllable variable** — a scene variable a driver/system could actually
change (e.g. speed), as opposed to one nobody can control (e.g. weather).
Tracked per-variable in `variable_metadata.csv`, and used by the
recommendation engine (Phase 7) so it never suggests something impossible
like "make the weather sunny."

**Recommendation engine** — the Phase 7 component that searches over
controllable variables to find changes that would have lowered a scene's
predicted risk, then reports the best one(s) in plain language.

**Vision-language model / multimodal model** — an ML model that combines
image or video understanding with natural-language text, e.g. answering
questions about a video or generating captions for it. The VRU-Accident
dataset's own annotations were produced using this class of model; this
project uses those annotations directly rather than running one itself.

**Bayesian network** — a more statistically rigorous cousin of a causal
graph, where the strength of each connection is learned from data with
proper uncertainty, instead of hand-picked or derived from feature
importance. Listed in the thesis's Future Work as a further step beyond
even the new direction's DAG.
