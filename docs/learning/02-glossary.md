# Glossary (plain-language)

**VRU (Vulnerable Road User)** — a pedestrian, cyclist, or anyone on the road
who isn't protected by a vehicle's frame.

**VRU-Accident dataset** — a public research collection of ~1,000 real
dashcam accident videos involving VRUs, with extra written descriptions
attached (what caused the accident, weather, road type, how it could have
been prevented, etc). This prototype uses just 3 of those videos.

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
down by a person. This prototype has no ML model in it — the risk formula
is entirely hand-written. See
[`../approach/02-ml-model-and-technologies.md`](../approach/02-ml-model-and-technologies.md).

**Object detection (YOLO, Faster R-CNN, etc.)** — an ML technique for
finding and labeling things in an image (e.g. "there's a car here, a
pedestrian there"). Mentioned in the thesis's Related Works and Future Work
as a plausible way to automate variable extraction later — not used in this
prototype.

**Vision-language model / multimodal model** — an ML model that combines
image or video understanding with natural-language text, e.g. answering
questions about a video or generating captions for it. The VRU-Accident
dataset's own annotations were produced using this class of model; this
prototype does not run one itself.

**Bayesian network** — a more statistically rigorous cousin of the simple
causal graph used here, where the strength of each connection is learned
from data with proper uncertainty, instead of hand-picked. Listed in the
thesis's Future Work as a natural next step beyond this prototype's fixed
weighted-sum formula.
