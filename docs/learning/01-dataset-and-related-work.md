# The dataset, and how this thesis compares to related work

## The VRU-Accident dataset

VRU-Accident is a public research benchmark of 1,000 real dashcam videos of
traffic accidents involving vulnerable road users (VRUs) — pedestrians and
cyclists. It was built for a different purpose than this thesis originally:
evaluating AI systems on video question answering and dense captioning
(i.e., can a model watch a crash and answer questions about it, or describe
it in detail, the way a person could).

**A full local copy of the actual dataset exists** at
`/Users/warshaw65/Desktop/VRU-Accident/` — all 1,000 videos, plus real
ground-truth annotations, not just the video files. See
[`../approach/03-dataset-source.md`](../approach/03-dataset-source.md) for
the exact folder layout, file schema, and how the project plans to use it.
In short, each video comes with:

| Category | What it captures |
|---|---|
| Weather and Light | Weather condition and lighting information |
| Traffic Environment | Traffic density and surrounding environment (called "location" in the raw annotation file) |
| Road Configuration | Road structure and lane conditions |
| Accident Type | Type of collision or accident |
| Accident Cause | Main cause contributing to the accident (called "accident reason" in the raw annotation file) |
| Accident Prevention Measure | Possible preventive actions |

— each as a real, human-verified multiple-choice ground-truth answer, not
just a category description — plus a dense free-text caption per video
describing behavior, spatial relationships, and collision details.

**v1** (the hand-built formula) only used 3 of the 1,000 videos (`VRU_9`,
`VRU_10`, `VRU_14`), with hand-guessed placeholder values for 4 simplified
variables (speed, visibility, proximity, weather) rather than pulling in
the dataset's real annotations — see
[`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md) for why that's
changing: the new direction uses the dataset's real ground truth directly
for the variables it already covers, and only relies on computer vision
(Phase 1) for the handful of variables (speed, pedestrian distance/count,
traffic density) that genuinely aren't in the annotations. The first
end-to-end pass targets ~50–100 videos, not the full 1,000, to keep
iteration fast.

## Where this thesis sits relative to other research

The thesis's Related Works chapter groups existing traffic-safety AI
research into three buckets. The new direction changes how this project
sits relative to two of them — worth being clear-eyed about, since it
affects how the final report should frame the contribution:

1. **Object detection / tracking** (YOLO, Faster R-CNN, transformer-based
   vision models) — good at finding and classifying cars, pedestrians, and
   cyclists in a frame. v1 didn't use these at all. **The new direction
   does** — Phase 1 uses exactly this category of model (YOLO + a tracker)
   as a feature-extraction front end. That's not a contradiction: this
   project still isn't trying to out-perform detection models at detection
   — it's using them as an input stage for a different downstream question
   ("what would happen if this scene were different"), the same way any
   pipeline uses off-the-shelf components for a subtask that isn't its
   novel contribution.
2. **Accident detection / prediction** (dashcam/CCTV footage → a danger
   score, typically a black-box deep learning model) — v1 positioned itself
   explicitly *against* this category, trading accuracy for
   interpretability. **The new direction's classifier (Phase 3) is,
   mechanically, an instance of this category** — a trained model
   predicting a risk score from features. What keeps the thesis's original
   contrast meaningful is *not* pretending the classifier is something
   other than correlational, but keeping the causal DAG (Phase 5) as a
   separate, genuinely different, hand-built component that this category
   of prior work doesn't have — see the "feature importance is not a
   causal weight" section in
   [`00-key-concepts.md`](00-key-concepts.md) for why this distinction has
   to stay sharp in the writing.
3. **Visual commonsense reasoning / multimodal learning** (models combining
   vision with language, which is what the VRU-Accident benchmark itself
   was built to evaluate) — neither v1 nor the new direction uses this
   category of model; the dataset's *annotations* (produced using this
   class of model, by the benchmark's original authors) are used as
   pre-computed ground truth instead.

Separately, causal reasoning as a field (Judea Pearl's causal graphs and
do-calculus are the classic reference, cited in the thesis) is well
established in areas like healthcare and economics, but the thesis notes
it's rarely applied to traffic-scene understanding specifically — that gap
is still the thing this project is aiming at; the new direction just
pursues it with a hybrid (trained classifier + separate causal DAG) instead
of a single hand-built formula.

## In short

v1 avoided ML entirely to keep a clean methodological contrast with
correlation-based prior work. The new direction reintroduces ML (both for
feature extraction and for risk prediction) but tries to preserve the
thesis's original contribution by keeping the causal claims in a separate,
still-hand-built component (the DAG) rather than asking the trained model
to carry both jobs. Whether that's a convincing enough distinction to
preserve the thesis's argument is worth discussing directly with the
supervisor — see
[`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md).
