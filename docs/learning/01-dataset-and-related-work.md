# The dataset, and how this thesis compares to related work

## The VRU-Accident dataset

VRU-Accident is a public research benchmark of roughly 1,000 real dashcam
videos of traffic accidents involving vulnerable road users (VRUs) —
pedestrians and cyclists. It was built for a different purpose than this
thesis originally: evaluating AI systems on video question answering and
dense captioning (i.e., can a model watch a crash and answer questions
about it, or describe it in detail, the way a person could).

What makes it useful for this thesis specifically is that each video comes
with structured, human-written information, not just raw footage:

| Category | What it captures |
|---|---|
| Weather and Light | Weather condition and lighting information |
| Traffic Environment | Traffic density and surrounding environment |
| Road Configuration | Road structure and lane conditions |
| Accident Type | Type of collision or accident |
| Accident Cause | Main cause contributing to the accident |
| Accident Prevention Measure | Possible preventive actions |

Plus dense captions describing vehicle/pedestrian behavior, spatial
relationships, and collision details in prose.

This prototype only uses **3 of the ~1,000 videos** (`VRU_9`, `VRU_10`,
`VRU_14`) — see
[`../02-next-steps-and-future-work.md`](../02-next-steps-and-future-work.md)
for why, and what expanding that would involve. Rather than pulling the
dataset's own annotation categories in automatically, this prototype's
4 variables (speed, visibility, proximity, weather) are a hand-picked,
simplified subset chosen for their direct relevance to a risk formula — a
person watches the actual video and annotates them, rather than parsing the
dataset's existing text annotations programmatically.

## Where this thesis sits relative to other research

The thesis's Related Works chapter groups existing traffic-safety AI
research into three buckets, and positions this project against all three:

1. **Object detection / tracking** (YOLO, Faster R-CNN, transformer-based
   vision models) — these are very good at finding and classifying cars,
   pedestrians, and cyclists in a frame, but they answer "what's in this
   scene," not "what would happen if this scene were different." This
   thesis doesn't compete with these models — it's a different kind of
   question entirely.
2. **Accident detection / prediction** (dashcam or CCTV footage → a
   danger score or "accident about to happen" flag) — typically trained
   deep learning models that work well but are black boxes: you get a
   number, not a reason. This thesis exists partly as a contrast to this
   category specifically — trading raw predictive accuracy for
   explainability.
3. **Visual commonsense reasoning / multimodal learning** (models that
   combine vision with language to describe *and* reason about a scene,
   which is what the VRU-Accident benchmark itself is designed to
   evaluate) — this is the category the dataset comes from, but this
   thesis uses the dataset's *idea* (rich, structured scene description)
   without using the multimodal models that power it.

Separately, causal reasoning as a field (Judea Pearl's causal graphs and
do-calculus are the classic reference, cited in the thesis) is well
established in areas like healthcare and economics, but the thesis notes
it's rarely applied to traffic-scene understanding specifically — that
combination (structured visual-scene data + lightweight causal-style
reasoning) is the gap the thesis is aiming at, even at prototype scale.

## In short

This project isn't trying to out-perform object detectors or deep
accident-prediction models on accuracy — it's demonstrating a different
capability (interpretable "what if" reasoning) that those approaches don't
offer, using a small, manually curated slice of a dataset that was built
for a related but different purpose.
