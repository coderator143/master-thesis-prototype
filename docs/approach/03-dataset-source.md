# The real dataset source

## Where it is

`/Users/warshaw65/Desktop/VRU-Accident/` — a sibling folder to this
project, **not** inside `thesis-prototype/`. It's a full local clone of the
actual VRU-Accident benchmark's official GitHub repository (the paper: *"
VRU-Accident: A Vision-Language Benchmark for Video Question Answering and
Dense Captioning for Accident Scene Understanding,"* ICCVW 2025, Best
Student Paper Award — see the thesis's own bibliography, refs [1]/[2]). It
has its own `.git`, its own `.venv`, and its own `README.md`/`main.py` (the
paper authors' evaluation code for benchmarking vision-language models —
not something this project needs to run, just useful to know it's there).

**Keep it where it is — don't copy it into `thesis-prototype/`.** It's 3 GB
of video alone, it's already its own git repository, and this project's
`.gitignore` already excludes `videos/` for size reasons. Reference it by
path from code/config instead.

## What's in it

```
VRU-Accident/
├── VRU_videos/
│   ├── CAP_DATA/       287 videos
│   ├── DADA_2000/      223 videos
│   ├── DoTA/           100 videos
│   └── MANUAL_DATA/    390 videos      (1,000 total — matches the thesis's dataset description exactly)
└── MetaData/
    ├── CAP_DATA/{CAP_DATA_VQA_annotation.json, CAP_DATA_Dense_Caption.json}
    ├── DADA_2000/{...}
    ├── DoTA/{...}
    └── MANUAL_DATA/{...}
```

Each of the 4 sources has the same two annotation files, same schema.

### VQA annotation JSON — real ground truth, 6 categories per video

Keyed by video path (e.g. `./VRU_videos/CAP_DATA/VRU_9.mp4`). Example
entry:

```json
{
  "weather and light": {
    "question": "...", "options": "A. rainy day, B. sunny afternoon, ...", "GT": "A"
  },
  "location": { "...": "...", "GT": "A" },
  "road type": { "...": "...", "GT": "A" },
  "accident type": { "...": "...", "GT": "A" },
  "accident reason": { "...": "...", "GT": "C" },
  "prevention method": { "...": "...", "GT": "B" }
}
```

Six categories, each a 4-option multiple choice question with a `GT`
(ground truth) answer letter — this is exactly Table 4.1 from the thesis
draft (Weather and Light, Traffic Environment, Road Configuration, Accident
Type, Accident Cause, Accident Prevention Measure), just with `location`
standing in loosely for traffic environment. **Use these values directly —
don't re-derive weather/road-type/etc. with computer vision when a
human-verified answer already exists.**

### Dense caption JSON — one rich paragraph per video

```json
{ "gt": "A male cyclist wearing a white short-sleeved shirt... the cyclist is thrown onto the car's hood and windshield, then flung forward onto the road..." }
```

Free text, not structured — but the natural source for a **severity proxy**
(Phase 3's training target — see
[`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md)), since
phrases like "thrown," "flung," "critical," vs. "swerved in time," "minor
contact" carry real severity information that the structured VQA fields
don't capture.

### What's genuinely missing (needs Phase 1's CV pipeline)

Vehicle speed, pedestrian distance, number of pedestrians, and traffic
density are **not** in either annotation file — they require actually
looking at the video frames (object detection + tracking), which is what
Phase 1 is for. See the roadmap for the specific approach per variable.

## Our 3 existing videos are already in here with real GT

`VRU_9`, `VRU_10`, and `VRU_14` (in `videos/test/CAP_DATA/`, with
placeholder guesses in `data/scene_variables.csv`) are `CAP_DATA` entries in
this dataset, with real GT annotations and real dense captions already
available. When Phase 1/2 work begins, these three should be re-populated
from real data rather than kept as hand-guessed placeholders — no reason to
guess when the answer is sitting right there.

**Important correction, and why the folder layout has a `source`
subfolder:** video IDs are only unique *within* a source, not across the
whole dataset — there's a different `VRU_9.mp4`, `VRU_10.mp4`, and
`VRU_14.mp4` in each of CAP_DATA, DADA_2000, DoTA, and MANUAL_DATA (4
different files per name, confirmed by differing file sizes). An earlier
version of this doc incorrectly said our 3 videos were from MANUAL_DATA —
they're actually CAP_DATA, confirmed by matching file sizes byte-for-byte.
**Always keep `source` alongside `video_id` when identifying a video** —
`video_id` alone is ambiguous. This is why `videos/test/` and
`videos/train/` are organized as `{split}/{source}/{video_id}.mp4`, and why
`data/scene_variables.csv` and the planned `data/scene_dataset.csv` both
have a `source` column.

## `Model_Response/` and `Models/` — not ground truth, worth knowing about

The repo also has a `Model_Response/` folder with outputs from various
vision-language models (Gemini 1.5 Flash, several InternVL variants, etc.)
run against the dataset's dense-captioning task, plus a `Models/` and `src/`
with the benchmark authors' own evaluation code. These are **other
researchers' model predictions**, not ground truth — useful only as a
reference point for "how well do existing models do at this task," not as
a data source for this project's own pipeline.

## Practical note for Phase 1 implementation

When Phase 1 code gets written, point it at this folder via a config
value/environment variable (e.g. `VRU_ACCIDENT_ROOT =
/Users/warshaw65/Desktop/VRU-Accident`) rather than a hardcoded relative
path — keeps the two repositories cleanly separated and makes the path
easy to change if the dataset ever moves.
