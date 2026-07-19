# Next steps and future work

## Where things stand

The Step-1 draft (the thesis proposal document) has already been submitted
and accepted on Moodle. What's left is Step 2: turn the proposed prototype
into a real, working piece of software, use it to produce actual results,
write those results up, and submit the final thesis report via Turnitin.
This repo is the "real, working piece of software" part of that.

## Immediate next steps (do these first)

1. **Watch the 3 videos and replace the placeholder annotations** in
   `data/scene_variables.csv` with real, honest observations — see
   [`01-how-to-run-the-prototype.md`](01-how-to-run-the-prototype.md).
   Nothing downstream is meaningful until this is done.
2. **Re-run `prototype.py`** and sanity-check the results: do the numbers
   and charts match what you'd intuitively expect from watching the videos?
   If a "safe-looking" video scores High Risk, that's worth investigating —
   either the annotation or a weight might need adjusting.
3. **Write up the results** in the final report, using
   `outputs/risk_results.csv`, `outputs/intervention_results.csv`, and the
   chart PNGs as figures/tables. The `docs/` pages here can be adapted
   directly into the report's Prototype / Methodology sections — they
   already explain the reasoning in the same order the thesis draft does.

## A realistic roadmap after that (in priority order)

1. **Add a few more videos** (not all ~1,000 — maybe 5–10 more, if more
   VRU-Accident clips are available) so the results aren't based on only 3
   data points. Still fully manual annotation — no claim of large-scale
   evaluation.
2. **A sensitivity check**: re-run the risk model with a couple of
   different weight sets (e.g. equal weights for all 4 variables vs. the
   current weather-heavy weights) to show the conclusions aren't overly
   sensitive to the exact numbers chosen. Cheap to do, since weights are
   already a parameter in `risk_model.py` rather than hardcoded.
3. **A small interactive demo (optional/stretch)**: a simple slider-based
   UI (e.g. using Streamlit) built directly on top of the existing
   `risk_model.py` functions, so risk scores update live as variables are
   changed — good for a live demonstration during the thesis defense.

## What the thesis draft calls out explicitly as future work (beyond this prototype's scope)

These are documented in the draft as *out of scope for now* — useful to
reference if asked "what would you do differently with more time," but not
things to attempt in the current prototype:

- **Automated variable extraction**: instead of a person watching each
  video and typing in labels, use computer vision (object detection,
  tracking, brightness analysis) to estimate speed/visibility/proximity
  directly from the footage.
- **A richer causal model**: add more variables (traffic density, road
  layout, pedestrian behavior, vehicle trajectory) to make the model more
  realistic, at the cost of more complexity.
- **A more principled scoring method**: replace the simple weighted-sum
  formula with something more rigorous, like a Bayesian network or a
  properly learned structural causal model, instead of hand-picked weights.
- **Full-dataset evaluation**: run the pipeline across the full ~1,000-video
  VRU-Accident benchmark instead of a handful of hand-picked clips.
- **An interactive traffic-safety tool**: turn this from a research
  prototype into something an actual safety analyst or educator could use.

The thesis is explicit that none of this is required for the current scope
— the goal was always to demonstrate the *idea* of intervention-based,
interpretable risk reasoning on a small scale, not to build a
production-grade system.
