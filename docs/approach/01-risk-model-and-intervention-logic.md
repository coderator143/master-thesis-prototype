# How the risk model works (v1 — legacy baseline)

> **v1, superseded by the phased plan.** This describes the original
> hand-built formula (`v1_legacy/risk_model.py`), still working, still the baseline
> the new direction is compared against. It is *not* where the causal DAG
> or the trained classifier described in
> [`../01-phases-and-roadmap.md`](../01-phases-and-roadmap.md) live — see
> [`02-ml-model-and-technologies.md`](02-ml-model-and-technologies.md) for
> those. Kept here, unmodified, as the documented baseline.

Part of the [approach used in this prototype](00-approach-overview.md) —
this page is the detailed mechanics; that page is the high-level "why."

## The four variables

| Variable | What it means | Possible values |
|---|---|---|
| Speed | How fast the vehicle was going | `low`, `medium`, `high` |
| Visibility | How clear the scene was to see | `high`, `medium`, `low` |
| Proximity | How close the vehicle was to the pedestrian | `far`, `medium`, `close` |
| Weather | The environmental condition | `clear`, `rainy`, `night` |

## Step 1 — turn each label into a number

Each value gets turned into a number between 0.2 (safest) and 0.9
(riskiest):

```
speed:      low → 0.2     medium → 0.5     high → 0.9
visibility: high → 0.2    medium → 0.5     low → 0.9
proximity:  far → 0.2     medium → 0.5     close → 0.9
weather:    clear → 0.2   rainy → 0.5      night → 0.9
```

Notice visibility and proximity are "reversed" on purpose — `high`
visibility is *safe* (0.2), but `high` speed is *risky* (0.9). Each mapping
just encodes "which end of this variable is dangerous."

## Step 2 — combine them with weights

Not all four variables matter equally, so each one gets a "weight" — a
percentage of how much it counts toward the final score. The four weights
always add up to 1.0 (100%):

| Variable | Weight |
|---|---|
| Weather | 0.35 (35%) |
| Visibility | 0.25 (25%) |
| Proximity | 0.25 (25%) |
| Speed | 0.15 (15%) |

The final risk score is just:

```
risk score = 0.35 × weather_number
           + 0.25 × visibility_number
           + 0.25 × proximity_number
           + 0.15 × speed_number
```

This always comes out between 0 and 1.

### Worked example

Say a video is: speed = `high` (0.9), visibility = `medium` (0.5), proximity
= `close` (0.9), weather = `clear` (0.2).

```
risk = 0.35×0.2 + 0.25×0.5 + 0.25×0.9 + 0.15×0.9
     = 0.07     + 0.125    + 0.225    + 0.135
     = 0.555  →  rounds to 0.56 → "Moderate Risk"
```

## Step 3 — turn the number into a plain label

| Score range | Label |
|---|---|
| 0.0 – 0.3 | Low Risk |
| 0.3 – 0.6 | Moderate Risk |
| 0.6 – 1.0 | High Risk |

## A note on where the weights came from (reconciliation)

The original thesis draft has two figures that don't quite agree with each
other, and it's worth being upfront about this in case it comes up when
discussing the thesis:

- One figure (the causal diagram, Fig 3.1) shows weights for weather,
  visibility, speed, and proximity that add up to 1.20 instead of 1.0 —
  clearly just a drafting slip, not a deliberate design.
- The only actual code shown in the draft (Fig 6.1) only implements three
  variables (speed, visibility, proximity) and doesn't include weather at
  all — it looks like an early, unfinished snippet, since the abstract and
  every table in the draft (Table 3.1, Table 4.2) describe **four**
  variables including weather as a first-class part of the model.

This prototype resolves both issues the same way: it implements all **four**
variables (matching the majority of the draft), and it fixes the weights by
dividing the original four numbers by 1.20 so they add up to exactly 1.0,
while keeping their original order (weather mattered most, speed mattered
least, visibility and proximity were roughly equal). That's where 0.35 /
0.25 / 0.25 / 0.15 comes from. This is a reasonable, defensible way to
"clean up" an inconsistent early draft — and it's worth mentioning
explicitly in the final report as a methodology decision, not hiding it.

## What "intervention" means here

An intervention is simply: **change one variable, keep the rest the same,
and see how the score moves.** For example, take the worked example above
(risk 0.56, Moderate) and ask "what if speed had been `low` instead of
`high`?":

```
new risk = 0.35×0.2 + 0.25×0.5 + 0.25×0.9 + 0.15×0.2
         = 0.07     + 0.125    + 0.225    + 0.03
         = 0.45  →  still "Moderate Risk", but noticeably lower
```

That comparison — before vs. after, and whether the risk *band* changed —
is the entire point of the prototype. See `v1_legacy/risk_model.py`'s
`simulate_intervention()` function for the actual code, and
`v1_legacy/outputs/intervention_results.csv` after running
`v1_legacy/prototype.py` for real numbers on the 3 videos.
