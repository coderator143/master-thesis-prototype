"""
Phase 2 (risk_label): derives a severity/risk training target for Phase 3
from each video's dense caption text, since the dataset has no direct risk
field -- see docs/01-phases-and-roadmap.md's Phase 3 section.

Design grounded in surveying all 103 real captions first (not guessed):
- 64/103 (62%) captions state an explicit speed ("...traveling at
  approximately 40 km/h...", range 0-70 km/h) -- used as the primary
  severity signal when present, since impact speed is a real, objective
  proxy for collision severity.
- For the other 39/103, falls back to the CV-derived vehicle_speed bucket
  from Phase 1 (data/scene_dataset.csv), using the same low/medium/high ->
  0.2/0.5/0.9 mapping v1's risk_model.py already used. Flagged explicitly
  via the risk_label_source column ("text" vs "cv_fallback") -- for the
  cv_fallback rows, the label is derived partly from the same feature
  (vehicle_speed) that also goes into Phase 3's classifier as an input,
  a mild leakage risk worth knowing about rather than hiding. See
  docs/01-phases-and-roadmap.md.
- Outcome/evasive-action language is sparse in this dataset (most captions
  describe collision mechanics, not injury outcomes) but real when present
  ("remains motionless": 7/103, "gets up": 5/103, "no apparent evasive
  action": 16/103) -- used as a secondary adjustment, not the primary
  signal.

Run: python mine_risk_labels.py
Reads: data/scene_dataset.csv (must already exist -- run extract_features.py
       first), the dataset's own dense caption JSON files.
Writes: data/scene_dataset.csv, adding risk_score_raw, risk_label_source,
        and risk_label columns.
"""

import re

import pandas as pd

import ground_truth

SCENE_DATASET_CSV = "data/scene_dataset.csv"

SPEED_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:-|to|–|—|\?)?\s*(\d{1,3}(?:\.\d+)?)?\s*km/h"
)
MAX_OBSERVED_SPEED_KMH = 70  # from surveying all 103 captions; used to normalize to [0, 1]

# Same low/medium/high -> 0.2/0.5/0.9 convention as v1's risk_model.py,
# used here only as a fallback base severity when a caption doesn't state
# an explicit speed.
CV_SPEED_BASE = {"low": 0.2, "medium": 0.5, "high": 0.8, "unknown": 0.5}

NO_EVASIVE_PHRASES = [
    "no apparent evasive", "does not slow down", "without significantly reducing speed",
    "without any apparent evasive", "continues moving forward", "fails to stop",
    "unable to stop", "collision is unavoidable", "does not appear to brake",
]
SEVERE_OUTCOME_PHRASES = [
    "motionless", "unconscious", "does not move", "remains still",
    "serious", "critical", "fatal", "killed", "unresponsive",
]
MILD_OUTCOME_PHRASES = ["gets up", "walks away", "stands up", "minor injur"]

ADJUSTMENT = 0.15


def _extract_speed_kmh(caption: str):
    m = SPEED_RE.search(caption)
    return float(m.group(1)) if m else None


def severity_score(caption: str, cv_speed_bucket: str) -> tuple:
    """Returns (score in [0, 1], "text" or "cv_fallback")."""
    text_speed = _extract_speed_kmh(caption)
    if text_speed is not None:
        score = min(text_speed / MAX_OBSERVED_SPEED_KMH, 1.0)
        source = "text"
    else:
        score = CV_SPEED_BASE.get(cv_speed_bucket, 0.5)
        source = "cv_fallback"

    low = caption.lower()
    if any(p in low for p in NO_EVASIVE_PHRASES):
        score += ADJUSTMENT
    if any(p in low for p in SEVERE_OUTCOME_PHRASES):
        score += ADJUSTMENT
    elif any(p in low for p in MILD_OUTCOME_PHRASES):
        score -= ADJUSTMENT

    return max(0.0, min(1.0, score)), source


def bucket(score: float, q1: float, q2: float) -> str:
    if score <= q1:
        return "low"
    if score <= q2:
        return "medium"
    return "high"


def main():
    df = pd.read_csv(SCENE_DATASET_CSV)

    scores, sources = [], []
    for _, row in df.iterrows():
        caption = ground_truth.get_dense_caption(row["source"], row["video_id"])
        score, source = severity_score(caption, row["vehicle_speed"])
        scores.append(score)
        sources.append(source)

    df["risk_score_raw"] = scores
    df["risk_label_source"] = sources

    q1, q2 = df["risk_score_raw"].quantile([1 / 3, 2 / 3])
    df["risk_label"] = df["risk_score_raw"].apply(lambda x: bucket(x, q1, q2))

    df.to_csv(SCENE_DATASET_CSV, index=False)

    print(f"Wrote risk_label (+ risk_score_raw, risk_label_source) to {SCENE_DATASET_CSV}\n")
    print("risk_label distribution:")
    print(df["risk_label"].value_counts())
    print("\nrisk_label_source distribution:")
    print(df["risk_label_source"].value_counts())


if __name__ == "__main__":
    main()
