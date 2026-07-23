"""
Loads real ground-truth scene annotations (weather, road type) for
VRU-Accident videos from the dataset's own VQA annotation files, instead of
guessing them with computer vision -- see docs/approach/03-dataset-source.md
for why. Kept separate from extract_features.py since this is a distinct
concern (parsing another repo's JSON, no CV) and gets reused by Phase 3
for dense-caption reading later.

Video IDs are only unique *within* a source (docs/approach/03-dataset-source.md),
so every lookup here takes (source, video_id) together, never video_id alone.
"""

import functools
import json
import os
import re
from pathlib import Path

VRU_ACCIDENT_ROOT = Path(
    os.environ.get("VRU_ACCIDENT_ROOT", "/Users/warshaw65/Desktop/VRU-Accident")
)
SOURCES = ["CAP_DATA", "DADA_2000", "DoTA", "MANUAL_DATA"]

# Matches an option marker "A." / "B." / "C." / "D.", preceded by either the
# start of the string or a comma/period (with optional whitespace between).
# A naive split on ", " breaks on ~9% of real entries in this dataset, since
# some option text uses no space before the next marker ("B.Cyclist...") or
# a period instead of a comma ("...surrounding. B. Drivers..."). Verified
# against all 6,000 real question entries across all 4 sources -- the only
# remaining failures (12/6000) are malformed source strings in the
# "prevention method" category (a stray extra marker embedded mid-sentence),
# which this module doesn't use.
_OPTION_RE = re.compile(r"(?:^|[,.])\s*([A-D])\.\s*")


def _parse_options(options: str) -> dict:
    """Splits an options string like 'A. rainy day, B. sunny afternoon, ...'
    into {"A": "rainy day", "B": "sunny afternoon", ...}."""
    matches = list(_OPTION_RE.finditer(options))
    letters = [m.group(1) for m in matches]
    if letters != ["A", "B", "C", "D"]:
        raise ValueError(f"could not parse options string: {options!r}")
    parsed = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(options)
        parsed[m.group(1)] = options[start:end].strip()
    return parsed


@functools.lru_cache(maxsize=None)
def _load_annotations(source: str) -> dict:
    """Loads and parses one source's VQA annotation JSON once, keyed by
    video_id instead of the raw './VRU_videos/<source>/<id>.mp4' path.
    Cached per source -- called once per source across an entire batch run,
    not once per video."""
    path = VRU_ACCIDENT_ROOT / "MetaData" / source / f"{source}_VQA_annotation.json"
    with open(path) as f:
        raw = json.load(f)
    return {Path(key).stem: value for key, value in raw.items()}


def get_ground_truth(source: str, video_id: str) -> dict:
    """Returns {"weather": <text>, "road_type": <text>, "location": <text>}
    for one video, resolved from the dataset's own ground-truth
    annotations.

    Raises KeyError if (source, video_id) isn't found, or ValueError if the
    options string for this entry can't be parsed -- both are expected to
    be caught by the per-video try/except in extract_features.py, not
    handled here.
    """
    annotations = _load_annotations(source)
    entry = annotations[video_id]

    weather_qa = entry["weather and light"]
    road_qa = entry["road type"]
    location_qa = entry["location"]

    weather_options = _parse_options(weather_qa["options"])
    road_options = _parse_options(road_qa["options"])
    location_options = _parse_options(location_qa["options"])

    return {
        "weather": weather_options[weather_qa["GT"]],
        "road_type": road_options[road_qa["GT"]],
        "location": location_options[location_qa["GT"]],
    }


@functools.lru_cache(maxsize=None)
def _load_dense_captions(source: str) -> dict:
    """Loads one source's dense-caption JSON once, keyed by video_id. Used
    by mine_risk_labels.py (Phase 2) to derive a severity proxy -- see
    docs/01-phases-and-roadmap.md's Phase 3 section for why."""
    path = VRU_ACCIDENT_ROOT / "MetaData" / source / f"{source}_Dense_Caption.json"
    with open(path) as f:
        raw = json.load(f)
    return {Path(key).stem: value["gt"] for key, value in raw.items()}


def get_dense_caption(source: str, video_id: str) -> str:
    """Returns the free-text dense caption for one video."""
    return _load_dense_captions(source)[video_id]
