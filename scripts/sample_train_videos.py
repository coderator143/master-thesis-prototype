"""
Incremental, re-runnable script: tops up videos/train/ to PER_SOURCE videos
per dataset source, stratified evenly across the 4 sources. Safe to re-run
after raising PER_SOURCE -- it only copies what's missing, never re-copies
or removes videos already present, so earlier Phase 1/2 work on existing
videos never needs to be redone.

Video IDs are only unique *within* a source (e.g. CAP_DATA/VRU_9.mp4 and
DADA_2000/VRU_9.mp4 are different videos) -- see
docs/approach/03-dataset-source.md. Every video is therefore identified by
(source, video_id), never video_id alone, and copied into a
source-namespaced subfolder to avoid collisions:
videos/train/<source>/<video_id>.mp4

Uses a fixed random seed per top-up batch (see SEED_BY_BATCH) so each
expansion is reproducible on its own.
"""

import csv
import json
import random
import shutil
from pathlib import Path

DATASET_ROOT = Path("/Users/warshaw65/Desktop/VRU-Accident")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = ["CAP_DATA", "DADA_2000", "DoTA", "MANUAL_DATA"]

# Raise this and re-run to top up. History: 25 (initial Phase 1 pass), then
# 75 (expanded before Phase 3, to keep a healthier samples-per-feature
# ratio after adding closing_risk/braking/location -- see
# docs/01-phases-and-roadmap.md). DoTA is the tightest source (100 videos
# total) -- 75/source stays comfortably within it (75 needed, 75 available
# after the initial 25 are excluded).
PER_SOURCE = 75

# A different seed per expansion round, so each round's *new* picks are
# independently reproducible rather than accidentally deterministic
# leftovers of a single larger sample() call.
SEED_BY_ROUND = {25: 42, 75: 43}

# (source, video_id) pairs already present in videos/test/ -- never sample
# these into train/. Video IDs collide across sources, so this must be
# scoped per-source, not a flat set of IDs.
ALREADY_IN_TEST = {
    ("CAP_DATA", "VRU_9"),
    ("CAP_DATA", "VRU_10"),
    ("CAP_DATA", "VRU_14"),
}


def candidate_ids(source):
    vqa_path = DATASET_ROOT / "MetaData" / source / f"{source}_VQA_annotation.json"
    with open(vqa_path) as f:
        data = json.load(f)
    ids = sorted(Path(key).stem for key in data.keys())
    return [vid for vid in ids if (source, vid) not in ALREADY_IN_TEST]


def already_copied(source):
    dest_dir = PROJECT_ROOT / "videos" / "train" / source
    if not dest_dir.exists():
        return set()
    return {p.stem for p in dest_dir.glob("*.mp4")}


def main():
    rng = random.Random(SEED_BY_ROUND.get(PER_SOURCE, PER_SOURCE))
    new_rows = []

    for source in SOURCES:
        existing = already_copied(source)
        needed = PER_SOURCE - len(existing)
        if needed <= 0:
            print(f"{source}: already has {len(existing)} >= {PER_SOURCE}, nothing to add")
            continue

        candidates = [v for v in candidate_ids(source) if v not in existing]
        new_sample = rng.sample(candidates, needed)

        dest_dir = PROJECT_ROOT / "videos" / "train" / source
        dest_dir.mkdir(parents=True, exist_ok=True)

        for vid in new_sample:
            src_path = DATASET_ROOT / "VRU_videos" / source / f"{vid}.mp4"
            dest_path = dest_dir / f"{vid}.mp4"
            shutil.copy2(src_path, dest_path)
            new_rows.append({
                "video_id": vid,
                "source": source,
                "split": "train",
                "filename": f"train/{source}/{vid}.mp4",
            })
        print(f"{source}: had {len(existing)}, added {len(new_sample)} -> {len(existing) + len(new_sample)}")

    manifest_path = PROJECT_ROOT / "data" / "train_video_manifest.csv"
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "source", "split", "filename"])
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nAdded {len(new_rows)} new videos this round. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
