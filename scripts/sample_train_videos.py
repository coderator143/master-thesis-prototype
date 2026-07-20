"""
One-off (but re-runnable) script: samples a training subset from the full
local VRU-Accident dataset and copies it into videos/train/, stratified
evenly across the dataset's 4 sources.

Video IDs are only unique *within* a source (e.g. CAP_DATA/VRU_9.mp4 and
DADA_2000/VRU_9.mp4 are different videos) -- see
docs/approach/03-dataset-source.md. Every video is therefore identified by
(source, video_id), never video_id alone, and copied into a
source-namespaced subfolder to avoid collisions:
videos/train/<source>/<video_id>.mp4

Uses a fixed random seed so the same sample can be reproduced or extended
later without re-copying already-selected videos.
"""

import csv
import json
import random
import shutil
from pathlib import Path

DATASET_ROOT = Path("/Users/warshaw65/Desktop/VRU-Accident")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = ["CAP_DATA", "DADA_2000", "DoTA", "MANUAL_DATA"]
PER_SOURCE = 25  # 25 x 4 sources = 100 videos total
SEED = 42

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


def main():
    rng = random.Random(SEED)
    manifest_rows = []

    for source in SOURCES:
        ids = candidate_ids(source)
        sample = rng.sample(ids, PER_SOURCE)

        dest_dir = PROJECT_ROOT / "videos" / "train" / source
        dest_dir.mkdir(parents=True, exist_ok=True)

        for vid in sample:
            src_path = DATASET_ROOT / "VRU_videos" / source / f"{vid}.mp4"
            dest_path = dest_dir / f"{vid}.mp4"
            shutil.copy2(src_path, dest_path)
            manifest_rows.append(
                {
                    "video_id": vid,
                    "source": source,
                    "split": "train",
                    "filename": f"train/{source}/{vid}.mp4",
                }
            )
        print(f"{source}: copied {len(sample)} videos")

    manifest_path = PROJECT_ROOT / "data" / "train_video_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "source", "split", "filename"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTotal: {len(manifest_rows)} videos copied into videos/train/")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
