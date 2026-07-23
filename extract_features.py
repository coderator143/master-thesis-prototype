"""
Phase 1: visual feature extraction.

Turns each video in videos/{test,train}/<source>/<video_id>.mp4 into a row
of scene variables, using YOLO detection+tracking and OpenCV, instead of
manually guessing them (that's v1's approach -- see risk_model.py). See
docs/01-phases-and-roadmap.md for the full plan this implements.

Ground-truth variables (weather, road_type) are pulled from the dataset's
own annotations via ground_truth.py, not derived by CV -- see
docs/approach/03-dataset-source.md for why.

Run: python extract_features.py
Produces: data/scene_dataset_raw.csv (per-video raw values, resumable),
          data/scene_dataset.csv (final bucketed schema).
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import ground_truth

VIDEO_ROOT = Path("videos")
RAW_CSV = Path("data/scene_dataset_raw.csv")
FINAL_CSV = Path("data/scene_dataset.csv")

TARGET_FPS = 2.0
MAX_SAMPLED_FRAMES = 30
DEFAULT_VID_STRIDE = 10  # fallback if a video's native fps can't be read

RESIZE_WIDTH = 320  # frames are resized to this width before flow/brightness
                     # analysis so values are comparable across the dataset's
                     # mixed resolutions (360p-1080p)

YOLO_MODEL_NAME = "yolov8n.pt"
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck
OBSTACLE_DISPLACEMENT_THRESHOLD = 0.03  # fraction of frame diagonal

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BRAKING_MIN_FLOW_PAIRS = 4  # need this many consecutive-frame flow readings
                            # before a first-half-vs-second-half trend is
                            # meaningful; short clips get "unknown" instead
BRAKING_RELATIVE_DROP = 0.15  # >=15% relative decrease in flow magnitude,
                               # second half of the clip vs. first half

RAW_FIELDNAMES = [
    "video_id", "source", "split", "filename",
    "vehicle_speed_raw", "pedestrian_distance_raw", "closing_risk_raw",
    "visibility_brightness_raw", "visibility_contrast_raw",
    "num_pedestrians", "traffic_density",
    "weather", "road_type", "location", "obstacle", "braking",
    "n_sampled_frames",
]


def list_all_videos() -> pd.DataFrame:
    """Globs videos/{test,train}/<source>/*.mp4 directly -- the filesystem
    layout is the source of truth for which videos exist and their
    (split, source, video_id), per docs/approach/03-dataset-source.md."""
    rows = []
    for split in ["test", "train"]:
        split_dir = VIDEO_ROOT / split
        if not split_dir.exists():
            continue
        for source_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for video_path in sorted(source_dir.glob("*.mp4")):
                rows.append({
                    "video_id": video_path.stem,
                    "source": source_dir.name,
                    "split": split,
                    "filename": f"{split}/{source_dir.name}/{video_path.stem}.mp4",
                })
    return pd.DataFrame(rows)


def get_vid_stride(video_path: Path, target_fps: float) -> int:
    """Frame-skip count so YOLO/OpenCV sample at roughly target_fps,
    regardless of a video's native frame rate."""
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not native_fps or native_fps <= 0:
        return DEFAULT_VID_STRIDE
    return max(1, round(native_fps / target_fps))


def process_video(model: YOLO, source: str, video_id: str, split: str, filename: str) -> dict:
    video_path = VIDEO_ROOT / filename
    vid_stride = get_vid_stride(video_path, TARGET_FPS)

    gray_frames = []
    background_masks = []
    person_track_ids = set()
    max_persons_in_one_frame = 0
    person_box_heights_norm = []
    vehicle_counts_per_frame = []
    vehicle_tracks = defaultdict(list)  # track_id -> [(frame_idx, cx, cy), ...]
    frame_diag = None

    results_stream = model.track(
        source=str(video_path),
        stream=True,
        tracker="bytetrack.yaml",
        vid_stride=vid_stride,
        device=DEVICE,
        verbose=False,
    )

    n_frames = 0
    for frame_idx, r in enumerate(results_stream):
        if n_frames >= MAX_SAMPLED_FRAMES:
            break

        orig = r.orig_img  # BGR, original resolution
        h, w = orig.shape[:2]
        if frame_diag is None:
            frame_diag = (h ** 2 + w ** 2) ** 0.5

        scale = RESIZE_WIDTH / w
        resized = cv2.resize(orig, (RESIZE_WIDTH, max(1, round(h * scale))))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=bool)

        n_vehicles_this_frame = 0
        n_persons_this_frame = 0
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)

            for (x1, y1, x2, y2), c, tid in zip(xyxy, cls, ids):
                rx1 = int(max(0, x1 * scale))
                ry1 = int(max(0, y1 * scale))
                rx2 = int(max(0, x2 * scale))
                ry2 = int(max(0, y2 * scale))
                mask[ry1:ry2, rx1:rx2] = True

                if c == PERSON_CLASS_ID:
                    n_persons_this_frame += 1
                    if tid is not None:
                        person_track_ids.add(int(tid))
                    person_box_heights_norm.append((y2 - y1) / h)
                elif c in VEHICLE_CLASS_IDS:
                    n_vehicles_this_frame += 1
                    if tid is not None:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        vehicle_tracks[int(tid)].append((frame_idx, cx, cy))

        max_persons_in_one_frame = max(max_persons_in_one_frame, n_persons_this_frame)
        vehicle_counts_per_frame.append(n_vehicles_this_frame)
        gray_frames.append(gray)
        background_masks.append(mask)
        n_frames += 1

    flow_series = _compute_flow_series(gray_frames, background_masks)
    vehicle_speed_raw = float(np.mean(flow_series)) if flow_series else np.nan
    braking = _compute_braking(flow_series)

    brightness_raw = float(np.mean([g.mean() for g in gray_frames])) if gray_frames else np.nan
    contrast_raw = float(np.mean([g.std() for g in gray_frames])) if gray_frames else np.nan
    pedestrian_distance_raw = max(person_box_heights_norm) if person_box_heights_norm else np.nan
    # Interaction proxy combining ego speed and how close the nearest
    # pedestrian got -- the kind of "closing risk" a Random Forest won't
    # reliably reconstruct on its own from the two variables sitting in
    # separate columns. Not real seconds-to-collision (neither input is in
    # calibrated physical units), just a monotonic "how urgent was this"
    # proxy: higher speed AND higher closeness (=1/distance-ish) -> higher.
    # NaN when no pedestrian was detected, same as pedestrian_distance_raw.
    closing_risk_raw = (
        pedestrian_distance_raw * vehicle_speed_raw
        if not (np.isnan(pedestrian_distance_raw) or np.isnan(vehicle_speed_raw))
        else np.nan
    )
    # Distinct track IDs is the better estimate when tracking succeeds (it
    # counts people who appear serially, not just simultaneously) -- but
    # ByteTrack occasionally returns ids=None for an entire frame under the
    # vid_stride frame-skipping used here, which would silently undercount
    # to 0 even when persons were clearly detected. Falling back to the
    # largest single-frame raw count guards against that failure mode.
    num_pedestrians = max(len(person_track_ids), max_persons_in_one_frame)
    traffic_density = float(np.mean(vehicle_counts_per_frame)) if vehicle_counts_per_frame else 0.0
    obstacle = _compute_obstacle(vehicle_tracks, frame_diag)

    gt = ground_truth.get_ground_truth(source, video_id)

    return {
        "video_id": video_id,
        "source": source,
        "split": split,
        "filename": filename,
        "vehicle_speed_raw": vehicle_speed_raw,
        "pedestrian_distance_raw": pedestrian_distance_raw,
        "closing_risk_raw": closing_risk_raw,
        "visibility_brightness_raw": brightness_raw,
        "visibility_contrast_raw": contrast_raw,
        "num_pedestrians": num_pedestrians,
        "traffic_density": traffic_density,
        "weather": gt["weather"],
        "road_type": gt["road_type"],
        "location": gt["location"],
        "obstacle": obstacle,
        "braking": braking,
        "n_sampled_frames": n_frames,
    }


def _compute_flow_series(gray_frames, background_masks) -> list:
    """Dense optical flow magnitude over background pixels only (all
    detected objects excluded), median per consecutive frame pair, as a
    time series across the clip. A dashcam's own speed shows up as
    background motion, not as any single tracked object's motion. Feeds
    both vehicle_speed_raw (the mean) and braking (the trend)."""
    flow_medians = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        background = ~background_masks[i]
        if background.sum() > 0:
            flow_medians.append(float(np.median(magnitude[background])))
    return flow_medians


def _compute_braking(flow_series) -> str:
    """Whether the ego-motion flow magnitude trends downward across the
    clip (second half meaningfully lower than the first half) -- a proxy
    for the vehicle decelerating before the collision. Note this describes
    the driver's *response* during the clip, not a fixed pre-existing scene
    condition like weather -- see docs/01-phases-and-roadmap.md for why
    that distinction matters when deciding what's "controllable" later."""
    if len(flow_series) < BRAKING_MIN_FLOW_PAIRS:
        return "unknown"
    half = len(flow_series) // 2
    first_half_mean = float(np.mean(flow_series[:half]))
    second_half_mean = float(np.mean(flow_series[half:]))
    if first_half_mean <= 0:
        return "unknown"
    relative_change = (second_half_mean - first_half_mean) / first_half_mean
    return "yes" if relative_change < -BRAKING_RELATIVE_DROP else "no"


def _compute_obstacle(vehicle_tracks, frame_diag) -> str:
    """Lower-priority heuristic (per the roadmap): a vehicle track that
    barely moves between its first and last detection is treated as a
    near-static obstacle (e.g. a parked car)."""
    if not frame_diag or frame_diag <= 0:
        return "unknown"
    for points in vehicle_tracks.values():
        if len(points) < 2:
            continue
        _, x0, y0 = points[0]
        _, x1, y1 = points[-1]
        displacement = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if displacement / frame_diag < OBSTACLE_DISPLACEMENT_THRESHOLD:
            return "yes"
    return "no"


def _tertile_index(series: pd.Series) -> pd.Series:
    """Maps values to 0/1/2 (bottom/middle/top third) using thresholds from
    this series' own distribution -- these are uncalibrated relative
    proxies, not values with fixed real-world units, so bucket boundaries
    have to come from the sample itself, not guessed constants."""
    valid = series.dropna()
    if valid.nunique() < 3:
        return pd.Series([np.nan] * len(series), index=series.index)
    q1, q2 = valid.quantile([1 / 3, 2 / 3])

    def idx(x):
        if pd.isna(x):
            return np.nan
        if x <= q1:
            return 0
        if x <= q2:
            return 1
        return 2

    return series.apply(idx)


def bucket_tertile(series: pd.Series, labels: tuple) -> pd.Series:
    index = _tertile_index(series)
    return index.apply(lambda i: labels[int(i)] if pd.notna(i) else "unknown")


def build_scene_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Applies tertile bucketing to the uncalibrated relative proxies
    (speed, distance, visibility) and passes the direct counts
    (num_pedestrians, traffic_density) through unbucketed -- see
    docs/01-phases-and-roadmap.md for why those two stay raw."""
    df = raw_df.copy()

    df["vehicle_speed"] = bucket_tertile(df["vehicle_speed_raw"], ("low", "medium", "high"))
    df["pedestrian_distance"] = bucket_tertile(df["pedestrian_distance_raw"], ("far", "medium", "close"))
    df["closing_risk"] = bucket_tertile(df["closing_risk_raw"], ("low", "medium", "high"))

    brightness_idx = _tertile_index(df["visibility_brightness_raw"])
    contrast_idx = _tertile_index(df["visibility_contrast_raw"])
    combined_idx = pd.concat([brightness_idx, contrast_idx], axis=1).min(axis=1, skipna=False)
    visibility_labels = ("low", "medium", "high")
    df["visibility"] = combined_idx.apply(
        lambda i: visibility_labels[int(i)] if pd.notna(i) else "unknown"
    )

    return df[[
        "video_id", "source", "split",
        "vehicle_speed", "pedestrian_distance", "closing_risk", "num_pedestrians",
        "weather", "visibility", "traffic_density", "road_type", "location",
        "obstacle", "braking",
    ]]


def main():
    videos_df = list_all_videos()
    print(f"Found {len(videos_df)} videos in {VIDEO_ROOT}/")

    done_pairs = set()
    write_header = True
    if RAW_CSV.exists():
        existing = pd.read_csv(RAW_CSV)
        done_pairs = set(zip(existing["source"], existing["video_id"]))
        write_header = False
        print(f"{len(done_pairs)} already processed in {RAW_CSV}, resuming")

    model = YOLO(YOLO_MODEL_NAME)

    failures = []
    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, row in videos_df.iterrows():
            key = (row["source"], row["video_id"])
            if key in done_pairs:
                continue
            t0 = time.time()
            try:
                result = process_video(model, row["source"], row["video_id"], row["split"], row["filename"])
                writer.writerow(result)
                f.flush()
                print(f"[{i + 1}/{len(videos_df)}] {row['source']}/{row['video_id']} ok ({time.time() - t0:.1f}s)")
            except Exception as e:
                failures.append((row["source"], row["video_id"], str(e)))
                print(f"[{i + 1}/{len(videos_df)}] {row['source']}/{row['video_id']} FAILED: {e} ({time.time() - t0:.1f}s)")

    print(f"\n{len(failures)} failures:")
    for s, v, e in failures:
        print(f"  {s}/{v}: {e}")

    raw_df = pd.read_csv(RAW_CSV)
    final_df = build_scene_dataset(raw_df)
    final_df.to_csv(FINAL_CSV, index=False)
    print(f"\nWrote {FINAL_CSV} ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
