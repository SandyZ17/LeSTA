#!/usr/bin/env python3
"""Generate conservative self-supervised frame labels from future motion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_aligned(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def to_float(value: str) -> float:
    return float(value) if value else np.nan


def generate(rows, horizon_s: float, min_displacement_m: float,
             min_speed_mps: float, active_fraction: float):
    times = np.asarray([int(row["lidar_stamp_ns"]) for row in rows], dtype=np.int64)
    x = np.asarray([to_float(row["pose_x_interp"]) for row in rows])
    y = np.asarray([to_float(row["pose_y_interp"]) for row in rows])
    speed = np.asarray([to_float(row["linear_x_pchip"]) for row in rows])
    valid = np.asarray([row["odom_interp_valid"] == "1" for row in rows])
    valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(speed)
    labels = np.full(len(rows), -1, dtype=np.int8)
    reasons = np.full(len(rows), "unknown", dtype=object)

    for index, stamp in enumerate(times):
        future_end = int(np.searchsorted(times, stamp + horizon_s * 1e9))
        if not valid[index] or future_end <= index + 1 or not valid[index:future_end].all():
            reasons[index] = "invalid_motion_interval"
            continue
        displacement = float(np.hypot(x[future_end - 1] - x[index],
                                      y[future_end - 1] - y[index]))
        future_speed = speed[index:future_end]
        moving_fraction = float(np.mean(future_speed > min_speed_mps))
        if (displacement >= min_displacement_m and
                moving_fraction >= active_fraction and
                float(np.min(future_speed)) >= -0.05):
            labels[index] = 1
            reasons[index] = "sustained_forward_motion"
        elif moving_fraction < active_fraction:
            reasons[index] = "stopped_or_insufficient_motion"
        else:
            reasons[index] = "insufficient_progress"
    return labels, reasons


def main(frame_dir: Path, horizon_s: float, min_displacement_m: float,
         min_speed_mps: float, active_fraction: float):
    rows = load_aligned(frame_dir / "aligned.csv")
    labels, reasons = generate(rows, horizon_s, min_displacement_m,
                               min_speed_mps, active_fraction)
    output = frame_dir / "traversability_labels.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "pcd_file", "label", "label_reason"])
        for row, label, reason in zip(rows, labels, reasons):
            writer.writerow([row["frame_index"], row["pcd_file"], int(label), reason])
    print(f"Input frames: {len(rows)}")
    print(f"Positive labels: {int(np.sum(labels == 1))}")
    print(f"Unknown labels: {int(np.sum(labels == -1))}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--min-displacement-m", type=float, default=0.25)
    parser.add_argument("--min-speed-mps", type=float, default=0.10)
    parser.add_argument("--active-fraction", type=float, default=0.70)
    args = parser.parse_args()
    main(args.frame_dir, args.horizon_s, args.min_displacement_m,
         args.min_speed_mps, args.active_fraction)
