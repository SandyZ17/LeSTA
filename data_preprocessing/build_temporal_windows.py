#!/usr/bin/env python3
"""Build fixed-length temporal windows from aligned LeSTA frame metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "linear_x_pchip", "angular_z_pchip", "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z", "delta_t",
)


def load_rows(frame_dir: Path):
    with (frame_dir / "aligned.csv").open(newline="") as stream:
        aligned = list(csv.DictReader(stream))
    with (frame_dir / "imu.csv").open(newline="") as stream:
        imu_rows = sorted(csv.DictReader(stream), key=lambda row: int(row["stamp_ns"]))
    imu_times = np.asarray([int(row["stamp_ns"]) for row in imu_rows], dtype=np.int64)
    return aligned, imu_rows, imu_times


def nearest_imu(imu_rows, imu_times, stamp_ns):
    index = int(np.searchsorted(imu_times, stamp_ns, side="left"))
    candidates = []
    if index < len(imu_rows):
        candidates.append(imu_rows[index])
    if index > 0:
        candidates.append(imu_rows[index - 1])
    return min(candidates, key=lambda row: abs(int(row["stamp_ns"]) - stamp_ns))


def build(frame_dir: Path, window_size: int, stride: int, max_imu_skew_ms: float):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_imu_skew_ms < 0:
        raise ValueError("max_imu_skew_ms must be non-negative")
    aligned, imu_rows, imu_times = load_rows(frame_dir)
    frame_features = []
    valid = []
    for index, row in enumerate(aligned):
        stamp_ns = int(row["lidar_stamp_ns"])
        imu = nearest_imu(imu_rows, imu_times, stamp_ns)
        imu_skew_ms = abs(int(imu["stamp_ns"]) - stamp_ns) / 1e6
        is_valid = row["odom_interp_valid"] == "1" and imu_skew_ms <= max_imu_skew_ms
        previous_stamp = int(aligned[index - 1]["lidar_stamp_ns"]) if index else stamp_ns
        frame_features.append([
            float(row["linear_x_pchip"]) if row["linear_x_pchip"] else np.nan,
            float(row["angular_z_pchip"]) if row["angular_z_pchip"] else np.nan,
            float(imu["accel_x"]), float(imu["accel_y"]), float(imu["accel_z"]),
            float(imu["gyro_x"]), float(imu["gyro_y"]), float(imu["gyro_z"]),
            (stamp_ns - previous_stamp) / 1e9,
        ])
        valid.append(is_valid)

    frame_features = np.asarray(frame_features, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    windows = []
    for end in range(window_size - 1, len(aligned), stride):
        start = end - window_size + 1
        if valid[start:end + 1].all():
            windows.append((start, end))

    output_dir = frame_dir / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = np.asarray([frame_features[start:end + 1] for start, end in windows], dtype=np.float32)
    frame_indices = np.asarray([list(range(start, end + 1)) for start, end in windows], dtype=np.int64)
    pcd_files = np.asarray([
        [aligned[index]["pcd_file"] for index in range(start, end + 1)]
        for start, end in windows
    ])
    np.savez_compressed(
        output_dir / "temporal_windows.npz",
        features=sequences,
        frame_indices=frame_indices,
        pcd_files=pcd_files,
        feature_names=np.asarray(FEATURE_NAMES),
        window_size=window_size,
    )

    with (output_dir / "temporal_windows.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_index", "start_frame", "target_frame", "window_size"])
        writer.writerows([
            [window_index, start, end, window_size]
            for window_index, (start, end) in enumerate(windows)
        ])

    print("Input frames: %d" % len(aligned))
    print("Valid frames: %d" % valid.sum())
    print("Temporal windows: %d" % len(windows))
    print("Saved: %s" % (output_dir / "temporal_windows.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-imu-skew-ms", type=float, default=10.0)
    args = parser.parse_args()
    build(args.frame_dir, args.window_size, args.stride, args.max_imu_skew_ms)
