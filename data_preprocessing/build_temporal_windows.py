#!/usr/bin/env python3
"""Build fixed-length temporal windows from aligned LeSTA frame metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "linear_x_causal", "angular_z_causal", "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z", "delta_t",
)
RISK_FEATURE_NAMES = (
    "longitudinal_accel", "lateral_accel", "accel_norm", "gyro_norm",
)


def load_rows(frame_dir: Path):
    with (frame_dir / "aligned.csv").open(newline="") as stream:
        aligned = list(csv.DictReader(stream))
    with (frame_dir / "imu.csv").open(newline="") as stream:
        imu_rows = sorted(csv.DictReader(stream), key=lambda row: int(row["stamp_ns"]))
    with (frame_dir / "terrain_features.csv").open(newline="") as stream:
        terrain_rows = list(csv.DictReader(stream))
    with (frame_dir / "traversability_labels.csv").open(newline="") as stream:
        label_rows = list(csv.DictReader(stream))
    imu_times = np.asarray([int(row["stamp_ns"]) for row in imu_rows], dtype=np.int64)
    return aligned, imu_rows, imu_times, terrain_rows, label_rows


def nearest_imu(imu_rows, imu_times, stamp_ns):
    index = int(np.searchsorted(imu_times, stamp_ns, side="right")) - 1
    if index < 0:
        return None
    return imu_rows[index]


def build(frame_dir: Path, window_size: int, stride: int, max_imu_skew_ms: float):
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_imu_skew_ms < 0:
        raise ValueError("max_imu_skew_ms must be non-negative")
    aligned, imu_rows, imu_times, terrain_rows, label_rows = load_rows(frame_dir)
    if len(terrain_rows) != len(aligned):
        raise ValueError("terrain_features.csv must contain one row per aligned frame")
    if len(label_rows) != len(aligned):
        raise ValueError("traversability_labels.csv must contain one row per aligned frame")
    terrain_feature_names = tuple(name for name in terrain_rows[0]
                                  if name not in ("frame_index", "pcd_file"))
    frame_features = []
    valid = []
    for index, row in enumerate(aligned):
        stamp_ns = int(row["lidar_stamp_ns"])
        imu = nearest_imu(imu_rows, imu_times, stamp_ns)
        if imu is None:
            raise ValueError("no causal IMU sample available for LiDAR frame")
        imu_skew_ms = (stamp_ns - int(imu["stamp_ns"])) / 1e6
        is_valid = row["odom_causal_valid"] == "1" and imu_skew_ms <= max_imu_skew_ms
        previous_stamp = int(aligned[index - 1]["lidar_stamp_ns"]) if index else stamp_ns
        delta_t = (stamp_ns - previous_stamp) / 1e9
        speed = float(row["linear_x_causal"]) if row["linear_x_causal"] else np.nan
        accel = np.asarray([float(imu["accel_x"]), float(imu["accel_y"]),
                            float(imu["accel_z"])])
        gyro = np.asarray([float(imu["gyro_x"]), float(imu["gyro_y"]),
                           float(imu["gyro_z"])])
        current_features = [
            speed,
            float(row["angular_z_causal"]) if row["angular_z_causal"] else np.nan,
            float(imu["accel_x"]), float(imu["accel_y"]), float(imu["accel_z"]),
            float(imu["gyro_x"]), float(imu["gyro_y"]), float(imu["gyro_z"]),
            delta_t,
            (speed - float(aligned[index - 1]["linear_x_causal"])) / delta_t
            if index and delta_t > 1e-6 and aligned[index - 1]["linear_x_causal"] else np.nan,
            speed * float(row["angular_z_causal"]) if row["angular_z_causal"] else np.nan,
            float(np.linalg.norm(accel)), float(np.linalg.norm(gyro)),
            *[float(terrain_rows[index][name]) for name in terrain_feature_names],
        ]
        frame_features.append(current_features)
        valid.append(is_valid and np.isfinite(current_features).all())

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
    labels = np.asarray([int(label_rows[end]["label"]) for _, end in windows], dtype=np.int8)
    np.savez_compressed(
        output_dir / "temporal_windows.npz",
        features=sequences,
        frame_indices=frame_indices,
        pcd_files=pcd_files,
        labels=labels,
        feature_names=np.asarray(FEATURE_NAMES + RISK_FEATURE_NAMES + terrain_feature_names),
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
