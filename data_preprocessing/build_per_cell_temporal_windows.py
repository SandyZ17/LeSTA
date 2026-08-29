#!/usr/bin/env python3
"""Build per-grid-cell temporal samples aligned in the map frame.

Each sample is one map cell observed over ``T`` frames.  The target is a
future-motion proxy label for that cell, so the resulting model output has the
same granularity as a GridMap/costmap cell.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


TERRAIN_NAMES = (
    "step", "slope", "roughness", "curvature", "height_range",
    "elevation_iqr", "roughness_p95", "occupied_cell_ratio",
)
MOTION_NAMES = (
    "linear_x_causal", "angular_z_causal", "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z", "delta_t", "longitudinal_accel",
    "lateral_accel", "accel_norm", "gyro_norm",
)


def read_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = raw.find(b"DATA binary")
    if marker < 0:
        raise ValueError(f"{path} is not a binary PCD")
    start = raw.find(b"\n", marker) + 1
    values = np.frombuffer(raw[start:], dtype="<f4")
    if values.size % 4:
        raise ValueError(f"unexpected point stride in {path}")
    return values.reshape(-1, 4)[:, :3]


def quaternion_matrix(q):
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def transform_from_row(row):
    rotation = quaternion_matrix([float(row[name]) for name in
                                  ("rotation_x", "rotation_y", "rotation_z", "rotation_w")])
    translation = np.asarray([float(row[name]) for name in
                              ("translation_x", "translation_y", "translation_z")])
    return rotation, translation


def cell_features(points, cell_size):
    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    means = np.bincount(inverse, weights=points[:, 2]) / np.bincount(inverse)
    heights = {tuple(cell): float(height) for cell, height in zip(unique, means)}
    output = {}
    for cell, height in heights.items():
        neighbour_cells = [(cell[0] + dx, cell[1] + dy)
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        neighbours = [heights[key] for key in neighbour_cells if key in heights]
        values = np.asarray(neighbours, dtype=np.float64)
        if values.size < 4:
            continue
        left = heights.get((cell[0] - 1, cell[1]), height)
        right = heights.get((cell[0] + 1, cell[1]), height)
        down = heights.get((cell[0], cell[1] - 1), height)
        up = heights.get((cell[0], cell[1] + 1), height)
        dx = (right - left) / (2.0 * cell_size)
        dy = (up - down) / (2.0 * cell_size)
        slope = math.atan(math.hypot(dx, dy))
        residuals = values - values.mean()
        roughness = float(np.sqrt(np.mean(residuals ** 2)))
        curvature = float(abs((left + right + up + down) / 4.0 - height))
        output[cell] = np.asarray([
            float(values.max() - values.min()), slope, roughness, curvature,
            float(np.percentile(values, 95) - np.percentile(values, 5)),
            float(np.percentile(values, 75) - np.percentile(values, 25)),
            float(np.percentile(np.abs(residuals), 95)),
            float(values.size / 9.0),
        ], dtype=np.float32)
    return output


def pose_at_or_before(rows, stamp_ns):
    candidates = [row for row in rows if int(row["stamp_ns"]) <= stamp_ns]
    return candidates[-1] if candidates else None


def future_cells(tf_rows, tf_times, stamp_ns, cell_size, horizon_s,
                 footprint_front, footprint_rear, footprint_half_width):
    end_ns = stamp_ns + int(horizon_s * 1e9)
    cells = set()
    start = int(np.searchsorted(tf_times, stamp_ns, side="left"))
    stop = int(np.searchsorted(tf_times, end_ns, side="right"))
    local = np.asarray([(x, y, 0.0)
                        for x in np.arange(-footprint_rear, footprint_front + cell_size / 2, cell_size)
                        for y in np.arange(-footprint_half_width, footprint_half_width + cell_size / 2, cell_size)],
                       dtype=np.float64)
    for row in tf_rows[start:stop]:
        rotation, translation = transform_from_row(row)
        points = (rotation @ local.T).T + translation
        cells.update(map(tuple, np.floor(points[:, :2] / cell_size).astype(np.int64)))
    return cells


def build(frame_dir: Path, window_size: int, stride: int, cell_size: float,
          horizon_s: float, footprint_front: float, footprint_rear: float,
          footprint_half_width: float):
    manifest = load_csv(frame_dir / "manifest.csv")
    aligned = load_csv(frame_dir / "aligned.csv")
    tf_rows = sorted(load_csv(frame_dir / "tf.csv"), key=lambda row: int(row["stamp_ns"]))
    tf_times = np.asarray([int(row["stamp_ns"]) for row in tf_rows], dtype=np.int64)
    static_rows = load_csv(frame_dir / "tf_static.csv")
    static = next(row for row in static_rows
                  if row["parent_frame"] == "base_link")
    lidar_to_base_r, lidar_to_base_t = transform_from_row(static)
    imu_rows = sorted(load_csv(frame_dir / "imu.csv"), key=lambda row: int(row["stamp_ns"]))
    imu_times = np.asarray([int(row["stamp_ns"]) for row in imu_rows], dtype=np.int64)

    by_frame = {}
    for index, (manifest_row, aligned_row) in enumerate(zip(manifest, aligned)):
        stamp = int(manifest_row["stamp_ns"])
        pose = pose_at_or_before(tf_rows, stamp)
        if pose is None:
            by_frame[index] = {}
            continue
        map_r, map_t = transform_from_row(pose)
        points = read_pcd(frame_dir / manifest_row["pcd_file"])
        points = points[(points[:, 0] >= 0.0) & (points[:, 0] <= 8.0) &
                        (np.abs(points[:, 1]) <= 4.0) & np.isfinite(points).all(axis=1)]
        points = (map_r @ (lidar_to_base_r @ points.T + lidar_to_base_t[:, None]) +
                  map_t[:, None]).T
        terrain = cell_features(points, cell_size)
        imu_index = int(np.searchsorted(imu_times, stamp, side="right")) - 1
        if imu_index < 0:
            by_frame[index] = {}
            continue
        imu = imu_rows[imu_index]
        previous_speed = float(aligned[index - 1]["linear_x_causal"]) if index and \
            aligned[index - 1]["linear_x_causal"] else np.nan
        speed = float(aligned_row["linear_x_causal"]) if aligned_row["linear_x_causal"] else np.nan
        yaw = float(aligned_row["angular_z_causal"]) if aligned_row["angular_z_causal"] else np.nan
        delta_t = ((stamp - int(aligned[index - 1]["lidar_stamp_ns"])) / 1e9
                   if index else np.nan)
        motion = np.asarray([
            speed, yaw, float(imu["accel_x"]), float(imu["accel_y"]), float(imu["accel_z"]),
            float(imu["gyro_x"]), float(imu["gyro_y"]), float(imu["gyro_z"]), delta_t,
            (speed - previous_speed) / delta_t if np.isfinite(previous_speed) and delta_t > 1e-6 else np.nan,
            speed * yaw if np.isfinite(speed) and np.isfinite(yaw) else np.nan,
            float(np.linalg.norm([float(imu["accel_x"]), float(imu["accel_y"]), float(imu["accel_z"])])),
            float(np.linalg.norm([float(imu["gyro_x"]), float(imu["gyro_y"]), float(imu["gyro_z"])])),
        ], dtype=np.float32)
        if not np.isfinite(motion).all():
            by_frame[index] = {}
            continue
        by_frame[index] = {cell: np.concatenate((values, motion))
                           for cell, values in terrain.items()}

    future_cell_sets = {}
    for end in range(window_size - 1, len(manifest), stride):
        future_cell_sets[end] = future_cells(
            tf_rows, tf_times, int(manifest[end]["stamp_ns"]), cell_size,
            horizon_s, footprint_front, footprint_rear, footprint_half_width)
    samples, labels, cell_indices, target_stamps = [], [], [], []
    for end in range(window_size - 1, len(manifest), stride):
        cells = by_frame[end].keys()
        for cell in cells:
            sequence = [by_frame[index].get(cell) for index in
                        range(end - window_size + 1, end + 1)]
            if any(value is None for value in sequence):
                continue
            stamp = int(manifest[end]["stamp_ns"])
            label = 1 if cell in future_cell_sets[end] else -1
            samples.append(sequence)
            labels.append(label)
            cell_indices.append(cell)
            target_stamps.append(stamp)

    output_dir = frame_dir / "per_cell_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = np.asarray(MOTION_NAMES + TERRAIN_NAMES)
    np.savez_compressed(
        output_dir / "temporal_cell_windows.npz",
        features=np.asarray(samples, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        cell_indices=np.asarray(cell_indices, dtype=np.int64),
        target_stamps=np.asarray(target_stamps, dtype=np.int64),
        feature_names=feature_names,
        window_size=window_size,
    )
    print(f"Per-cell samples: {len(samples)}")
    print(f"Positive labels: {sum(label == 1 for label in labels)}")
    print(f"Unknown labels: {sum(label == -1 for label in labels)}")
    print(f"Saved: {output_dir / 'temporal_cell_windows.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cell-size", type=float, default=0.2)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--footprint-front", type=float, default=0.8)
    parser.add_argument("--footprint-rear", type=float, default=0.5)
    parser.add_argument("--footprint-half-width", type=float, default=0.6)
    args = parser.parse_args()
    build(args.frame_dir, args.window_size, args.stride, args.cell_size,
          args.horizon_s, args.footprint_front, args.footprint_rear,
          args.footprint_half_width)
