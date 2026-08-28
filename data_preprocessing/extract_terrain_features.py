#!/usr/bin/env python3
"""Extract lightweight terrain features from binary x/y/z PCD frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "step", "slope", "roughness", "curvature", "height_range",
    "elevation_iqr", "roughness_p95", "occupied_cell_ratio",
)
INVALID_FEATURES = (np.nan,) * len(FEATURE_NAMES)


def read_binary_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    data_pos = raw.find(b"DATA binary")
    if data_pos < 0:
        raise ValueError(f"{path} is not a binary PCD")
    payload_start = raw.find(b"\n", data_pos) + 1
    points = np.frombuffer(raw[payload_start:], dtype="<f4")
    if points.size % 4:
        raise ValueError(f"unexpected point stride in {path}")
    return points.reshape(-1, 4)[:, :3]


def terrain_features(points: np.ndarray, cell_size: float, x_min: float,
                     x_max: float, y_abs_max: float) -> tuple[float, ...]:
    roi = points[
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (np.abs(points[:, 1]) <= y_abs_max) & np.isfinite(points).all(axis=1)
    ]
    if len(roi) < 20:
        return INVALID_FEATURES

    # Mean elevation per XY cell is computed with bincount for fast batch
    # preprocessing. The ROI and cell aggregation suppress isolated returns.
    cells = np.floor(roi[:, :2] / cell_size).astype(np.int32)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    elevations = (np.bincount(inverse, weights=roi[:, 2]) /
                  np.bincount(inverse)).astype(np.float32)
    if len(elevations) < 6:
        return INVALID_FEATURES

    xy = unique.astype(np.float64) * cell_size
    design = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
    coefficients, _, _, _ = np.linalg.lstsq(design, elevations, rcond=None)
    residuals = elevations - design @ coefficients
    slope = float(np.arctan(np.hypot(coefficients[0], coefficients[1])))
    absolute_residuals = np.abs(residuals)
    roughness = float(np.sqrt(np.mean(residuals ** 2)))
    roughness_p95 = float(np.percentile(absolute_residuals, 95))
    centered = np.column_stack((xy, elevations)) - np.mean(
        np.column_stack((xy, elevations)), axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
    curvature = float(eigenvalues[0] / max(float(eigenvalues.sum()), 1e-12))

    # Maximum height jump between adjacent occupied cells. A dense temporary
    # grid keeps this part vectorized while the ROI bounds memory usage.
    grid_x = int(np.ceil((x_max - x_min) / cell_size)) + 2
    grid_y = int(np.ceil(2 * y_abs_max / cell_size)) + 2
    dense = np.full((grid_x, grid_y), np.nan, dtype=np.float32)
    ix = np.floor((unique[:, 0] - x_min) / cell_size).astype(int)
    iy = np.floor((unique[:, 1] + y_abs_max) / cell_size).astype(int)
    inside = (ix >= 0) & (ix < grid_x) & (iy >= 0) & (iy < grid_y)
    dense[ix[inside], iy[inside]] = elevations[inside]
    jumps_x = np.abs(dense[1:, :] - dense[:-1, :])
    jumps_y = np.abs(dense[:, 1:] - dense[:, :-1])
    all_jumps = np.concatenate((jumps_x.ravel(), jumps_y.ravel()))
    finite_jumps = all_jumps[np.isfinite(all_jumps)]
    step = float(finite_jumps.max()) if finite_jumps.size else 0.0
    height_range = float(np.percentile(elevations, 95) -
                         np.percentile(elevations, 5))
    elevation_iqr = float(np.percentile(elevations, 75) -
                          np.percentile(elevations, 25))
    occupied_cell_ratio = float(len(elevations) / (grid_x * grid_y))
    return (step, slope, roughness, curvature, height_range, elevation_iqr,
            roughness_p95, occupied_cell_ratio)


def build(frame_dir: Path, cell_size: float, x_min: float, x_max: float,
          y_abs_max: float) -> None:
    pcd_files = sorted(frame_dir.glob("*.pcd"))
    if not pcd_files:
        raise FileNotFoundError(f"no PCD files found in {frame_dir}")
    output = frame_dir / "terrain_features.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "pcd_file", *FEATURE_NAMES])
        invalid = 0
        for frame_index, path in enumerate(pcd_files):
            values = terrain_features(read_binary_pcd(path), cell_size,
                                      x_min, x_max, y_abs_max)
            invalid += int(not np.isfinite(values).all())
            writer.writerow([frame_index, path.name, *values])
    print(f"Input PCD frames: {len(pcd_files)}")
    print(f"Invalid feature rows: {invalid}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=0.20)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--y-abs-max", type=float, default=4.0)
    args = parser.parse_args()
    build(args.frame_dir, args.cell_size, args.x_min, args.x_max,
          args.y_abs_max)
