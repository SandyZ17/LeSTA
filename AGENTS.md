# Repository Guidelines

## Project Structure

LeSTA combines a ROS 2 Humble stack with a Python/PyTorch training package:

- `lesta_ros2/`: C++17 ROS 2 nodes, headers, Python launch files, YAML configuration, and the `SaveTrainingData` service. Core algorithms are under `src/core/`; ROS adapters are under `src/ros/`.
- `height_mapping/height_mapping_core_ros2/`: reusable height-map library used by `lesta_ros2`.
- `pylesta/`: Python training and inference code; models, datasets, losses, utilities, configs, and CLI tools are grouped by package.
- `assets/`: README figures and demonstration animations. Keep generated datasets, checkpoints, bags, and `.pcd` files out of commits; `.gitignore` already excludes them.

## Temporal Model Roadmap

Work on temporal learning belongs on `feature/temporal-model`; keep the ROS 2/Docker baseline stable.

- [ ] Record and synchronize LiDAR, TF, vehicle speed, acceleration, steering, wheel speed, and IMU data.
- [ ] Define a sequence dataset format with aligned timestamps and 5–10-frame windows. Split by driving sequence to prevent leakage between train and test data.
- [ ] Extend label generation with vehicle motion and IMU signals while retaining footprint and height-step labels as the baseline.
- [ ] Implement a feature-enhanced Temporal MLP using terrain features plus vehicle-state features.
- [ ] Preserve the original LeSTA MLP and, where possible, learn a residual correction rather than replacing it immediately.
- [ ] Compare the baseline MLP, feature-enhanced MLP, Temporal MLP, small GRU/LSTM, and XGBoost under the same data split.
- [ ] Export the selected model to TorchScript/LibTorch and integrate inference into the ROS 2 pipeline.
- [ ] Evaluate end-to-end latency, processing rate, dropped frames, resource usage, classification quality, planning success, vibration, and vehicle stability.

Do not add large point-cloud or Transformer models until the lightweight temporal baseline has been measured and shown to be insufficient.

## Current Experimental Findings

The first Temporal MLP experiment uses 6,548 ten-frame windows with 13
features per frame. It runs in Docker with PyTorch CUDA 12.9.1. On the current
motion-derived proxy labels, the best validation results were:

- all features: AUROC 0.8600, AP 0.8465;
- terrain-only (`step`, `slope`, `roughness`, `curvature`): AUROC 0.5647;
- motion-only (speed, yaw rate, IMU, time interval): AUROC 0.8695, AP 0.8434.

These results are diagnostic, not a traversability claim. The positive labels
are based on sustained future motion, while the inputs include current motion
signals, so the experiment has a label/input shortcut. The `-1` samples are
unknown rather than verified negatives, and the PU loss can report zero after
non-negative risk clipping. Do not use these scores as the final model
selection criterion.

Next priorities are to improve terrain descriptors, add independently
observable failure or risk signals, and repeat the comparison with sequence-
level splits and a static LeSTA baseline. Record AUROC, AP, calibration,
latency, dropped frames, and downstream planning/vehicle-stability metrics.

The enhanced 21-feature rerun added four derived motion-risk features
(`longitudinal_accel`, `lateral_accel`, `accel_norm`, `gyro_norm`) and four
additional terrain summaries. With the same 30-epoch CUDA experiment, the
best validation AUROC was 0.8330 for all features, 0.5828 for terrain-only,
and 0.8700 for motion-only. The result does not yet support using the terrain
features for model selection: the labels are still motion-derived and the
new risk quantities are also proxies. Preserve this ablation when adding
real failure, slip, emergency-stop, or path-deviation events.

The first leave-one-sequence-out evaluation used two rampway sequences for
training and one for validation. Best validation scores were AUROC/AP
0.7093/0.7347 on `rampway3`, 0.7566/0.6001 on `rampway2`, and 0.8613/0.6508
on `rampway1`. The large variation confirms that the earlier single-sequence
scores overstate generalization. Keep sequence-level splits for future model
comparisons and report per-sequence results, not only an aggregate average.

During the corrected terrain-feature evaluation, `step` was initially zero
for every frame because integer grid indices were divided by `cell_size` a
second time. The index conversion is now fixed and all three rampway
sequences were regenerated. Terrain-only leave-one-sequence-out results are
AUROC/AP 0.5621/0.6027 (`rampway3`), 0.8601/0.7659 (`rampway2`), and
0.6758/0.5460 (`rampway1`), showing substantial sequence sensitivity. These
are diagnostic proxy-label results; rerun all-feature comparisons after this
fix before selecting a model.

The causal-alignment fix keeps future PCHIP fields only for offline label
generation. Model windows now use the latest IMU and odometry samples at or
before each LiDAR timestamp, reject odometry older than 200 ms, and record
sensor age. After regeneration, the three rampway sets contain 2,658, 879,
and 1,593 valid ten-frame windows respectively; all 21 features are finite
and no aligned sensor timestamp is later than its LiDAR timestamp. Re-run the
cross-sequence model comparison with these causal windows before interpreting
performance.

After regenerating the corrected features, the all-feature leave-one-sequence-
out rerun achieved AUROC/AP 0.6246/0.6677 (`rampway3`), 0.5947/0.4560
(`rampway2`), and 0.8596/0.6525 (`rampway1`). The mean AUROC/AP is 0.6930/
0.5921, with strong split dependence. These results do not establish robust
cross-environment traversability prediction; they motivate better labels,
sequence balancing, and evaluation against independently observed failures.

## Build, Test, and Development Commands

From the workspace root (or inside the provided Docker container):

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The Docker workflow is `docker compose build lesta` followed by `docker compose run --rm lesta`. The build uses CUDA 12.9, ROS 2 Humble, Grid Map, PCL, and the LibTorch installation bundled with the PyTorch base image. Install the Python package with `pip install -e src/LeSTA/pylesta`.

Run training with `python src/LeSTA/pylesta/tools/train.py --config ...`; use `pylesta/configs/lesta.yaml` for model and training settings. Launch ROS workflows with `ros2 launch lesta label_generation.launch.py`, `traversability_prediction.launch.py`, or `traversability_mapping.launch.py`.

## Coding Style and Naming

Use four spaces in Python and conventional C++ formatting consistent with the existing code. Use `PascalCase` for C++ classes, `camelCase` for C++ methods/variables, `snake_case` for Python functions and variables, and descriptive lowercase names for YAML files. Keep public C++ interfaces in the matching `include/` tree and implementations in `src/`.

## Testing Guidelines

There is currently no dedicated unit-test suite or coverage threshold. Before submitting changes, build with `colcon`, run the Python smoke test against a representative checkpoint, and for ROS changes replay a suitable rosbag and verify topics and RViz output. Do not commit bags, point clouds, checkpoints, or generated `build/`, `install/`, and `log/` directories.

## Commits and Pull Requests

Recent commits use short imperative subjects such as `Docs: Update README.md`, `fix (link): ...`, and `refactoring`; follow that concise style and keep unrelated changes separate. Pull requests should explain the behavior change, list validation commands and environment assumptions, link an issue when applicable, and include screenshots or RViz evidence for visualization/launch changes. Update relevant README or YAML documentation when interfaces or parameters change.
