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
