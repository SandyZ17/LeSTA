# Dataset Preprocessing

This directory contains download and conversion tools for the temporal LeSTA
experiments. Raw bags and generated files are intentionally kept out of Git.

## Dataset

The first dataset is the official Urban Traversability Dataset used with
LeSTA. It provides Velodyne LiDAR packets, IMU, `/tf`, and `/tf_static` in ROS 1
bag files. The original data and download links are documented at:

<https://github.com/Ikhyeon-Cho/urban-traversability-dataset>

## Download

From this directory, download the small parking-lot sequence first:

```bash
python3 download_urban_traversability.py --sequence parking_lot
```

The script uses the official Google Drive file ID and requires `gdown`.
Additional sequences can be added to the script after their official file IDs
have been verified; do not download the full campus sequence before the
pipeline is validated.

## ROS 1 packet decoding

### Original ROS1 LeSTA reproduction

The `lesta_ros1` branch includes the original ROS1 packages and a separate
LibTorch-enabled image. Build it from the repository root:

```bash
docker compose -f docker-compose.ros1.yml build lesta-ros1
docker compose -f docker-compose.ros1.yml up -d lesta-ros1
```

Place the original TorchScript checkpoint at
`models/LeSTA_campus_road_libtorch.pt`; it is mounted inside the container as
`/root/Downloads/LeSTA_campus_road_libtorch.pt`, matching the original YAML.
Inside the container, source the workspace before launching ROS:

```bash
source /opt/ros/noetic/setup.bash
source /workspace/ros_ws/devel/setup.bash
roscore
```

In another terminal, start the original node:

```bash
docker exec -it lesta-ros1 bash
source /opt/ros/noetic/setup.bash
source /workspace/ros_ws/devel/setup.bash
roslaunch lesta traversability_prediction.launch
```

In a third terminal, start the Velodyne decoder and replay the original
Rampway2 packet bag:

```bash
docker exec -it lesta-ros1 bash
source /opt/ros/noetic/setup.bash
roslaunch /workspace/LeSTA/data_preprocessing/ros1/vlp16packet_to_pointcloud.launch \
  packet_topic:=/velodyne_tilted/velodyne_packets
```

Then run:

```bash
rosbag play /workspace/bags/multi_level_rampway/multi_level_rampway2.orig.bag \
  --clock -r 3
```

The original ROS1 output is published on
`/lesta/prediction/traversability`. Compare it with the ROS2 output using the
same bag segment, playback rate, and RViz fixed frame.

Build the isolated decoder image from this directory:

```bash
docker build -f ros1/Dockerfile -t lesta-ros1-decoder:latest .
docker run --rm --name lesta-ros1-extract \
  -v "$PWD/datasets/urban_traversability/raw":/data:ro \
  -v "$PWD/datasets/urban_traversability/frames":/output \
  lesta-ros1-decoder:latest \
  /workspace/data_preprocessing/ros1/extract_frames.sh \
  /data/parking_lot_traversability_01-2.bag /output/parking_lot 1
```

The current parking-lot extraction produces 9,177 timestamped binary PCD
frames and a matching `manifest.csv`. Generated data is ignored by Git.

Extract auxiliary data from the same bag after the PCD extraction:

```bash
docker run --rm --entrypoint /bin/bash \
  -v "$PWD/datasets/urban_traversability/raw":/data:ro \
  -v "$PWD/datasets/urban_traversability/frames":/output \
  lesta-ros1-decoder:latest -lc \
  'source /opt/ros/noetic/setup.bash && python3 \
   /workspace/data_preprocessing/ros1/extract_auxiliary_data.py \
   /data/parking_lot_traversability_01-2.bag /output/parking_lot'
```

This writes `imu.csv`, `odometry_wheel.csv`, `cmd_vel.csv`, `tf.csv`,
`tf_static.csv`, and `aligned.csv` beside the PCD frames. `aligned.csv`
keeps PCHIP-interpolated pose/speed fields for offline future-motion label
generation, but temporal model inputs use `linear_x_causal` and
`angular_z_causal`, which only read samples at or before each LiDAR timestamp.
It also contains causal IMU values and the sensor ages for each LiDAR frame.
PCHIP remains available for offline label generation; odometry gaps over
200 ms are marked invalid for model inputs.

## Terrain features and temporal windows

Extract lightweight terrain geometry from the binary PCD frames. The default
ROI is `x=[0, 8] m`, `|y|<=4 m` in the LiDAR frame and uses 20 cm XY cells:

```bash
python3 extract_terrain_features.py \
  --frame-dir datasets/urban_traversability/frames/parking_lot
```

Generate conservative self-supervised labels from sustained future motion:

```bash
python3 generate_traversability_labels.py \
  --frame-dir datasets/urban_traversability/frames/parking_lot
```

Frames with at least 1 s of valid data, 0.25 m displacement, and sustained
forward speed are labeled `1`; all other frames remain `-1` (unknown). This
dataset has no explicit failure signal, so this step does not invent negative
labels from terrain thresholds.

Build the first 10-frame temporal sample set from aligned motion/IMU data and
the extracted terrain features:

```bash
python3 build_temporal_windows.py \
  --frame-dir datasets/urban_traversability/frames/parking_lot \
  --window-size 10 --stride 1
```

The generated `temporal/temporal_windows.npz` contains 21 features per frame
and a target `labels` array. Features include speed, yaw rate, IMU, frame
interval, derived longitudinal/lateral acceleration, acceleration and gyro
norms, and eight terrain features including `step`, `slope`, `roughness`, and
`curvature`.

For costmap-compatible temporal prediction, build per-cell samples aligned to
the map frame:

```bash
python3 build_per_cell_temporal_windows.py \
  --frame-dir datasets/urban_traversability/frames/multi_level_rampway \
  --window-size 10 --stride 1
```

This writes `per_cell_temporal/temporal_cell_windows.npz` with shape
`(samples, 10, 21)`. Each sample represents one map cell and its label is
assigned from the future vehicle footprint; unvisited cells remain unknown.
The existing frame-level windows remain available as a diagnostic baseline.

Run the lightweight initial Temporal MLP experiment from the repository root:

```bash
python3 pylesta/tools/train_temporal_mlp.py \
  --windows data_preprocessing/datasets/urban_traversability/frames/parking_lot/temporal/temporal_windows.npz \
  --output logs/temporal_mlp.pt
```

The experiment uses chronological train/validation splitting and PU loss;
`-1` labels are treated as unknown rather than as negative terrain labels.

For cross-sequence evaluation, pass two files to `--windows` and one complete
held-out sequence to `--val-windows`. This is the required evaluation mode
once multiple driving sequences are available.

For the required GPU run, use the CUDA-enabled Compose profile:

```bash
docker compose --profile training run --rm temporal-train
```

Replay a saved Temporal MLP checkpoint sequentially over a causal validation
sequence and measure ranking quality and GPU latency:

```bash
docker compose --profile training run --rm temporal-train \
  python3 /workspace/lesta_ws/src/LeSTA/pylesta/tools/evaluate_temporal_replay.py \
  --checkpoint /workspace/lesta_ws/src/LeSTA/logs/cv_causal_all_train12_val3.pt \
  --windows /workspace/lesta_ws/src/LeSTA/data_preprocessing/datasets/urban_traversability/frames/multi_level_rampway3/temporal/temporal_windows.npz \
  --device cuda
```

To override training arguments, provide the full command explicitly because
Compose treats arguments after the service name as a command replacement:

```bash
docker compose --profile training run --rm temporal-train \
  python3 /workspace/lesta_ws/src/LeSTA/pylesta/tools/train_temporal_mlp.py \
  --windows /workspace/lesta_ws/src/LeSTA/data_preprocessing/datasets/urban_traversability/frames/parking_lot/temporal/temporal_windows.npz \
  --output /workspace/lesta_ws/src/LeSTA/logs/temporal_mlp_terrain.pt \
  --device cuda --epochs 30 --feature-mode terrain
```

The service passes through all NVIDIA GPUs and the script fails explicitly if
CUDA is not visible. Verify that NVIDIA Container Toolkit is installed.

The training script also supports input ablations. Use `--feature-mode terrain`
to keep only the four terrain features, or `--feature-mode motion` to keep
speed, yaw rate, IMU, and time interval. It reports AUROC and average
precision, but these metrics evaluate the motion-derived proxy labels rather
than independently verified traversability labels.

## Planned conversion

The next preprocessing step will convert ROS 1 Velodyne packets into ordered
point-cloud frames, pair them with timestamps and poses, and generate temporal
windows for the Temporal MLP. Do not randomly split adjacent frames; split by
complete driving sequence.
