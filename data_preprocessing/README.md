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

## Planned conversion

The next preprocessing step will convert ROS 1 Velodyne packets into ordered
point-cloud frames, pair them with timestamps and poses, and generate temporal
windows for the Temporal MLP. Do not randomly split adjacent frames; split by
complete driving sequence.
