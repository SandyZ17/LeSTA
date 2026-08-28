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

The script uses the official Google Drive folder and requires `gdown`. Use
`--sequence campus_full` only after the preprocessing pipeline is verified;
the full sequence is much larger.

## Planned conversion

The next preprocessing step will convert ROS 1 Velodyne packets into ordered
point-cloud frames, pair them with timestamps and poses, and generate temporal
windows for the Temporal MLP. Do not randomly split adjacent frames; split by
complete driving sequence.
