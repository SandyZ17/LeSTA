#!/usr/bin/env python3
"""Download official Urban Traversability Dataset sequences from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FILES = {
    "parking_lot": (
        "parking_lot_traversability_01-2.bag",
        "16uqGjp5Svjoft2CSFxC_qJuc2IymwIeD",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=sorted(FILES), default="parking_lot")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/urban_traversability"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename, file_id = FILES[args.sequence]
    command = [
        sys.executable,
        "-m",
        "gdown",
        f"https://drive.google.com/uc?id={file_id}",
        "-O",
        str(args.output_dir / filename),
        "--continue",
    ]
    print("Downloading official dataset file into", args.output_dir / filename)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
