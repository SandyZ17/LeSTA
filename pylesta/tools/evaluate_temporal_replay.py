#!/usr/bin/env python3
"""Replay causal Temporal MLP windows and report quality and latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train_temporal_mlp import TemporalMLP, ranking_metrics


def load_windows(paths):
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    names = arrays[0]["feature_names"].tolist()
    if any(data["feature_names"].tolist() != names for data in arrays):
        raise ValueError("all window files must have identical feature names")
    return (np.concatenate([data["features"] for data in arrays]).astype(np.float32),
            np.concatenate([data["labels"] for data in arrays]).astype(np.int64), names)


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features, labels, names = load_windows(args.windows)
    if names != checkpoint["feature_names"]:
        raise ValueError("checkpoint and window feature names differ")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = TemporalMLP(len(names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    features = (features - checkpoint["mean"]) / checkpoint["std"]

    scores = []
    elapsed = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            batch = torch.from_numpy(features[start:start + args.batch_size]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            output = torch.sigmoid(model(batch))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - begin) / len(batch))
            scores.extend(output.cpu().numpy())

    scores = np.asarray(scores)
    auroc, ap = ranking_metrics(scores, labels)
    latency_ms = 1000.0 * float(np.mean(elapsed))
    print(f"device={device} windows={len(features)} features={len(names)}")
    print(f"auroc={auroc:.4f} ap={ap:.4f} mean_window_latency_ms={latency_ms:.4f} "
          f"throughput_windows_per_s={1000.0 / latency_ms:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    main(parser.parse_args())
