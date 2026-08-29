#!/usr/bin/env python3
"""Export a trained TemporalMLP checkpoint for the ROS2 LibTorch node."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from train_temporal_mlp import TemporalMLP


class InferenceModule(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_dim = len(checkpoint["feature_names"])
    model = TemporalMLP(feature_dim, args.hidden_dim)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    wrapper = InferenceModule(model, checkpoint["mean"], checkpoint["std"]).eval()
    example = torch.zeros(1, args.sequence_length, feature_dim, dtype=torch.float32)
    scripted = torch.jit.trace(wrapper, example, strict=True)
    scripted.save(str(args.output))
    print(f"Exported {args.output} with input [N,{args.sequence_length},{feature_dim}]")
    print("Features:", ", ".join(checkpoint["feature_names"]))


if __name__ == "__main__":
    main()
