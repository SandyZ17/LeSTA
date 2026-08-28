#!/usr/bin/env python3
"""Small CPU-friendly Temporal MLP experiment for LeSTA windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TemporalMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence):
        encoded = self.frame_mlp(sequence)
        pooled = torch.cat((encoded.mean(dim=1), encoded.amax(dim=1), encoded[:, -1]), dim=1)
        return self.head(pooled).squeeze(1)


def pu_loss(logits, labels, positive_prior: float):
    positive = labels == 1
    unknown = labels == -1
    if not positive.any() or not unknown.any():
        raise ValueError("each split needs both positive and unknown samples")
    positive_risk = nn.functional.softplus(-logits[positive]).mean()
    unlabeled_negative_risk = nn.functional.softplus(logits[unknown]).mean()
    positive_negative_risk = nn.functional.softplus(logits[positive]).mean()
    risk = positive_prior * positive_risk + unlabeled_negative_risk - positive_prior * positive_negative_risk
    return torch.clamp(risk, min=0.0)


def evaluate(model, loader, device, positive_prior):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for features, target in loader:
            logits = model(features.to(device, non_blocking=True))
            scores.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(target.numpy())
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    logits = torch.logit(torch.from_numpy(scores).clamp(1e-6, 1 - 1e-6))
    loss = float(pu_loss(logits, torch.from_numpy(labels), positive_prior).detach())
    return loss, float(scores[labels == 1].mean()), float(scores[labels == -1].mean())


def main(args):
    torch.manual_seed(args.seed)
    data = np.load(args.windows, allow_pickle=False)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    split = int(len(features) * args.train_ratio)
    train_x, val_x = features[:split], features[split:]
    train_y, val_y = labels[:split], labels[split:]
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable. Run through Docker with --gpus all.")
    device = torch.device(args.device)
    print(f"Using device: {device}")
    loader_kwargs = {"pin_memory": device.type == "cuda"}
    train = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                       batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                     batch_size=args.batch_size, **loader_kwargs)
    model = TemporalMLP(features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train:
            if not ((batch_y == 1).any() and (batch_y == -1).any()):
                continue
            optimizer.zero_grad()
            loss = pu_loss(model(batch_x.to(device, non_blocking=True)),
                           batch_y.to(device, non_blocking=True), args.positive_prior)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        val_loss, pos_score, unknown_score = evaluate(model, val, device, args.positive_prior)
        print(f"epoch={epoch:02d} train_pu_loss={np.mean(train_losses):.5f} "
              f"val_pu_loss={val_loss:.5f} positive_score={pos_score:.4f} "
              f"unknown_score={unknown_score:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "feature_names": data["feature_names"].tolist()}, output)
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/temporal_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-prior", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    main(args)
