from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinymss.losses import MultiDomainLoss
from tinymss.model import TinyHybridMSS, count_parameters
from tinymss.train_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "kaggle_t4x2.yaml"))
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = TinyHybridMSS(**cfg["model"])
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,} ({params / 1_000_000:.2f}M)")

    sample_rate = int(cfg["model"]["sample_rate"])
    frames = int(args.seconds * sample_rate)
    mixture = torch.randn(1, 2, frames)
    with torch.no_grad():
        estimate = model(mixture)
    print(f"Input:  {tuple(mixture.shape)}")
    print(f"Output: {tuple(estimate.shape)}")

    loss_fn = MultiDomainLoss(**cfg["loss"])
    target = torch.randn_like(estimate)
    loss = loss_fn(estimate, target)
    print(f"Smoke loss: {loss.total.item():.4f}")


if __name__ == "__main__":
    main()

