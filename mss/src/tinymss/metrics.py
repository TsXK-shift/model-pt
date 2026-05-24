from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / (target.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps))
    noise = estimate - projection
    ratio = projection.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))

