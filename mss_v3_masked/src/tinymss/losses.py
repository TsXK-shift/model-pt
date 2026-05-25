from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    time_l1: torch.Tensor
    spectral_logmag: torch.Tensor
    spectral_convergence: torch.Tensor
    complex_l1: torch.Tensor
    ratio_mask: torch.Tensor


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Iterable[int],
        hop_ratio: float = 0.25,
        logmag_weight: float = 1.0,
        convergence_weight: float = 0.5,
        complex_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.fft_sizes = [int(size) for size in fft_sizes]
        self.hop_ratio = hop_ratio
        self.logmag_weight = logmag_weight
        self.convergence_weight = convergence_weight
        self.complex_weight = complex_weight
        for size in self.fft_sizes:
            self.register_buffer(f"window_{size}", torch.hann_window(size), persistent=False)

    def _stft(self, wav: torch.Tensor, fft_size: int) -> torch.Tensor:
        window = getattr(self, f"window_{fft_size}")
        hop = max(1, int(round(fft_size * self.hop_ratio)))
        flat = wav.float().reshape(-1, wav.shape[-1])
        return torch.stft(
            flat,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window=window,
            center=True,
            return_complex=True,
        )

    def forward(
        self, estimate: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        estimate = estimate.float()
        target = target.float()
        logmag_loss = estimate.new_tensor(0.0)
        convergence_loss = estimate.new_tensor(0.0)
        complex_loss = estimate.new_tensor(0.0)
        ratio_mask_loss = estimate.new_tensor(0.0)

        for fft_size in self.fft_sizes:
            est_spec = self._stft(estimate, fft_size)
            tgt_spec = self._stft(target, fft_size)
            est_mag = est_spec.abs().clamp_min(1e-7)
            tgt_mag = tgt_spec.abs().clamp_min(1e-7)

            logmag_loss = logmag_loss + F.l1_loss(torch.log1p(est_mag), torch.log1p(tgt_mag))
            diff_norm = torch.linalg.vector_norm(tgt_mag - est_mag, dim=(-2, -1))
            tgt_norm = torch.linalg.vector_norm(tgt_mag, dim=(-2, -1))
            active = tgt_norm > 1e-3
            if active.any():
                convergence = diff_norm[active] / tgt_norm[active].clamp_min(1e-3)
                convergence_loss = convergence_loss + convergence.clamp_max(10.0).mean()
            complex_loss = complex_loss + F.l1_loss(torch.view_as_real(est_spec), torch.view_as_real(tgt_spec))
            if estimate.dim() == 4 and target.dim() == 4:
                est_ratio_mag = est_mag.reshape(*estimate.shape[:-1], est_mag.shape[-2], est_mag.shape[-1])
                tgt_ratio_mag = tgt_mag.reshape(*target.shape[:-1], tgt_mag.shape[-2], tgt_mag.shape[-1])
                est_ratio = est_ratio_mag / est_ratio_mag.sum(dim=1, keepdim=True).clamp_min(1e-7)
                tgt_ratio = tgt_ratio_mag / tgt_ratio_mag.sum(dim=1, keepdim=True).clamp_min(1e-7)
                ratio_mask_loss = ratio_mask_loss + F.l1_loss(est_ratio, tgt_ratio)

        scale = 1.0 / len(self.fft_sizes)
        return logmag_loss * scale, convergence_loss * scale, complex_loss * scale, ratio_mask_loss * scale


class MultiDomainLoss(nn.Module):
    def __init__(
        self,
        time_l1: float,
        spectral_logmag: float,
        spectral_convergence: float,
        complex_l1: float,
        fft_sizes: Iterable[int],
        hop_ratio: float,
        ratio_mask: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_weight = time_l1
        self.logmag_weight = spectral_logmag
        self.convergence_weight = spectral_convergence
        self.complex_weight = complex_l1
        self.ratio_weight = ratio_mask
        self.stft_loss = MultiScaleSTFTLoss(
            fft_sizes=fft_sizes,
            hop_ratio=hop_ratio,
            logmag_weight=spectral_logmag,
            convergence_weight=spectral_convergence,
            complex_weight=complex_l1,
        )

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> LossBreakdown:
        time_l1 = F.l1_loss(estimate, target)
        logmag, convergence, complex_l1, ratio_mask = self.stft_loss(estimate, target)
        total = (
            self.time_weight * time_l1
            + self.logmag_weight * logmag
            + self.convergence_weight * convergence
            + self.complex_weight * complex_l1
            + self.ratio_weight * ratio_mask
        )
        return LossBreakdown(total, time_l1, logmag, convergence, complex_l1, ratio_mask)
