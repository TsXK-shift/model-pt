from __future__ import annotations

import torch


def apply_softmask(
    mixture: torch.Tensor,
    estimates: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project estimates back onto the mixture with source-exclusive soft masks.

    The network predicts each source waveform independently. When a weak
    checkpoint copies too much of the mixture into every source, this cleanup
    uses the predicted source magnitudes only as masks and reconstructs stems
    from the original mixture phase. It cannot create missing separation, but it
    reduces the common "same song in every stem" failure mode.
    """

    if mixture.dim() != 2:
        raise ValueError(f"Expected mixture shape [channels, time], got {tuple(mixture.shape)}")
    if estimates.dim() != 3:
        raise ValueError(f"Expected estimates shape [sources, channels, time], got {tuple(estimates.shape)}")
    if estimates.shape[1:] != mixture.shape:
        raise ValueError("Mixture and estimate shapes do not match.")

    power = max(0.25, float(power))
    window = torch.hann_window(win_length, device=mixture.device, dtype=mixture.dtype)
    frames = mixture.shape[-1]

    mix_spec = torch.stft(
        mixture,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    source_count, channels, _ = estimates.shape
    est_spec = torch.stft(
        estimates.reshape(source_count * channels, frames),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    ).reshape(source_count, channels, mix_spec.shape[-2], mix_spec.shape[-1])

    weights = est_spec.abs().clamp_min(eps).pow(power)
    masks = weights / weights.sum(dim=0, keepdim=True).clamp_min(eps)
    refined_spec = masks * mix_spec.unsqueeze(0)
    refined = torch.istft(
        refined_spec.reshape(source_count * channels, mix_spec.shape[-2], mix_spec.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=frames,
    ).reshape(source_count, channels, frames)
    return refined
