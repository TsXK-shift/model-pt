from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


def ensure_stereo(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 2:
        raise ValueError(f"Expected audio shape [channels, time], got {tuple(wav.shape)}")
    if wav.shape[0] == 1:
        return wav.repeat(2, 1)
    if wav.shape[0] > 2:
        return wav[:2]
    return wav


def crop_or_pad(wav: torch.Tensor, frames: int) -> torch.Tensor:
    if wav.shape[-1] > frames:
        return wav[..., :frames]
    if wav.shape[-1] < frames:
        return F.pad(wav, (0, frames - wav.shape[-1]))
    return wav


def load_audio(
    path: str | Path,
    sample_rate: int,
    frame_offset: int = 0,
    num_frames: int = -1,
    target_frames: Optional[int] = None,
) -> torch.Tensor:
    frames = num_frames if num_frames is not None and num_frames >= 0 else -1
    audio, sr = sf.read(
        str(path),
        start=frame_offset,
        frames=frames,
        always_2d=True,
        dtype="float32",
    )
    wav = torch.from_numpy(np.asarray(audio).T.copy())
    wav = ensure_stereo(wav)
    if sr != sample_rate:
        new_frames = max(1, int(round(wav.shape[-1] * sample_rate / int(sr))))
        wav = F.interpolate(
            wav.reshape(-1, 1, wav.shape[-1]),
            size=new_frames,
            mode="linear",
            align_corners=False,
        ).reshape(2, -1)
    if target_frames is not None:
        wav = crop_or_pad(wav, target_frames)
    return wav


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().clamp(-1.0, 1.0)
    sf.write(str(path), wav.T.numpy(), sample_rate)


def peak_normalize(wav: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    max_abs = wav.abs().amax().clamp_min(1e-8)
    if max_abs <= peak:
        return wav
    return wav * (peak / max_abs)
