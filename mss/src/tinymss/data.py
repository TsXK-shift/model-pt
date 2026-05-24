from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset

from .audio import crop_or_pad, ensure_stereo


@dataclass(frozen=True)
class TrackInfo:
    name: str
    path: Path
    frames: int
    sample_rate: int


def _db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


class MUSDB18HQDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        sources: Sequence[str],
        sample_rate: int,
        segment_seconds: float,
        samples_per_epoch: int | None,
        augment: dict[str, Any] | None = None,
        source_remix_prob: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.sources = list(sources)
        self.sample_rate = sample_rate
        self.segment_frames = int(round(segment_seconds * sample_rate))
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment or {"enabled": False}
        self.source_remix_prob = source_remix_prob if split == "train" else 0.0

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.tracks = self._scan_tracks()
        if not self.tracks:
            raise RuntimeError(f"No MUSDB tracks found in {self.split_dir}")

    def _scan_tracks(self) -> list[TrackInfo]:
        tracks: list[TrackInfo] = []
        for track_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            expected = ["mixture", *self.sources]
            missing = [s for s in expected if not (track_dir / f"{s}.wav").exists()]
            if missing:
                continue
            info = sf.info(str(track_dir / "mixture.wav"))
            tracks.append(
                TrackInfo(
                    name=track_dir.name,
                    path=track_dir,
                    frames=info.frames,
                    sample_rate=info.samplerate,
                )
            )
        return tracks

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.tracks)

    def _native_segment(self, track: TrackInfo, index: int | None = None) -> tuple[int, int]:
        native_frames = int(round(self.segment_frames * track.sample_rate / self.sample_rate))
        max_offset = max(0, track.frames - native_frames)
        if self.split == "train":
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
        else:
            # Stable validation windows make early stopping much less noisy.
            index = 0 if index is None else index
            offset = 0 if max_offset == 0 else (index * 9973) % (max_offset + 1)
        return offset, native_frames

    def _load_source(self, track: TrackInfo, source: str, offset: int, frames: int) -> torch.Tensor:
        audio, sr = sf.read(
            str(track.path / f"{source}.wav"),
            start=offset,
            frames=frames,
            always_2d=True,
            dtype="float32",
        )
        wav = torch.from_numpy(np.asarray(audio).T.copy())
        wav = ensure_stereo(wav)
        if sr != self.sample_rate:
            new_frames = max(1, int(round(wav.shape[-1] * self.sample_rate / sr)))
            flat = wav.reshape(-1, 1, wav.shape[-1])
            wav = F.interpolate(flat, size=new_frames, mode="linear", align_corners=False)
            wav = wav.reshape(2, -1)
        return crop_or_pad(wav, self.segment_frames)

    def _load_stems(self, index: int) -> torch.Tensor:
        base_track = self.tracks[index % len(self.tracks)]
        use_remix = random.random() < self.source_remix_prob
        stems = []
        if use_remix:
            for source_index, source in enumerate(self.sources):
                track = random.choice(self.tracks)
                offset, frames = self._native_segment(track, index + source_index)
                stems.append(self._load_source(track, source, offset, frames))
        else:
            offset, frames = self._native_segment(base_track, index)
            for source in self.sources:
                stems.append(self._load_source(base_track, source, offset, frames))
        return torch.stack(stems, dim=0)

    def _augment(self, stems: torch.Tensor) -> torch.Tensor:
        cfg = self.augment
        if not cfg.get("enabled", False):
            return stems

        gain_db = float(cfg.get("gain_db", 0.0))
        if gain_db > 0:
            gains = torch.empty(stems.shape[0], 1, 1).uniform_(-gain_db, gain_db)
            stems = stems * torch.pow(10.0, gains / 20.0)

        if random.random() < float(cfg.get("channel_swap_prob", 0.0)):
            stems = stems.flip(dims=[1])

        polarity_prob = float(cfg.get("polarity_flip_prob", 0.0))
        if polarity_prob > 0:
            signs = torch.where(
                torch.rand(stems.shape[0], 1, 1) < polarity_prob,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            )
            stems = stems * signs

        dropout_prob = float(cfg.get("source_dropout_prob", 0.0))
        if dropout_prob > 0:
            keep = (torch.rand(stems.shape[0], 1, 1) >= dropout_prob).float()
            if keep.sum() > 0:
                stems = stems * keep

        if random.random() < float(cfg.get("speed_perturb_prob", 0.0)):
            factor = random.uniform(float(cfg.get("speed_min", 0.98)), float(cfg.get("speed_max", 1.02)))
            stems = self._speed_perturb(stems, factor)

        return stems

    def _speed_perturb(self, stems: torch.Tensor, factor: float) -> torch.Tensor:
        # Interpolation is intentionally cheap for Kaggle CPU workers. It changes
        # speed and pitch together, which is acceptable as augmentation noise.
        source_count, channels, frames = stems.shape
        new_frames = max(16, int(math.ceil(frames / factor)))
        flat = stems.reshape(source_count * channels, 1, frames)
        stretched = F.interpolate(flat, size=new_frames, mode="linear", align_corners=False)
        stretched = stretched.reshape(source_count, channels, new_frames)
        if new_frames > frames:
            start = random.randint(0, new_frames - frames)
            return stretched[..., start : start + frames]
        return crop_or_pad(stretched, frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        stems = self._load_stems(index)
        if self.split == "train":
            stems = self._augment(stems)
        mixture = stems.sum(dim=0)
        return {"mixture": mixture, "sources": stems}
