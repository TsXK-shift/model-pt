from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinymss.audio import crop_or_pad, ensure_stereo, peak_normalize, save_audio
from tinymss.model import TinyHybridMSS


def load_checkpoint(path: str, device: torch.device) -> tuple[TinyHybridMSS, dict]:
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = TinyHybridMSS(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def separate_track(
    model: TinyHybridMSS,
    mixture: torch.Tensor,
    segment_frames: int,
    overlap: float,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")

    source_count = len(model.sources)
    channels, total_frames = mixture.shape
    hop = max(1, int(round(segment_frames * (1.0 - overlap))))
    window = torch.hann_window(segment_frames, periodic=False).clamp_min(1e-4)
    out = torch.zeros(source_count, channels, total_frames + segment_frames)
    weight = torch.zeros(1, 1, total_frames + segment_frames)

    starts = list(range(0, total_frames, hop))
    for start in tqdm(starts, desc="separating"):
        chunk = crop_or_pad(mixture[:, start : start + segment_frames], segment_frames)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            estimate = model(chunk.unsqueeze(0).to(device))[0].float().cpu()
        end = start + segment_frames
        out[..., start:end] += estimate * window.view(1, 1, -1)
        weight[..., start:end] += window.view(1, 1, -1)

    out = out[..., :total_frames] / weight[..., :total_frames].clamp_min(1e-6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    sample_rate = int(cfg["model"]["sample_rate"])

    mixture, sr = torchaudio.load(args.input)
    mixture = ensure_stereo(mixture)
    if sr != sample_rate:
        mixture = torchaudio.functional.resample(mixture, sr, sample_rate)

    segment_frames = int(round(args.segment_seconds * sample_rate))
    estimates = separate_track(
        model=model,
        mixture=mixture,
        segment_frames=segment_frames,
        overlap=args.overlap,
        device=device,
        amp=not args.no_amp,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(model.sources):
        wav = estimates[idx]
        if args.normalize:
            wav = peak_normalize(wav)
        save_audio(out_dir / f"{source}.wav", wav, sample_rate)


if __name__ == "__main__":
    main()

