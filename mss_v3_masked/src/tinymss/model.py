from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int) -> int:
    for groups in (8, 6, 4, 3, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Downsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskResidualBlock2d(nn.Module):
    """Depthwise-separable residual block: large context without wasting params."""

    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.in_proj = nn.Conv2d(channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden,
        )
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.norm(x))
        y = self.in_proj(y)
        y = F.silu(self.depthwise(y))
        y = self.out_proj(y)
        return x + self.scale * y


class AxialAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(channels)
        self.time_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.freq_norm = nn.LayerNorm(channels)
        self.freq_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, freq, frames = x.shape

        xt = x.permute(0, 2, 3, 1).reshape(bsz * freq, frames, channels)
        xtn = self.time_norm(xt)
        xt = xt + self.time_attn(xtn, xtn, xtn, need_weights=False)[0]
        x = xt.reshape(bsz, freq, frames, channels).permute(0, 3, 1, 2)

        xf = x.permute(0, 3, 2, 1).reshape(bsz * frames, freq, channels)
        xfn = self.freq_norm(xf)
        xf = xf + self.freq_attn(xfn, xfn, xfn, need_weights=False)[0]
        xf = xf + self.ffn(xf)
        return xf.reshape(bsz, frames, freq, channels).permute(0, 3, 2, 1)


class WaveContext(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        widths = [32, 64, 96, out_channels]
        layers: list[nn.Module] = []
        in_channels = 2
        for width in widths:
            layers.extend(
                [
                    nn.Conv1d(in_channels, width, kernel_size=9, stride=4, padding=4),
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(),
                ]
            )
            in_channels = width
        self.encoder = nn.Sequential(*layers)
        self.to_film = nn.Conv1d(out_channels, out_channels * 2, kernel_size=1)

    def forward(self, mixture: torch.Tensor, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(mixture.float())
        context = F.interpolate(context, size=frames, mode="linear", align_corners=False)
        gamma, beta = self.to_film(context).chunk(2, dim=1)
        return gamma, beta


class TinyHybridMSS(nn.Module):
    """Small competitive-mask separator.

    The previous hybrid refiner could learn the bad shortcut "copy the mixture to
    every source". This model predicts source masks with softmax over sources, so
    each time-frequency bin is shared between stems instead of duplicated.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        sources: Sequence[str] = ("vocals", "drums", "bass", "other"),
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        base_channels: int = 56,
        channel_mults: Sequence[int] = (1, 2, 4, 6),
        bottleneck_layers: int = 2,
        attention_heads: int = 4,
        mask_temperature: float = 1.0,
        gradient_checkpointing: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.sources = list(sources)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mask_temperature = mask_temperature
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        widths = [base_channels * int(mult) for mult in channel_mults]
        self.in_proj = ConvNormAct2d(8, widths[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level, width in enumerate(widths):
            dilation = 2**min(level, 4)
            self.encoders.append(
                nn.Sequential(
                    MaskResidualBlock2d(width, dilation=dilation),
                    MaskResidualBlock2d(width, dilation=dilation),
                )
            )
            if level < len(widths) - 1:
                self.downs.append(Downsample2d(width, widths[level + 1]))

        bottleneck_width = widths[-1]
        self.wave_context = WaveContext(bottleneck_width)
        self.bottleneck = nn.Sequential(
            *[AxialAttentionBlock(bottleneck_width, attention_heads) for _ in range(bottleneck_layers)]
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = bottleneck_width
        for skip_width in reversed(widths[:-1]):
            self.up_projs.append(ConvNormAct2d(current + skip_width, skip_width))
            self.decoders.append(
                nn.Sequential(
                    MaskResidualBlock2d(skip_width),
                    MaskResidualBlock2d(skip_width),
                )
            )
            current = skip_width

        self.mask_head = nn.Sequential(
            ConvNormAct2d(current, current),
            nn.Conv2d(current, len(self.sources) * 2, kernel_size=1),
        )

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _stft(self, mixture: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames = mixture.shape
        spec = torch.stft(
            mixture.float().reshape(bsz * channels, frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return spec.reshape(bsz, channels, spec.shape[-2], spec.shape[-1])

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        bsz, source_count, channels, freq, frames = spec.shape
        wav = torch.istft(
            spec.reshape(bsz * source_count * channels, freq, frames),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        return wav.reshape(bsz, source_count, channels, length)

    def _features(self, spec: torch.Tensor) -> torch.Tensor:
        mag = spec.abs().clamp_min(1e-8)
        phase = spec / mag
        compressed = mag.pow(0.3) * phase
        real_imag = torch.view_as_real(compressed).permute(0, 1, 4, 2, 3).reshape(
            spec.shape[0], spec.shape[1] * 2, spec.shape[2], spec.shape[3]
        )
        logmag = torch.log1p(mag)
        mid = 0.5 * (spec[:, 0] + spec[:, 1])
        side = 0.5 * (spec[:, 0] - spec[:, 1])
        midside = torch.stack([torch.log1p(mid.abs()), torch.log1p(side.abs())], dim=1)
        return torch.cat([real_imag, logmag, midside], dim=1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        length = mixture.shape[-1]
        spec = self._stft(mixture)
        x = self.in_proj(self._features(spec))

        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = self._maybe_checkpoint(encoder, x)
            skips.append(x)
            if level < len(self.downs):
                x = self.downs[level](x)

        gamma, beta = self.wave_context(mixture, x.shape[-1])
        x = x * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        x = self._maybe_checkpoint(self.bottleneck, x)

        for up, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._maybe_checkpoint(decoder, up(x))

        logits = self.mask_head(x)
        if logits.shape[-2:] != spec.shape[-2:]:
            logits = F.interpolate(logits, size=spec.shape[-2:], mode="bilinear", align_corners=False)

        bsz, _, freq, frames = logits.shape
        source_count = len(self.sources)
        logits = logits.reshape(bsz, source_count, 2, freq, frames)
        temperature = max(float(self.mask_temperature), 1e-4)
        masks = torch.softmax(logits.float() / temperature, dim=1)
        estimate_spec = masks.to(spec.real.dtype) * spec.unsqueeze(1)
        return self._istft(estimate_spec, length)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
