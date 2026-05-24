from __future__ import annotations

import math
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


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct2d(channels, channels, dilation=dilation),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


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
        context = self.encoder(mixture)
        context = F.interpolate(context, size=frames, mode="linear", align_corners=False)
        gamma, beta = self.to_film(context).chunk(2, dim=1)
        return gamma, beta


class SpectralUNet(nn.Module):
    def __init__(
        self,
        sources: Sequence[str],
        n_fft: int,
        hop_length: int,
        win_length: int,
        base_channels: int,
        channel_mults: Sequence[int],
        bottleneck_layers: int,
        attention_heads: int,
        mask_limit: float,
    ) -> None:
        super().__init__()
        self.sources = list(sources)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mask_limit = mask_limit
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        widths = [base_channels * mult for mult in channel_mults]
        self.in_proj = ConvNormAct2d(6, widths[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level, width in enumerate(widths):
            self.encoders.append(nn.Sequential(ResidualBlock2d(width, 2**level), ResidualBlock2d(width, 2**level)))
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
            self.decoders.append(nn.Sequential(ResidualBlock2d(skip_width), ResidualBlock2d(skip_width)))
            current = skip_width

        out_channels = len(self.sources) * 2 * 2
        self.mask_head = nn.Sequential(
            ConvNormAct2d(current, current),
            nn.Conv2d(current, out_channels, kernel_size=1),
        )

    def _stft(self, mixture: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames = mixture.shape
        spec = torch.stft(
            mixture.reshape(bsz * channels, frames),
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
        logmag = torch.log1p(mag).reshape(spec.shape[0], spec.shape[1], spec.shape[2], spec.shape[3])
        return torch.cat([real_imag, logmag], dim=1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        length = mixture.shape[-1]
        spec = self._stft(mixture)
        x = self.in_proj(self._features(spec))

        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if level < len(self.downs):
                x = self.downs[level](x)

        gamma, beta = self.wave_context(mixture, x.shape[-1])
        x = x * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(up(x))

        mask = self.mask_head(x)
        if mask.shape[-2:] != spec.shape[-2:]:
            mask = F.interpolate(mask, size=spec.shape[-2:], mode="bilinear", align_corners=False)

        bsz, _, freq, frames = mask.shape
        source_count = len(self.sources)
        mask = mask.reshape(bsz, source_count, 2, 2, freq, frames).permute(0, 1, 2, 4, 5, 3).contiguous()
        mask = self.mask_limit * torch.view_as_complex(torch.tanh(mask.float()))
        estimate_spec = mask * spec.unsqueeze(1)
        return self._istft(estimate_spec, length)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(F.silu(self.norm(x)))
        y = F.glu(y, dim=1)
        y = self.pointwise(y)
        return residual + y


class TemporalRefiner(nn.Module):
    def __init__(
        self,
        source_count: int,
        hidden: int,
        layers: int,
        kernel_size: int,
        residual_scale: float,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.source_count = source_count
        self.residual_scale = residual_scale
        self.gradient_checkpointing = gradient_checkpointing
        in_channels = 2 + source_count * 2
        self.in_proj = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, kernel_size, dilation=2 ** (idx % 8)) for idx in range(layers)]
        )
        self.out = nn.Conv1d(hidden, source_count * 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mixture: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        bsz, source_count, channels, frames = coarse.shape
        x = torch.cat([mixture, coarse.reshape(bsz, source_count * channels, frames)], dim=1)
        x = self.in_proj(x)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        residual = self.out(x)
        residual = residual.reshape(bsz, source_count, channels, frames)
        return coarse + self.residual_scale * residual


class TinyHybridMSS(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        sources: Sequence[str],
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        base_channels: int = 24,
        channel_mults: Sequence[int] = (1, 2, 4, 6),
        bottleneck_layers: int = 2,
        attention_heads: int = 4,
        tcn_hidden: int = 192,
        tcn_layers: int = 12,
        tcn_kernel: int = 7,
        gradient_checkpointing: bool = True,
        residual_scale: float = 0.25,
        mask_limit: float = 2.0,
        mix_consistency: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.sources = list(sources)
        self.mix_consistency = mix_consistency
        self.spectral = SpectralUNet(
            sources=sources,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            base_channels=base_channels,
            channel_mults=channel_mults,
            bottleneck_layers=bottleneck_layers,
            attention_heads=attention_heads,
            mask_limit=mask_limit,
        )
        self.refiner = TemporalRefiner(
            source_count=len(self.sources),
            hidden=tcn_hidden,
            layers=tcn_layers,
            kernel_size=tcn_kernel,
            residual_scale=residual_scale,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        coarse = self.spectral(mixture)
        estimate = self.refiner(mixture, coarse)
        if self.mix_consistency:
            residual = mixture.unsqueeze(1) - estimate.sum(dim=1, keepdim=True)
            estimate = estimate + residual / len(self.sources)
        return estimate


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
