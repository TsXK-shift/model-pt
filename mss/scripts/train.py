from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import os

class _PlainProgress:
    def __init__(self, iterable, desc=None, disable=False, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def set_postfix(self, **kwargs):
        pass

def tqdm(iterable, desc=None, disable=False, **kwargs):
    return _PlainProgress(iterable, desc=desc, disable=disable, **kwargs)



# --- PLAIN_PROGRESS_START ---
import os

class _PlainProgress:
    def __init__(self, iterable, desc=None, disable=False, **kwargs):
        self.iterable = iterable
        self.desc = desc or "progress"
        self.disable = disable
        self.count = 0
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = "?"
        self.every = int(os.environ.get("PROGRESS_EVERY", "10"))

    def __iter__(self):
        for item in self.iterable:
            self.count += 1
            if (not self.disable) and self.every > 0 and self.count % self.every == 0:
                print(f"{self.desc} step={self.count}/{self.total}", flush=True)
            yield item

    def set_postfix(self, **kwargs):
        if self.disable:
            return
        parts = []
        for key, value in kwargs.items():
            try:
                parts.append(f"{key}={float(value):.4f}")
            except Exception:
                parts.append(f"{key}={value}")
        print(f"{self.desc} step={self.count}/{self.total} " + " ".join(parts), flush=True)

def tqdm(iterable, desc=None, disable=False, **kwargs):
    return _PlainProgress(iterable, desc=desc, disable=disable, **kwargs)
# --- PLAIN_PROGRESS_END ---

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinymss.data import MUSDB18HQDataset
from tinymss.losses import MultiDomainLoss
from tinymss.metrics import si_sdr
from tinymss.model import TinyHybridMSS, count_parameters
from tinymss.train_utils import (
    EMA,
    cleanup_distributed,
    clone_config,
    is_main_process,
    load_config,
    reduce_mean,
    save_json,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


def build_datasets(cfg: dict[str, Any], data_override: str | None) -> tuple[MUSDB18HQDataset, MUSDB18HQDataset]:
    data_cfg = cfg["data"]
    root = data_override or data_cfg["root"]
    train_set = MUSDB18HQDataset(
        root=root,
        split="train",
        sources=data_cfg["sources"],
        sample_rate=int(data_cfg["sample_rate"]),
        segment_seconds=float(data_cfg["segment_seconds"]),
        samples_per_epoch=int(data_cfg["train_samples_per_epoch"]),
        augment=data_cfg.get("augment", {}),
        source_remix_prob=float(data_cfg.get("source_remix_prob", 0.0)),
    )
    valid_set = MUSDB18HQDataset(
        root=root,
        split="test",
        sources=data_cfg["sources"],
        sample_rate=int(data_cfg["sample_rate"]),
        segment_seconds=float(data_cfg["segment_seconds"]),
        samples_per_epoch=int(data_cfg["valid_samples_per_epoch"]),
        augment={"enabled": False},
        source_remix_prob=0.0,
    )
    return train_set, valid_set


def make_loader(
    dataset: MUSDB18HQDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA | None,
    cfg: dict[str, Any],
    epoch: int,
    best_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "config": cfg,
            "epoch": epoch,
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_loss", math.inf))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: MultiDomainLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA | None,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int,
    cfg: dict[str, Any],
) -> dict[str, float]:
    model.train()
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0))
    log_every = int(cfg["train"].get("log_every", 25))
    running = {"loss": 0.0, "time": 0.0, "stft": 0.0}
    optimizer.zero_grad(set_to_none=True)

    simple_log = os.environ.get("SIMPLE_LOG", "0") == "1"
    iterator = tqdm(loader, desc=f"epoch {epoch} train", disable=(not is_main_process(rank)) or simple_log, mininterval=10.0)
    for step, batch in enumerate(iterator, start=1):
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["sources"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            estimate = model(mixture)
            losses = loss_fn(estimate, target)
            loss = losses.total / grad_accum

        scaler.scale(loss).backward()
        if step % grad_accum == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(unwrap_model(model))

    if len(loader) % grad_accum != 0:
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(unwrap_model(model))

        running["loss"] += float(losses.total.detach())
        running["time"] += float(losses.time_l1.detach())
        running["stft"] += float((losses.spectral_logmag + losses.spectral_convergence).detach())

        if is_main_process(rank) and step % log_every == 0:
            msg = (
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={running['loss'] / step:.4f} "
                f"time={running['time'] / step:.4f} "
                f"stft={running['stft'] / step:.4f}"
            )
            if simple_log:
                print(msg, flush=True)
            else:
                iterator.set_postfix(loss=running["loss"] / step, time=running["time"] / step, stft=running["stft"] / step)

    count = torch.tensor(float(len(loader)), device=device)
    totals = torch.tensor([running["loss"], running["time"], running["stft"]], device=device)
    totals = reduce_mean(totals, world_size)
    count = reduce_mean(count, world_size)
    return {"loss": float(totals[0] / count), "time_l1": float(totals[1] / count), "stft": float(totals[2] / count)}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: MultiDomainLoss,
    ema: EMA | None,
    device: torch.device,
    rank: int,
    world_size: int,
    use_amp: bool,
) -> dict[str, float]:
    raw_model = unwrap_model(model)
    if ema is not None:
        ema.apply_to(raw_model)
    model.eval()
    totals = torch.zeros(3, device=device)

    simple_log = os.environ.get("SIMPLE_LOG", "0") == "1"
    iterator = tqdm(loader, desc="valid", disable=(not is_main_process(rank)) or simple_log, mininterval=10.0)
    for batch in iterator:
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["sources"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            estimate = model(mixture)
            losses = loss_fn(estimate, target)
        score = si_sdr(estimate, target).mean()
        totals += torch.tensor([float(losses.total), float(losses.time_l1), float(score)], device=device)

    totals = reduce_mean(totals, world_size)
    count = reduce_mean(torch.tensor(float(len(loader)), device=device), world_size)
    if ema is not None:
        ema.restore(raw_model)
    return {"loss": float(totals[0] / count), "time_l1": float(totals[1] / count), "si_sdr": float(totals[2] / count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="runs/tiny-hybrid")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data is not None:
        cfg = clone_config(cfg)
        cfg["data"]["root"] = args.data

    distributed, rank, local_rank, world_size = setup_distributed()
    seed_everything(int(cfg.get("seed", 1337)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)

    train_set, valid_set = build_datasets(cfg, args.data)
    train_loader, train_sampler = make_loader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        distributed=distributed,
        shuffle=True,
    )
    valid_loader, valid_sampler = make_loader(
        valid_set,
        batch_size=1,
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        distributed=distributed,
        shuffle=False,
    )

    model = TinyHybridMSS(**cfg["model"]).to(device)
    if is_main_process(rank):
        params = count_parameters(model)
        print(f"Trainable parameters: {params:,} ({params / 1_000_000:.2f}M)")
        save_json(out_dir / "resolved_config.json", cfg)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    loss_fn = MultiDomainLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    ema = EMA(unwrap_model(model), ema_decay) if ema_decay > 0 else None

    start_epoch = 0
    best_loss = math.inf
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, ema, device)
        if is_main_process(rank):
            print(f"Resumed from epoch {start_epoch}, best valid loss {best_loss:.4f}")

    bad_epochs = 0
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if valid_sampler is not None:
            valid_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, ema, device, epoch, rank, world_size, cfg
        )
        scheduler.step()

        should_validate = (epoch + 1) % int(cfg["train"].get("validate_every", 1)) == 0
        valid_stats = {"loss": math.inf, "time_l1": math.inf, "si_sdr": -math.inf}
        if should_validate:
            valid_stats = validate(
                model,
                valid_loader,
                loss_fn,
                ema,
                device,
                rank,
                world_size,
                use_amp=amp_enabled,
            )

        if is_main_process(rank):
            print(
                f"epoch={epoch} train_loss={train_stats['loss']:.4f} "
                f"valid_loss={valid_stats['loss']:.4f} valid_si_sdr={valid_stats['si_sdr']:.2f}dB"
            )

            save_checkpoint(
                out_dir / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                cfg,
                epoch,
                best_loss,
            )

            if valid_stats["loss"] < best_loss:
                best_loss = valid_stats["loss"]
                bad_epochs = 0
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    cfg,
                    epoch,
                    best_loss,
                )
            else:
                bad_epochs += 1

            stop_now = bad_epochs >= int(cfg["train"].get("early_stopping_patience", 999999))
            if stop_now:
                print("Early stopping triggered.")
        else:
            stop_now = False

        stop_tensor = torch.tensor(1 if stop_now else 0, device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            break

    cleanup_distributed()


if __name__ == "__main__":
    main()