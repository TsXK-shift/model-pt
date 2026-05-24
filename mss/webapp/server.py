from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(SRC))

APP_TITLE = "Tiny Hybrid MSS Console"
DEFAULT_WORK_DIR = Path(os.environ.get("TINYMSS_WEB_WORKDIR", ROOT / "web_runs"))
MAX_UPLOAD_MB = int(os.environ.get("TINYMSS_MAX_UPLOAD_MB", "512"))
EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOBS_LOCK = threading.Lock()
JOBS: dict[str, "JobState"] = {}


@dataclass
class JobState:
    id: str
    status: str
    created_at: float
    checkpoint: str
    input_name: str
    segment_seconds: float
    overlap: float
    normalize: bool
    amp: bool
    progress: float = 0.0
    message: str = "Queued"
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    finished_at: float | None = None


def _json_job(job: JobState) -> dict[str, Any]:
    payload = asdict(job)
    payload["elapsed_seconds"] = round((job.finished_at or time.time()) - job.created_at, 1)
    return payload


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in updates.items():
            setattr(job, key, value)


def _checkpoint_roots() -> list[Path]:
    env = os.environ.get("TINYMSS_CHECKPOINT_DIRS")
    roots: list[Path] = []
    if env:
        roots.extend(Path(item).expanduser() for item in env.split(os.pathsep) if item.strip())
    roots.extend(
        [
            ROOT / "runs" / "tiny-hybrid" / "checkpoints",
            ROOT / "export_checkpoint",
            ROOT.parent / "export_checkpoint",
            ROOT.parent / "runs" / "tiny-hybrid" / "checkpoints",
            Path.cwd() / "export_checkpoint",
            Path.cwd() / "runs" / "tiny-hybrid" / "checkpoints",
        ]
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _find_checkpoints() -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _checkpoint_roots():
        if not root.exists():
            continue
        for path in sorted(root.glob("*.pt")):
            try:
                resolved = path.resolve()
                stat = resolved.stat()
            except OSError:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            checkpoints.append(
                {
                    "path": key,
                    "name": resolved.name,
                    "root": str(root),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "label": f"{resolved.name} - {stat.st_size / (1024 * 1024):.1f} MB",
                }
            )
    checkpoints.sort(key=lambda item: (item["name"] != "best.pt", item["name"] != "last.pt", -item["modified"]))
    return checkpoints


def _resolve_checkpoint(path: str | None) -> Path:
    checkpoints = _find_checkpoints()
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No .pt checkpoint found.")
    if not path:
        return Path(checkpoints[0]["path"])
    requested = Path(path).expanduser()
    try:
        requested = requested.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid checkpoint path: {path}") from exc
    allowed = {Path(item["path"]).resolve() for item in checkpoints}
    if requested not in allowed:
        raise HTTPException(status_code=400, detail="Checkpoint is outside allowed checkpoint directories.")
    return requested


def _parse_training_log() -> dict[str, Any]:
    candidates = [
        ROOT.parent / "export_checkpoint" / "train_watch.log",
        ROOT / "export_checkpoint" / "train_watch.log",
        ROOT.parent / "logs" / "train_watch.log",
        ROOT / "logs" / "train_watch.log",
    ]
    env_log = os.environ.get("TINYMSS_TRAIN_LOG")
    if env_log:
        candidates.insert(0, Path(env_log))
    log_path = next((path for path in candidates if path.exists()), None)
    if log_path is None:
        return {"available": False, "epochs": []}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    pattern = re.compile(
        r"epoch=(\d+) train_loss=([0-9.]+) valid_loss=([0-9.]+) valid_si_sdr=([-0-9.]+)dB"
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "valid_loss": float(match.group(3)),
                "valid_si_sdr": float(match.group(4)),
            }
        )
    best_loss = min(rows, key=lambda row: row["valid_loss"]) if rows else None
    best_sdr = max(rows, key=lambda row: row["valid_si_sdr"]) if rows else None
    return {
        "available": True,
        "path": str(log_path),
        "epochs": rows[-24:],
        "last": rows[-1] if rows else None,
        "best_valid_loss": best_loss,
        "best_valid_si_sdr": best_sdr,
    }


def _require_torch():
    try:
        import torch
        import torch.nn.functional as torch_f

        from tinymss.model import TinyHybridMSS
    except Exception as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError(f"PyTorch/model runtime is not available: {exc}") from exc
    return torch, torch_f, TinyHybridMSS


def _load_audio(path: Path, sample_rate: int, torch: Any, torch_f: Any):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = torch.from_numpy(np.asarray(audio).T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if int(sr) != sample_rate:
        new_frames = max(1, int(round(wav.shape[-1] * sample_rate / int(sr))))
        flat = wav.reshape(-1, 1, wav.shape[-1])
        wav = torch_f.interpolate(flat, size=new_frames, mode="linear", align_corners=False)
        wav = wav.reshape(2, -1)
    return wav


def _save_audio(path: Path, wav: Any, sample_rate: int, normalize: bool) -> None:
    wav = wav.detach().cpu().float()
    if normalize:
        peak = float(wav.abs().amax().clamp_min(1e-8))
        if peak > 0.98:
            wav = wav * (0.98 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.T.numpy(), sample_rate)


def _load_model(checkpoint_path: Path, device: Any):
    torch, _, TinyHybridMSS = _require_torch()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    cfg = checkpoint["config"]
    model = TinyHybridMSS(**cfg["model"]).to(device)
    state = dict(checkpoint["model"])
    ema_state = checkpoint.get("ema")
    if ema_state and ema_state.get("shadow"):
        state.update(ema_state["shadow"])
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _separate_job(job_id: str, input_path: Path, checkpoint_path: Path, out_dir: Path) -> None:
    try:
        torch, torch_f, _ = _require_torch()
        _set_job(job_id, status="loading", message="Loading checkpoint", progress=0.02)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg = _load_model(checkpoint_path, device)
        sample_rate = int(cfg["model"]["sample_rate"])
        source_names = list(cfg["model"]["sources"])

        job = JOBS[job_id]
        mixture = _load_audio(input_path, sample_rate, torch, torch_f)
        segment_frames = int(round(job.segment_seconds * sample_rate))
        hop = max(1, int(round(segment_frames * (1.0 - job.overlap))))
        total_frames = int(mixture.shape[-1])
        starts = list(range(0, total_frames, hop))
        window = torch.hann_window(segment_frames, periodic=False).clamp_min(1e-4)
        estimates = torch.zeros(len(source_names), 2, total_frames + segment_frames)
        weight = torch.zeros(1, 1, total_frames + segment_frames)

        _set_job(job_id, status="separating", message="Separating stems", progress=0.05)
        with torch.no_grad():
            for idx, start in enumerate(starts, start=1):
                chunk = mixture[:, start : start + segment_frames]
                if chunk.shape[-1] < segment_frames:
                    chunk = torch_f.pad(chunk, (0, segment_frames - chunk.shape[-1]))
                with torch.autocast(device_type="cuda", enabled=job.amp and device.type == "cuda"):
                    estimate = model(chunk.unsqueeze(0).to(device))[0].float().cpu()
                end = start + segment_frames
                estimates[..., start:end] += estimate * window.view(1, 1, -1)
                weight[..., start:end] += window.view(1, 1, -1)
                progress = 0.05 + 0.85 * (idx / max(1, len(starts)))
                _set_job(
                    job_id,
                    progress=round(progress, 4),
                    message=f"Chunk {idx}/{len(starts)}",
                )

        estimates = estimates[..., :total_frames] / weight[..., :total_frames].clamp_min(1e-6)
        outputs: dict[str, str] = {}
        _set_job(job_id, status="writing", message="Writing WAV files", progress=0.93)
        for idx, source in enumerate(source_names):
            stem_path = out_dir / f"{source}.wav"
            _save_audio(stem_path, estimates[idx], sample_rate, job.normalize)
            outputs[source] = str(stem_path)

        zip_path = out_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for source, stem_path in outputs.items():
                zf.write(stem_path, f"{source}.wav")
        outputs["zip"] = str(zip_path)

        _set_job(
            job_id,
            status="done",
            message="Done",
            progress=1.0,
            outputs=outputs,
            finished_at=time.time(),
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            message="Failed",
            error=str(exc),
            finished_at=time.time(),
        )


app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    torch_available = importlib.util.find_spec("torch") is not None
    cuda = False
    device = "cpu"
    if torch_available:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            device = torch.cuda.get_device_name(0) if cuda else "cpu"
        except Exception:
            cuda = False
            device = "torch import failed"
    return {
        "title": APP_TITLE,
        "root": str(ROOT),
        "work_dir": str(DEFAULT_WORK_DIR),
        "torch_available": torch_available,
        "cuda": cuda,
        "device": device,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/checkpoints")
def checkpoints() -> dict[str, Any]:
    return {"checkpoints": _find_checkpoints()}


@app.get("/api/training-summary")
def training_summary() -> dict[str, Any]:
    return _parse_training_log()


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    checkpoint: str | None = Form(default=None),
    segment_seconds: float = Form(default=8.0),
    overlap: float = Form(default=0.5),
    normalize: bool = Form(default=False),
    amp: bool = Form(default=True),
) -> JSONResponse:
    if segment_seconds < 1.0 or segment_seconds > 30.0:
        raise HTTPException(status_code=400, detail="segment_seconds must be between 1 and 30.")
    if overlap < 0.0 or overlap >= 0.9:
        raise HTTPException(status_code=400, detail="overlap must be in [0, 0.9).")

    checkpoint_path = _resolve_checkpoint(checkpoint)
    job_id = uuid.uuid4().hex[:12]
    job_dir = DEFAULT_WORK_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    input_path = input_dir / f"input{suffix}"

    bytes_written = 0
    with input_path.open("wb") as handle:
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_MB} MB.")
            handle.write(chunk)

    job = JobState(
        id=job_id,
        status="queued",
        created_at=time.time(),
        checkpoint=str(checkpoint_path),
        input_name=audio.filename or input_path.name,
        segment_seconds=segment_seconds,
        overlap=overlap,
        normalize=normalize,
        amp=amp,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    background_tasks.add_task(EXECUTOR.submit, _separate_job, job_id, input_path, checkpoint_path, output_dir)
    return JSONResponse(_json_job(job))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _json_job(job)


@app.get("/api/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        path = job.outputs.get(name)
    if not path:
        raise HTTPException(status_code=404, detail="Output not found.")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing.")
    media_type = "application/zip" if file_path.suffix == ".zip" else "audio/wav"
    return FileResponse(file_path, filename=file_path.name, media_type=media_type)
