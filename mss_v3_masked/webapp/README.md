# Tiny Mask MSS Web App

Local web console for running trained Tiny Mask MSS checkpoints.

## Install

```bash
pip install -r requirements-web.txt
```

The web dependencies do not pin PyTorch because the correct build depends on
your GPU/CUDA environment. Kaggle already provides PyTorch. For local GPU
inference, install the matching PyTorch build from the official PyTorch command
selector before running a separation job.

## Run

```bash
python scripts/serve_web.py --host 0.0.0.0 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## Checkpoint Discovery

The app searches these locations:

- `mss/runs/tiny-mask-v3/checkpoints`
- `mss/runs/tiny-hybrid/checkpoints`
- `mss/export_checkpoint`
- `mss/export_checkpoint-v3`
- `../export_checkpoint`
- `../export_checkpoint-v3`
- `../runs/tiny-mask-v3/checkpoints`
- `../runs/tiny-hybrid/checkpoints`

To override:

```bash
set TINYMSS_CHECKPOINT_DIRS=C:\path\to\checkpoints
```

On Linux/macOS use `:` between directories. On Windows use `;`.

## Output

Separated stems are written under:

```text
mss/web_runs/<job-id>/output
```

Use `best.pt` for first listening tests. Keep `last.pt` for resuming training.
