const state = {
  status: null,
  checkpoints: [],
  job: null,
  pollTimer: null,
};

const els = {
  runtimeLine: document.getElementById("runtimeLine"),
  devicePill: document.getElementById("devicePill"),
  checkpointPill: document.getElementById("checkpointPill"),
  checkpointSelect: document.getElementById("checkpointSelect"),
  refreshButton: document.getElementById("refreshButton"),
  audioInput: document.getElementById("audioInput"),
  dropZone: document.getElementById("dropZone"),
  fileName: document.getElementById("fileName"),
  fileMeta: document.getElementById("fileMeta"),
  waveCanvas: document.getElementById("waveCanvas"),
  segmentInput: document.getElementById("segmentInput"),
  overlapInput: document.getElementById("overlapInput"),
  overlapValue: document.getElementById("overlapValue"),
  normalizeInput: document.getElementById("normalizeInput"),
  ampInput: document.getElementById("ampInput"),
  startButton: document.getElementById("startButton"),
  jobId: document.getElementById("jobId"),
  progressBar: document.getElementById("progressBar"),
  jobStatus: document.getElementById("jobStatus"),
  elapsed: document.getElementById("elapsed"),
  jobMessage: document.getElementById("jobMessage"),
  logBox: document.getElementById("logBox"),
  stemsGrid: document.getElementById("stemsGrid"),
  zipLink: document.getElementById("zipLink"),
  trainSummaryState: document.getElementById("trainSummaryState"),
  lastEpoch: document.getElementById("lastEpoch"),
  bestLoss: document.getElementById("bestLoss"),
  bestSdr: document.getElementById("bestSdr"),
  lossCanvas: document.getElementById("lossCanvas"),
};

function log(line) {
  const now = new Date().toLocaleTimeString();
  els.logBox.textContent += `[${now}] ${line}\n`;
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function fmtSeconds(value) {
  if (!Number.isFinite(value)) return "0.0s";
  if (value < 60) return `${value.toFixed(1)}s`;
  const min = Math.floor(value / 60);
  const sec = Math.round(value % 60);
  return `${min}m ${sec}s`;
}

function drawEmptyWave() {
  const canvas = els.waveCanvas;
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.height;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#d8dfdb";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

async function drawWave(file) {
  drawEmptyWave();
  if (!file) return;
  try {
    const buffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(buffer.slice(0));
    const channel = audioBuffer.getChannelData(0);
    const canvas = els.waveCanvas;
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 600;
    const height = canvas.height;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfcfb";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#12766b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(channel.length / width));
    for (let x = 0; x < width; x += 1) {
      let min = 1;
      let max = -1;
      const start = x * step;
      for (let i = 0; i < step && start + i < channel.length; i += 1) {
        const v = channel[start + i];
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
      ctx.moveTo(x, (1 + min) * height * 0.5);
      ctx.lineTo(x, (1 + max) * height * 0.5);
    }
    ctx.stroke();
    audioContext.close();
  } catch (err) {
    log(`Waveform preview skipped: ${err.message}`);
  }
}

function drawLoss(summary) {
  const canvas = els.lossCanvas;
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.height;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfb";
  ctx.fillRect(0, 0, width, height);
  const rows = (summary.epochs || []).filter((row) => Number.isFinite(row.valid_loss));
  if (!rows.length) {
    ctx.strokeStyle = "#d8dfdb";
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    return;
  }
  const values = rows.map((row) => row.valid_loss);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(1e-6, max - min);
  ctx.strokeStyle = "#273f7a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const x = rows.length === 1 ? width / 2 : (index / (rows.length - 1)) * width;
    const y = height - 14 - ((row.valid_loss - min) / spread) * (height - 28);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function refreshStatus() {
  const [statusRes, ckptRes, trainRes] = await Promise.all([
    fetch("/api/status"),
    fetch("/api/checkpoints"),
    fetch("/api/training-summary"),
  ]);
  state.status = await statusRes.json();
  state.checkpoints = (await ckptRes.json()).checkpoints || [];
  const summary = await trainRes.json();

  els.runtimeLine.textContent = state.status.root;
  els.devicePill.textContent = state.status.cuda ? state.status.device : "CPU";
  els.checkpointPill.textContent = `${state.checkpoints.length} checkpoint(s)`;

  els.checkpointSelect.innerHTML = "";
  for (const ckpt of state.checkpoints) {
    const option = document.createElement("option");
    option.value = ckpt.path;
    option.textContent = ckpt.label;
    els.checkpointSelect.appendChild(option);
  }
  if (!state.checkpoints.length) {
    const option = document.createElement("option");
    option.textContent = "No checkpoint found";
    els.checkpointSelect.appendChild(option);
  }

  if (summary.available) {
    els.trainSummaryState.textContent = "Log loaded";
    els.lastEpoch.textContent = summary.last ? summary.last.epoch : "-";
    els.bestLoss.textContent = summary.best_valid_loss ? summary.best_valid_loss.valid_loss.toFixed(4) : "-";
    els.bestSdr.textContent = summary.best_valid_si_sdr ? `${summary.best_valid_si_sdr.valid_si_sdr.toFixed(2)} dB` : "-";
    drawLoss(summary);
  } else {
    els.trainSummaryState.textContent = "No log";
    drawLoss({ epochs: [] });
  }
}

function renderJob(job) {
  state.job = job;
  els.jobId.textContent = job.id || "Idle";
  els.progressBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  els.jobStatus.textContent = job.status || "Idle";
  els.elapsed.textContent = fmtSeconds(job.elapsed_seconds || 0);
  els.jobMessage.textContent = job.message || "";
  if (job.error) log(`Error: ${job.error}`);
  if (job.status === "done") {
    renderOutputs(job);
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    els.startButton.disabled = false;
  }
  if (job.status === "error") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    els.startButton.disabled = false;
  }
}

function renderOutputs(job) {
  const stems = ["vocals", "drums", "bass", "other"];
  els.stemsGrid.innerHTML = "";
  for (const stem of stems) {
    if (!job.outputs || !job.outputs[stem]) continue;
    const item = document.createElement("article");
    item.className = "stem";
    const url = `/api/jobs/${job.id}/download/${stem}`;
    item.innerHTML = `
      <h3>${stem}</h3>
      <audio controls src="${url}"></audio>
      <a href="${url}">Download ${stem}.wav</a>
    `;
    els.stemsGrid.appendChild(item);
  }
  if (job.outputs && job.outputs.zip) {
    els.zipLink.href = `/api/jobs/${job.id}/download/zip`;
    els.zipLink.classList.remove("disabled");
  }
}

async function pollJob(id) {
  const res = await fetch(`/api/jobs/${id}`);
  const job = await res.json();
  renderJob(job);
}

async function startJob() {
  const file = els.audioInput.files[0];
  if (!file) {
    log("Select an audio file first.");
    return;
  }
  if (!state.checkpoints.length) {
    log("No checkpoint found.");
    return;
  }
  els.startButton.disabled = true;
  els.stemsGrid.innerHTML = "";
  els.zipLink.classList.add("disabled");
  els.progressBar.style.width = "0%";
  const form = new FormData();
  form.append("audio", file);
  form.append("checkpoint", els.checkpointSelect.value);
  form.append("segment_seconds", els.segmentInput.value);
  form.append("overlap", els.overlapInput.value);
  form.append("normalize", els.normalizeInput.checked ? "true" : "false");
  form.append("amp", els.ampInput.checked ? "true" : "false");
  log(`Queued ${file.name}`);
  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    log(`Request failed: ${err.detail || res.statusText}`);
    els.startButton.disabled = false;
    return;
  }
  const job = await res.json();
  renderJob(job);
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(() => pollJob(job.id), 1500);
}

els.refreshButton.addEventListener("click", () => {
  refreshStatus().then(() => log("Refreshed")).catch((err) => log(err.message));
});

els.audioInput.addEventListener("change", () => {
  const file = els.audioInput.files[0];
  if (!file) return;
  els.fileName.textContent = file.name;
  els.fileMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  drawWave(file);
});

els.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.dropZone.classList.add("active");
});

els.dropZone.addEventListener("dragleave", () => {
  els.dropZone.classList.remove("active");
});

els.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.dropZone.classList.remove("active");
  if (event.dataTransfer.files.length) {
    els.audioInput.files = event.dataTransfer.files;
    els.audioInput.dispatchEvent(new Event("change"));
  }
});

els.overlapInput.addEventListener("input", () => {
  els.overlapValue.textContent = `${Math.round(Number(els.overlapInput.value) * 100)}%`;
});

els.startButton.addEventListener("click", () => {
  startJob().catch((err) => {
    log(err.message);
    els.startButton.disabled = false;
  });
});

window.addEventListener("resize", () => drawWave(els.audioInput.files[0]));

drawEmptyWave();
refreshStatus().catch((err) => log(err.message));

