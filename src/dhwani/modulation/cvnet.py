"""Engine B: CVNet-RF (huggingface.co/sohelimi/cvnet-rf) adapter.

Verified facts about the model (README + inference.py, checked 2026-09):
* Trained on RadioML 2018.01A: 24 classes, frames of 1024 raw IQ samples.
* Input (1, 2, 1024), z-scored per channel.
* Checkpoint dict key "model_state"; variants complex_best.pt (50.3% val
  acc, 379k params) and real_best.pt (54.4% val acc, 934k params).
* Trained 100 epochs (Adam, label smoothing 0.1, phase-rotation + AWGN
  augmentation) on an 80/10/10 split of 2.56M frames.
* Accuracy is ~50-54% *averaged over -20..+30 dB SNR* - honest but modest.
  The fusion stage therefore treats this engine as advisory, never
  authoritative, and out-of-distribution inputs are flagged.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK"
]

# The download is pinned: a moved branch or a swapped file raises instead
# of loading. Values from huggingface.co/api/models/sohelimi/cvnet-rf.
REPO = "sohelimi/cvnet-rf"
REVISION = "df32f6cd9bb033835465928307610bba1c376708"
SHA256 = {
    "real": "8a6861dec969d01bcf0198ef2e0fc52c0938c6dacf613ca8b1078eba449ff53c",
    "complex": "30447995f7fc2931fc24cea415a78301bab1353bdb33dd017a73279a1c582aa1",
}

_model_cache = {}


def check_sha256(path: str, expected: str):
    with open(path, "rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != expected:
        raise ValueError(f"CVNet checkpoint {path}: sha256 {got} does not "
                         f"match the pinned {expected}")


def _build_model(variant: str, torch):
    import torch.nn.functional as F

    class ComplexConv1d(torch.nn.Module):
        def __init__(self, in_ch, out_ch, k, padding=0):
            super().__init__()
            self.real_conv = torch.nn.Conv1d(in_ch, out_ch, k, 1, padding)
            self.imag_conv = torch.nn.Conv1d(in_ch, out_ch, k, 1, padding)

        def forward(self, xr, xi):
            return (self.real_conv(xr) - self.imag_conv(xi),
                    self.real_conv(xi) + self.imag_conv(xr))

    class ComplexBatchNorm1d(torch.nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.bn_real = torch.nn.BatchNorm1d(nf)
            self.bn_imag = torch.nn.BatchNorm1d(nf)

        def forward(self, xr, xi):
            return self.bn_real(xr.contiguous()), self.bn_imag(xi.contiguous())

    class ComplexMaxPool1d(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.pool = torch.nn.MaxPool1d(k)

        def forward(self, xr, xi):
            return self.pool(xr), self.pool(xi)

    class ComplexCNN(torch.nn.Module):
        def __init__(self, num_classes=24, dropout=0.5):
            super().__init__()
            self.conv1 = ComplexConv1d(1, 64, 7, padding=3)
            self.bn1 = ComplexBatchNorm1d(64)
            self.pool1 = ComplexMaxPool1d(2)
            self.conv2 = ComplexConv1d(64, 128, 5, padding=2)
            self.bn2 = ComplexBatchNorm1d(128)
            self.pool2 = ComplexMaxPool1d(2)
            self.conv3 = ComplexConv1d(128, 256, 3, padding=1)
            self.bn3 = ComplexBatchNorm1d(256)
            self.pool3 = ComplexMaxPool1d(2)
            self.dropout = torch.nn.Dropout(dropout)
            self.fc1 = torch.nn.Linear(256 * 128 * 2, 512)
            self.fc2 = torch.nn.Linear(512, num_classes)

        def forward(self, x):
            xr, xi = x[:, 0:1, :], x[:, 1:2, :]
            for conv, bn, pool in ((self.conv1, self.bn1, self.pool1),
                                   (self.conv2, self.bn2, self.pool2),
                                   (self.conv3, self.bn3, self.pool3)):
                xr, xi = conv(xr, xi)
                xr, xi = bn(xr, xi)
                xr, xi = F.relu(xr), F.relu(xi)
                xr, xi = pool(xr, xi)
            x = torch.cat([xr, xi], dim=1).flatten(1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)

    class RealCNN(torch.nn.Module):
        def __init__(self, num_classes=24, dropout=0.5):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(2, 64, 7, padding=3), torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(), torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(64, 128, 5, padding=2), torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(), torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(128, 256, 3, padding=1), torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(), torch.nn.MaxPool1d(2),
                torch.nn.Flatten(), torch.nn.Dropout(dropout),
                torch.nn.Linear(256 * 128, 512), torch.nn.ReLU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(512, num_classes))

        def forward(self, x):
            return self.net(x)

    return ComplexCNN() if variant == "complex" else RealCNN()


def _resolve_device(pref: str, torch):
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cvnet(checkpoint_path: str = "", variant: str = "real",
               device: str = "auto"):
    """Load CVNet-RF; downloads the pinned checkpoint when no local path is
    given. Only tensors are unpickled (weights_only=True): a checkpoint
    carrying pickled code raises. Returns (model, device) or raises."""
    key = (checkpoint_path, variant, device)
    if key in _model_cache:
        return _model_cache[key]
    import torch
    if not checkpoint_path:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=REPO, revision=REVISION,
                                          filename=f"{variant}_best.pt")
        check_sha256(checkpoint_path, SHA256[variant])
    dev = _resolve_device(device, torch)
    model = _build_model(variant, torch)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(dev).eval()
    _model_cache[key] = (model, dev)
    return model, dev


def classify_cvnet(x: np.ndarray, checkpoint_path: str = "",
                   variant: str = "real", device: str = "auto",
                   frame_size: int = 1024, max_frames: int = 32) -> Optional[dict]:
    """Run CVNet-RF over frames of the (channelised) signal; averages the
    softmax over frames. Returns None if the model is unavailable."""
    try:
        import torch
        import torch.nn.functional as F
        model, dev = load_cvnet(checkpoint_path, variant, device=device)
    except Exception as e:
        return {"error": f"CVNet-RF unavailable: {e}"}

    x = np.asarray(x, dtype=np.complex64)
    n_frames = min(max_frames, len(x) // frame_size)
    if n_frames == 0:
        return {"error": "signal shorter than one CVNet frame (1024 samples)"}
    frames = np.stack([
        np.stack([x[i * frame_size:(i + 1) * frame_size].real,
                  x[i * frame_size:(i + 1) * frame_size].imag])
        for i in range(n_frames)])
    # per-channel z-score, as in the model's own inference.py
    frames = (frames - frames.mean(axis=2, keepdims=True)) / \
        (frames.std(axis=2, keepdims=True) + 1e-8)
    with torch.no_grad():
        t = torch.tensor(frames, dtype=torch.float32, device=dev)
        logits = model(t)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    ranked = {CLASSES[i]: float(probs[i]) for i in np.argsort(probs)[::-1]}
    return {"probabilities": ranked, "n_frames": n_frames,
            "variant": variant,
            "note": "RadioML-2018 distribution; ~54% val acc averaged over "
                    "-20..30 dB SNR - advisory only"}
