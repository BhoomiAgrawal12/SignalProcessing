"""
CVNet-RF Inference
==================
Classify the modulation scheme of a raw I/Q signal using CVNet-RF.

Usage:
    python inference.py --model complex --iq path/to/iq.npy
    python inference.py --model real    --iq path/to/iq.npy

Input:
    A numpy array of shape (2, 1024) — row 0 is I, row 1 is Q.
    If your array is (1024, 2), it will be transposed automatically.

Output:
    Top-5 modulation predictions with confidence scores.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

# ---------------------------------------------------------------------------
# Modulation class labels (RadioML 2018.01A order)
# ---------------------------------------------------------------------------
CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK"
]

# ---------------------------------------------------------------------------
# Model definitions (must match training architecture exactly)
# ---------------------------------------------------------------------------

class ComplexConv1d(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.imag_conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x_real, x_imag):
        return (self.real_conv(x_real) - self.imag_conv(x_imag),
                self.real_conv(x_imag) + self.imag_conv(x_real))


class ComplexBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.bn_real = torch.nn.BatchNorm1d(num_features, eps=eps)
        self.bn_imag = torch.nn.BatchNorm1d(num_features, eps=eps)

    def forward(self, x_real, x_imag):
        return self.bn_real(x_real.contiguous()), self.bn_imag(x_imag.contiguous())


class ComplexMaxPool1d(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.pool = torch.nn.MaxPool1d(kernel_size)

    def forward(self, x_real, x_imag):
        magnitude = (x_real ** 2 + x_imag ** 2).sqrt()
        idx = self.pool(magnitude)
        return self.pool(x_real), self.pool(x_imag)


class ComplexCNN(torch.nn.Module):
    def __init__(self, num_classes=24, dropout=0.5):
        super().__init__()
        self.conv1 = ComplexConv1d(1, 64, kernel_size=7, padding=3)
        self.bn1   = ComplexBatchNorm1d(64)
        self.pool1 = ComplexMaxPool1d(2)

        self.conv2 = ComplexConv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = ComplexBatchNorm1d(128)
        self.pool2 = ComplexMaxPool1d(2)

        self.conv3 = ComplexConv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = ComplexBatchNorm1d(256)
        self.pool3 = ComplexMaxPool1d(2)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(256 * 128 * 2, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x_real, x_imag = x[:, 0:1, :], x[:, 1:2, :]

        x_real, x_imag = self.conv1(x_real, x_imag)
        x_real, x_imag = self.bn1(x_real, x_imag)
        x_real = F.relu(x_real); x_imag = F.relu(x_imag)
        x_real, x_imag = self.pool1(x_real, x_imag)

        x_real, x_imag = self.conv2(x_real, x_imag)
        x_real, x_imag = self.bn2(x_real, x_imag)
        x_real = F.relu(x_real); x_imag = F.relu(x_imag)
        x_real, x_imag = self.pool2(x_real, x_imag)

        x_real, x_imag = self.conv3(x_real, x_imag)
        x_real, x_imag = self.bn3(x_real, x_imag)
        x_real = F.relu(x_real); x_imag = F.relu(x_imag)
        x_real, x_imag = self.pool3(x_real, x_imag)

        x = torch.cat([x_real, x_imag], dim=1)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class RealCNN(torch.nn.Module):
    def __init__(self, num_classes=24, dropout=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2, 64,  kernel_size=7, padding=3), torch.nn.BatchNorm1d(64),  torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=5, padding=2), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256 * 128, 512), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def preprocess(iq: np.ndarray) -> torch.Tensor:
    """Normalize and reshape I/Q array to (1, 2, 1024) tensor."""
    if iq.shape == (1024, 2):
        iq = iq.T                          # → (2, 1024)
    assert iq.shape == (2, 1024), f"Expected shape (2,1024) or (1024,2), got {iq.shape}"
    # Zero mean, unit variance per channel
    iq = (iq - iq.mean(axis=1, keepdims=True)) / (iq.std(axis=1, keepdims=True) + 1e-8)
    return torch.tensor(iq, dtype=torch.float32).unsqueeze(0)   # (1, 2, 1024)


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    if model_type == "complex":
        model = ComplexCNN(num_classes=24, dropout=0.5)
    else:
        model = RealCNN(num_classes=24, dropout=0.5)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def classify(model, tensor: torch.Tensor, device: torch.device, top_k: int = 5):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    top_idx  = probs.argsort()[::-1][:top_k]
    results  = [(CLASSES[i], float(probs[i])) for i in top_idx]
    return results


def main():
    parser = argparse.ArgumentParser(description="CVNet-RF Inference")
    parser.add_argument("--model",      choices=["complex", "real"], default="complex")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint. Defaults to complex_best.pt or real_best.pt")
    parser.add_argument("--iq",         default=None,
                        help="Path to .npy file of shape (2,1024) or (1024,2)")
    parser.add_argument("--top_k",      type=int, default=5)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint path
    if args.checkpoint is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.checkpoint = os.path.join(script_dir, f"{args.model}_best.pt")

    print(f"Loading {args.model} model from {args.checkpoint} ...")
    model = load_model(args.model, args.checkpoint, device)

    # Load or generate demo I/Q
    if args.iq is None:
        print("No --iq file provided. Running on synthetic QPSK demo signal.")
        t    = np.linspace(0, 1, 1024)
        bits = np.random.randint(0, 4, 8)
        phases = [k * np.pi / 2 for k in bits for _ in range(128)]
        iq = np.array([np.cos(phases), np.sin(phases)], dtype=np.float32)
    else:
        iq = np.load(args.iq).astype(np.float32)

    tensor  = preprocess(iq)
    results = classify(model, tensor, device, top_k=args.top_k)

    print(f"\nTop-{args.top_k} predictions:")
    print(f"{'Rank':<6} {'Modulation':<12} {'Confidence':>10}")
    print("-" * 32)
    for rank, (label, conf) in enumerate(results, 1):
        bar = "█" * int(conf * 30)
        print(f"  {rank:<4} {label:<12} {conf*100:>8.1f}%  {bar}")

    print(f"\nDecision: {results[0][0]}  ({results[0][1]*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
