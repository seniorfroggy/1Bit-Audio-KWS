"""Evaluate the 1-bit ResNet18 on SpeechCommands: accuracy + per-sample latency.

Compares:
  * baseline: model.BinaryResNet18 with +/-1 float weights reconstructed from
    the packed checkpoint.
  * fast:     inference.BinaryResNet18Inference (NEON XNOR/popcount kernel).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from inference import BinaryResNet18Inference, build_float_reference

SAMPLE_RATE = 16000
DATA_DIR = "./data"
DEFAULT_CKPT = "resnet18_1bit.pt"


SPEECH_COMMANDS_LABELS: List[str] = [
    "wow", "go", "learn", "one", "cat", "forward", "zero", "stop", "happy",
    "four", "follow", "down", "dog", "nine", "yes", "left", "marvin", "bed",
    "tree", "off", "backward", "on", "right", "house", "six", "three", "two",
    "up", "eight", "bird", "sheila", "five", "visual", "seven", "no",
]


def load_label_index(repo_root: Path) -> dict:
    pkl = repo_root / "labels.pickle"
    if pkl.exists():
        with open(pkl, "rb") as f:
            raw = pickle.load(f)
        uniq = sorted(set(raw)) if isinstance(raw, (list, tuple)) else list(raw.keys())
        return {label: idx for idx, label in enumerate(uniq)}
    return {label: idx for idx, label in enumerate(SPEECH_COMMANDS_LABELS)}


def build_transform() -> nn.Module:
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64),
        torchaudio.transforms.AmplitudeToDB(),
    )


class _SoundfileSpeechCommands(torch.utils.data.Dataset):
    def __init__(self, inner):
        self._inner = inner

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        path = self._inner._walker[idx]
        label = os.path.basename(os.path.dirname(path))
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        wav = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
        return wav, sr, label, "", 0


class SpeechCommandsCollator:
    def __init__(self, label_to_index: dict):
        self.label_to_index = label_to_index

    def __call__(self, batch):
        tensors, targets = [], []
        for waveform, _sr, label, _spk, _utt in batch:
            if waveform.shape[1] < SAMPLE_RATE:
                pad = torch.zeros(1, SAMPLE_RATE - waveform.shape[1])
                waveform = torch.cat([waveform, pad], dim=1)
            elif waveform.shape[1] > SAMPLE_RATE:
                waveform = waveform[:, :SAMPLE_RATE]
            tensors.append(waveform)
            targets.append(self.label_to_index[label])
        return torch.stack(tensors), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    transform: nn.Module,
    limit: int | None,
    name: str,
) -> Tuple[float, float, int]:
    """Returns (accuracy, mean_ms_per_sample, n_samples)."""
    model.eval()
    total = 0
    correct = 0
    total_s = 0.0

    warm = next(iter(loader))
    waveforms, labels = warm
    specs = transform(waveforms)
    _ = model(specs)

    for waveforms, labels in loader:
        specs = transform(waveforms)
        t0 = time.perf_counter()
        logits = model(specs)
        t1 = time.perf_counter()
        total_s += t1 - t0
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.size(0)
        if limit is not None and total >= limit:
            break

    acc = 100.0 * correct / max(total, 1)
    ms = 1000.0 * total_s / max(total, 1)
    print(f"  [{name}] acc={acc:.2f}%  n={total}  mean_latency={ms:.2f} ms/sample")
    return acc, ms, total


@torch.no_grad()
def sanity_logit_diff(fast: nn.Module, ref: nn.Module, loader: DataLoader, transform):
    waveforms, _ = next(iter(loader))
    specs = transform(waveforms)
    yf = fast(specs)
    yr = ref(specs)
    diff = (yf - yr).abs().max().item()
    print(f"  max |fast - baseline| logit diff on first batch: {diff:.3e}")
    return diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after approximately this many samples")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch.set_num_threads value for fair comparison")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download", action="store_true",
                        help="allow torchaudio to download SpeechCommands if missing")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="only run the fast model (useful once parity is verified)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print(f"torch={torch.__version__}  threads={torch.get_num_threads()}  ckpt={args.ckpt}")

    repo_root = Path(__file__).resolve().parent
    label_to_index = load_label_index(repo_root)
    num_classes = len(label_to_index)
    print(f"num_classes={num_classes}")

    print("Loading SpeechCommands v0.02 testing split ...")
    inner_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        args.data_dir, url="speech_commands_v0.02",
        subset="testing", download=args.download,
    )
    test_dataset = _SoundfileSpeechCommands(inner_dataset)
    print(f"  test dataset size = {len(test_dataset)}")

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=SpeechCommandsCollator(label_to_index),
        pin_memory=False,
    )

    transform = build_transform()

    print("Building fast model ...")
    fast_model = BinaryResNet18Inference.from_packed_checkpoint(
        args.ckpt, num_classes=num_classes
    )

    if not args.skip_baseline:
        print("Building baseline (float +/-1) model ...")
        baseline_model = build_float_reference(args.ckpt, num_classes=num_classes)
        print("Sanity check: logit parity between baseline and fast ...")
        sanity_logit_diff(fast_model, baseline_model, loader, transform)

        print("Running BASELINE (float +/-1, F.conv2d):")
        base_acc, base_ms, n_base = run_eval(
            baseline_model, loader, transform, args.limit, "baseline"
        )
    else:
        base_acc = base_ms = None

    print("Running FAST:")
    fast_acc, fast_ms, n_fast = run_eval(
        fast_model, loader, transform, args.limit, "fast"
    )

    print("\n=== SUMMARY ===")
    if base_acc is not None:
        print(f"  baseline: acc={base_acc:.2f}%   latency={base_ms:.2f} ms/sample")
        print(f"  fast    : acc={fast_acc:.2f}%   latency={fast_ms:.2f} ms/sample")
        print(f"  accuracy delta: {fast_acc - base_acc:+.2f} pp")
        print(f"  speedup       : {base_ms / fast_ms:.2f}x")
    else:
        print(f"  fast    : acc={fast_acc:.2f}%   latency={fast_ms:.2f} ms/sample")


if __name__ == "__main__":
    main()
