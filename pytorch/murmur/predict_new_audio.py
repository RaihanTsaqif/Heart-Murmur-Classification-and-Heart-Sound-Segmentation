"""
Run NEW audio files through the CirCor-trained murmur CRNN.

Preprocessing is identical to training:
    read_audio(target_fs=2000, bandpass 25-400)  -> log-mel 128 (@15/25 ms)
    -> z-score (SNV) -> +delta +delta2 (384 ch) -> 5 s windows (333 frames)

Per file we slide 5 s windows (50% overlap) over the whole recording and average
the softmax across windows -> one probability per class (more robust than a single
center crop for arbitrary-length files). Files shorter than 5 s are wrap-padded.

Usage (run from the repo root so `model`/`utils` import correctly):
    python predict_new_audio.py path\to\a.wav path\to\b.wav
    python predict_new_audio.py path\to\folder_of_wavs
    python predict_new_audio.py --window center a.wav   # single center crop instead
"""
import argparse
import glob
import json
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import model.model as module_arch
from utils.util import read_audio
from utils.audio_feature_extractor import LogMelExtractor, standard_normal_variate

# --- fixed preprocessing constants (must match training) ------------------- #
SR = 2000                                   # resample target == training rate
DURATION = 5                                # seconds
HOP_MS = 15
CYCLE_LEN = int(DURATION * 1000 / HOP_MS)   # 333 frames per 5 s window

# --- model: (label, arch-config, weights, class names idx0,idx1) ------------ #
MODELS = [
    {
        "label": "CirCor murmur CRNN (Present/Absent)",
        "config": "config/config_crnn.json",  # this run has no config.json of its own
        "weights": "saved/training with circor murmur using authors architecture and oversampling/best_model.pth",
        "classes": ["Present", "Absent"],     # index 0 = positive (murmur present)
    },
]


def load_model(cfg_path, weights_path, device):
    arch = json.load(open(cfg_path))["arch"]
    net = getattr(module_arch, arch["type"])(**arch["args"])
    ckpt = torch.load(weights_path, weights_only=False, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    return net.to(device).eval()


def feature_for_file(path):
    """resample 4k/anything -> 2k, bandpass, log-mel, z-score, +delta+delta2.
    Returns (384, T) float32 -- identical to the dataset's online preprocessing
    *before* cropping."""
    audio, fs = read_audio(path, target_fs=SR, filter=True)
    feat = LogMelExtractor(audio, fs, log=True, snv=False)   # (128, T)
    feat = standard_normal_variate(feat)
    d = librosa.feature.delta(feat)
    d2 = librosa.feature.delta(d)
    feat = np.concatenate((feat, d, d2), axis=0)             # (384, T)
    return feat.astype(np.float32)


def windows(feat, mode="slide"):
    """Yield (384, 333) crops. 'slide' = 50%-overlap windows over the whole
    file; 'center' = single center crop (matches the original eval scheme)."""
    _, T = feat.shape
    if T < CYCLE_LEN:                                        # too short -> wrap-pad
        yield np.pad(feat, ((0, 0), (0, CYCLE_LEN - T)), mode="wrap")
        return
    if mode == "center":
        s = (T - CYCLE_LEN) // 2
        yield feat[:, s:s + CYCLE_LEN]
        return
    hop = CYCLE_LEN // 2
    starts = list(range(0, T - CYCLE_LEN + 1, hop))
    if starts[-1] != T - CYCLE_LEN:
        starts.append(T - CYCLE_LEN)                         # ensure the tail is covered
    for s in starts:
        yield feat[:, s:s + CYCLE_LEN]


@torch.no_grad()
def predict(net, feat, device, mode):
    """Mean softmax over all windows -> (p_class0, p_class1)."""
    x = torch.from_numpy(np.stack(list(windows(feat, mode)))).to(device).float()
    probs = F.softmax(net(x), dim=1).mean(0).cpu().numpy()
    return probs


def collect_paths(args_paths):
    paths = []
    for p in args_paths:
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, "*.wav")))
        else:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="wav file(s) or folder(s)")
    ap.add_argument("--window", choices=["slide", "center"], default="slide",
                    help="'slide' = avg over 50%%-overlap 5 s windows (default); "
                         "'center' = single center 5 s crop")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   |   window mode: {args.window}\n")

    nets = [(m, load_model(m["config"], m["weights"], device)) for m in MODELS]

    paths = collect_paths(args.paths)
    if not paths:
        print("No .wav files found."); return

    for path in paths:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: FILE NOT FOUND"); continue
        feat = feature_for_file(path)
        secs = feat.shape[1] * HOP_MS / 1000
        print(f"=== {os.path.basename(path)}  (~{secs:.1f}s) ===")
        for m, net in nets:
            p = predict(net, feat, device, args.window)
            idx = int(np.argmax(p))
            print(f"  {m['label']:38s} -> {m['classes'][idx]:9s} "
                  f"(P[{m['classes'][0]}]={p[0]:.3f}, P[{m['classes'][1]}]={p[1]:.3f})")
        print()


if __name__ == "__main__":
    main()
