"""
Run NEW audio files through the CirCor-trained murmur CRNN.

Preprocessing is identical to training:
    read_audio(target_fs=2000, bandpass 25-400)  -> log-mel 128 (@15/25 ms)
    -> z-score (SNV) -> +delta +delta2 (384 ch) -> 5 s windows (333 frames)

Per file we slide 5 s windows (50% overlap) over the whole recording and average
the softmax across windows -> one probability per class (more robust than a single
center crop for arbitrary-length files). Files shorter than 5 s are wrap-padded.

Usage (works from any directory; config/weights are resolved relative to THIS file):
    # model/ + utils/ come from the upstream murmur repo (SiyuLou). Point to a
    # clone of it with --murmur-repo, or drop that clone next to this repo as
    # ../AutomaticHeartSoundClassification, or set $MURMUR_REPO.
    python pytorch/murmur/predict_new_audio.py path\to\a.wav path\to\b.wav
    python pytorch/murmur/predict_new_audio.py path\to\folder_of_wavs
    python pytorch/murmur/predict_new_audio.py --window center a.wav   # single center crop
    python pytorch/murmur/predict_new_audio.py --murmur-repo D:\AutomaticHeartSoundClassification a.wav
"""
import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# Paths are anchored to THIS file, so the script runs from any CWD. ---------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))  # has configs/ models/

# `model` and `utils` live in the upstream murmur repo (not vendored here);
# they're imported in _load_upstream() once we know where that clone is.
module_arch = read_audio = LogMelExtractor = standard_normal_variate = None


def _load_upstream(murmur_repo):
    """Add the upstream murmur repo to sys.path and bind model/utils to globals."""
    global module_arch, read_audio, LogMelExtractor, standard_normal_variate
    murmur_repo = os.path.abspath(murmur_repo)
    if not os.path.isdir(os.path.join(murmur_repo, "model")):
        raise SystemExit(
            f"murmur repo not found at: {murmur_repo}\n"
            "Clone SiyuLou/AutomaticHeartSoundClassification and pass it with "
            "--murmur-repo PATH (or set $MURMUR_REPO).")
    sys.path.insert(0, murmur_repo)
    import model.model as _ma
    from utils.util import read_audio as _ra
    from utils.audio_feature_extractor import (LogMelExtractor as _lme,
                                               standard_normal_variate as _snv)
    module_arch, read_audio = _ma, _ra
    LogMelExtractor, standard_normal_variate = _lme, _snv

# --- fixed preprocessing constants (must match training) ------------------- #
SR = 2000                                   # resample target == training rate
DURATION = 5                                # seconds
HOP_MS = 15
CYCLE_LEN = int(DURATION * 1000 / HOP_MS)   # 333 frames per 5 s window

# --- model: (label, arch-config, weights, class names idx0,idx1) ------------ #
# config + weights ship in THIS repo (configs/ and models/), resolved via _REPO_ROOT.
MODELS = [
    {
        "label": "CirCor murmur CRNN (Present/Absent)",
        "config": os.path.join(_REPO_ROOT, "configs", "config_crnn.json"),
        "weights": os.path.join(_REPO_ROOT, "models", "murmur_crnn_circor.pth"),
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


def _default_murmur_repo():
    """Best-effort guess: $MURMUR_REPO, else a sibling clone next to this repo."""
    env = os.environ.get("MURMUR_REPO")
    if env:
        return env
    sibling = os.path.join(os.path.dirname(_REPO_ROOT), "AutomaticHeartSoundClassification")
    return sibling if os.path.isdir(sibling) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="wav file(s) or folder(s)")
    ap.add_argument("--window", choices=["slide", "center"], default="slide",
                    help="'slide' = avg over 50%%-overlap 5 s windows (default); "
                         "'center' = single center 5 s crop")
    ap.add_argument("--murmur-repo", default=_default_murmur_repo(),
                    help="path to a clone of SiyuLou/AutomaticHeartSoundClassification "
                         "(provides model/ + utils/). Defaults to $MURMUR_REPO or a "
                         "sibling 'AutomaticHeartSoundClassification' folder.")
    args = ap.parse_args()

    if not args.murmur_repo:
        raise SystemExit(
            "Could not locate the upstream murmur repo. Pass --murmur-repo PATH "
            "(clone of SiyuLou/AutomaticHeartSoundClassification) or set $MURMUR_REPO.")
    _load_upstream(args.murmur_repo)

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
