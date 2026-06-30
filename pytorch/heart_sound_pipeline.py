"""
End-to-end heart-sound analysis pipeline.

For each audio file (or every audio file in a folder) it runs:

  STAGE 1  Murmur detection      -> CirCor CRNN (log-mel + CNN-BiLSTM)
           "Present" / "Absent" with a confidence number.

  STAGE 2  (only if a murmur is present)
           Segmentation          -> fold-1 LSTM+CRF segmenter (FSST features)
           splits the recording into S1 / Systole / S2 / Diastole.

  STAGE 3  Timing localization   -> pool murmur-band energy by cardiac phase
           reports Systolic vs Diastolic energy share as percentages and
           names the murmur timing.

Run from this repository, pointing to the two cloned upstream repos:
    python pytorch/heart_sound_pipeline.py audio.wav --seg-repo /path/to/heart-sounds-segmentation --murmur-repo /path/to/AutomaticHeartSoundClassification
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REPO_SEG = None
REPO_MURMUR = None
MURMUR_CFG = os.path.join(ROOT, "configs", "config_crnn.json")
MURMUR_WEIGHTS = os.path.join(ROOT, "models", "murmur_crnn_circor.pth")
SEG_WEIGHTS = os.path.join(ROOT, "models", "segmenter_finetuned_circor.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- murmur model -------------------------------------------------------------#
MURMUR_CLASSES = ["Present", "Absent"]       # index 0 = murmur present
SR_MURMUR = 2000
HOP_MS = 15
CYCLE_LEN = int(5 * 1000 / HOP_MS)           # 333 frames per 5 s window

# --- segmenter ----------------------------------------------------------------#
FS_SEG = 1000
FRAME = 2000
MIN_TAIL = 256
MURMUR_BAND = (150, 450)
SEG_NAMES = {0: "S1", 1: "Systole", 2: "S2", 3: "Diastole"}

AUDIO_EXTS = ("*.wav", "*.flac", "*.ogg")


# ============================== STAGE 1: murmur ============================== #
def load_murmur_model():
    import model.model as module_arch
    from utils.util import read_audio
    # NOTE: not using upstream's LogMelExtractor -- its positional librosa call
    # breaks on librosa>=0.10. murmur_feature() computes the identical log-mel.
    from utils.audio_feature_extractor import standard_normal_variate

    arch = json.load(open(MURMUR_CFG))["arch"]
    net = getattr(module_arch, arch["type"])(**arch["args"])
    ckpt = torch.load(MURMUR_WEIGHTS, weights_only=False, map_location=DEVICE)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    return net, read_audio, standard_normal_variate


def murmur_feature(path, read_audio, snv_fn):
    import librosa
    audio, fs = read_audio(path, target_fs=SR_MURMUR, filter=True)
    # version-safe log-mel (== upstream LogMelExtractor log=True, snv=False):
    # 128 mels, 15 ms hop, 25 ms window, log(mel+1e-8)
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=float), sr=fs, n_mels=128,
        hop_length=int(fs * HOP_MS / 1000), win_length=int(fs * 25 / 1000))
    feat = np.log(mel + 1e-8)                                     # (128, T)
    feat = snv_fn(feat)
    d = librosa.feature.delta(feat)
    d2 = librosa.feature.delta(d)
    feat = np.concatenate((feat, d, d2), axis=0)                  # (384, T)
    return feat.astype(np.float32)


def murmur_windows(feat):
    _, T = feat.shape
    if T < CYCLE_LEN:
        yield np.pad(feat, ((0, 0), (0, CYCLE_LEN - T)), mode="wrap")
        return
    hop = CYCLE_LEN // 2
    starts = list(range(0, T - CYCLE_LEN + 1, hop))
    if starts[-1] != T - CYCLE_LEN:
        starts.append(T - CYCLE_LEN)
    for s in starts:
        yield feat[:, s:s + CYCLE_LEN]


@torch.no_grad()
def detect_murmur(net, feat):
    x = torch.from_numpy(np.stack(list(murmur_windows(feat)))).to(DEVICE).float()
    probs = F.softmax(net(x), dim=1).mean(0).cpu().numpy()        # (P_present, P_absent)
    return float(probs[0]), float(probs[1])


# ===================== STAGE 2+3: segmentation + timing ===================== #
def load_segmenter():
    from hss.model.lit_model_crf import LitModelCRF
    model = LitModelCRF(input_size=44, batch_size=1, device=DEVICE)
    ckpt = torch.load(SEG_WEIGHTS, map_location=DEVICE, weights_only=False)
    sd = {k: v for k, v in ckpt["state_dict"].items()
          if not (k.endswith("h0") or k.endswith("c0"))}
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    return model


SEG_WINDOW = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
SEG_TRUNC = (25, 200)
SEG_NFFT = 128


class FSSTSsqueezepy:
    """Drop-in replacement for the repo's `ssq`-based FSST, using ssqueezepy.

    Produces the SAME (seq_len, 44) feature the segmenter was trained on:
    synchrosqueezed STFT (modulated=False to match `ssq`'s phase convention),
    truncated to 25-200 Hz, real & imag z-scored separately then stacked.
    Verified ~0.99 feature correlation vs `ssq` -> reuses the trained weights.
    Works on Windows and can run on GPU (gpu=True)."""

    def __init__(self, fs=1000, window=SEG_WINDOW, truncate_freq=SEG_TRUNC,
                 n_fft=SEG_NFFT, gpu=False):
        if gpu:
            os.environ["SSQ_GPU"] = "1"
        import ssqueezepy  # noqa: F401  (import after env var so GPU mode registers)
        self.sq = ssqueezepy
        self.fs, self.window, self.n_fft = fs, window, n_fft
        self.lo, self.hi = truncate_freq

    @staticmethod
    def _np(a):
        # GPU mode returns CUDA torch tensors; bring them to host first
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def __call__(self, x):
        xn = x.detach().cpu().numpy().astype(np.float64)
        out = self.sq.ssq_stft(xn, window=self.window, n_fft=self.n_fft,
                               hop_len=1, fs=self.fs, modulated=False)
        Tx = self._np(out[0])                         # (n_freq, n_time) complex
        freqs = self._np(out[2]).squeeze()
        m = (freqs >= self.lo) & (freqs <= self.hi)
        s = Tx[m, :]
        r, i = s.real, s.imag
        r = (r - r.mean()) / (r.std() + 1e-12)
        i = (i - i.mean()) / (i.std() + 1e-12)
        z = np.concatenate([r, i], axis=0).T.astype(np.float32)   # (n_time, 44)
        return torch.from_numpy(z)


def make_fsst(backend="ssq", gpu=False):
    if backend == "ssqueezepy":
        return FSSTSsqueezepy(fs=FS_SEG, gpu=gpu)
    from hss.transforms import FSST                  # ssq (C++), linux-only
    return FSST(FS_SEG, window=SEG_WINDOW, truncate_freq=SEG_TRUNC, stack=True)


def segment(model, fsst, sig_1k):
    out = []
    for start in range(0, len(sig_1k), FRAME):
        seg = sig_1k[start:start + FRAME]
        if len(seg) < MIN_TAIL:
            break
        feat = fsst(torch.tensor(seg, dtype=torch.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            decoded = model.model.decode(feat)[0]
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.cpu().numpy()
        out.append(np.asarray(decoded, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def murmur_energy(sig_1k):
    sos = scipy.signal.butter(4, MURMUR_BAND, btype="band", fs=FS_SEG, output="sos")
    return scipy.signal.sosfiltfilt(sos, sig_1k) ** 2


def analyze_timing(model, fsst, path):
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    sig = scipy.signal.resample_poly(audio, FS_SEG, fs) if fs != FS_SEG else audio
    sig = (sig / (np.std(sig) + 1e-9)).astype(np.float64)

    labels = segment(model, fsst, sig)
    energy = murmur_energy(sig)
    m = min(len(labels), len(energy))
    labels, energy = labels[:m], energy[:m]

    E = {k: (float(np.median(energy[labels == k])) if (labels == k).any() else 0.0)
         for k in range(4)}
    frac = {k: float((labels == k).mean()) for k in range(4)}
    n_cycles = int(np.sum((labels[1:] == 0) & (labels[:-1] != 0)))  # entries into S1
    return E, frac, n_cycles


# ================================ orchestration ============================== #
def to_wsl_path(p):
    """Translate a Windows path (C:\\foo\\bar) to WSL form (/mnt/c/foo/bar) when
    running under WSL, so either path style works from the command line."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)", p)
    if m and os.path.isdir("/mnt"):
        return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    return p


def require_path(path, kind):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def configure_repos(args):
    global REPO_SEG, REPO_MURMUR, MURMUR_CFG, MURMUR_WEIGHTS, SEG_WEIGHTS

    REPO_SEG = require_path(to_wsl_path(args.seg_repo), "segmentation repo")
    REPO_MURMUR = require_path(to_wsl_path(args.murmur_repo), "murmur repo")
    MURMUR_CFG = require_path(to_wsl_path(args.murmur_config), "murmur config")
    MURMUR_WEIGHTS = require_path(to_wsl_path(args.murmur_weights), "murmur weights")
    SEG_WEIGHTS = require_path(to_wsl_path(args.segmenter_weights), "segmenter weights")

    require_path(os.path.join(REPO_SEG, "hss"), "segmentation repo hss package")
    require_path(os.path.join(REPO_MURMUR, "model"), "murmur repo model package")
    require_path(os.path.join(REPO_MURMUR, "utils"), "murmur repo utils package")

    sys.path.insert(0, REPO_SEG)
    sys.path.insert(0, REPO_MURMUR)


def collect_paths(args_paths):
    out = []
    for p in args_paths:
        p = to_wsl_path(p)
        if os.path.isdir(p):
            for pat in AUDIO_EXTS:
                out += sorted(glob.glob(os.path.join(p, pat)))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="audio file(s) or folder(s)")
    ap.add_argument("--seg-repo", required=True,
                    help="path to cloned alvgaona/heart-sounds-segmentation repo")
    ap.add_argument("--murmur-repo", required=True,
                    help="path to cloned SiyuLou/AutomaticHeartSoundClassification repo")
    ap.add_argument("--murmur-config", default=MURMUR_CFG,
                    help="CRNN architecture config (default: this repo's configs/config_crnn.json)")
    ap.add_argument("--murmur-weights", default=MURMUR_WEIGHTS,
                    help="murmur CRNN weights (default: this repo's models/murmur_crnn_circor.pth)")
    ap.add_argument("--segmenter-weights", default=SEG_WEIGHTS,
                    help="segmenter weights (default: this repo's models/segmenter_finetuned_circor.pth)")
    ap.add_argument("--csv", help="optional path to write a results table")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P[Present] cutoff to call a murmur (default 0.5)")
    ap.add_argument("--fsst-backend", choices=["ssq", "ssqueezepy"], default="ssq",
                    help="FSST implementation for segmentation. 'ssq' (default, C++, "
                         "linux-only) or 'ssqueezepy' (Windows-friendly, GPU-capable, "
                         "~0.99 feature-equivalent so weights are reused)")
    ap.add_argument("--fsst-gpu", action="store_true",
                    help="run ssqueezepy FSST on GPU (only affects --fsst-backend ssqueezepy)")
    args = ap.parse_args()

    configure_repos(args)

    print(f"Device: {DEVICE}")
    print(f"Murmur repo  : {REPO_MURMUR}")
    print(f"Segment repo : {REPO_SEG}")
    print(f"Murmur model : {MURMUR_WEIGHTS}")
    print(f"Segmenter    : {SEG_WEIGHTS}")
    print(f"FSST backend : {args.fsst_backend}"
          f"{' (GPU)' if args.fsst_gpu and args.fsst_backend == 'ssqueezepy' else ''}\n")

    t_load = time.perf_counter()
    net, read_audio, snv = load_murmur_model()
    seg_model = load_segmenter()
    fsst = make_fsst(args.fsst_backend, args.fsst_gpu)
    print(f"(models + FSST backend loaded in {time.perf_counter() - t_load:.1f}s)\n")

    paths = collect_paths(args.paths)
    if not paths:
        print("No audio files found."); return

    rows = []
    wall0 = time.perf_counter()
    for path in paths:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: NOT FOUND\n"); continue
        name = os.path.basename(path)

        t0 = time.perf_counter()
        feat = murmur_feature(path, read_audio, snv)
        secs = feat.shape[1] * HOP_MS / 1000
        p_present, p_absent = detect_murmur(net, feat)
        t_detect = time.perf_counter() - t0

        present = p_present >= args.threshold
        verdict = "PRESENT" if present else "ABSENT"
        conf = p_present if present else p_absent

        print(f"=== {name}  (~{secs:.1f}s) ===")
        print(f"  [1] Murmur detection : {verdict}   confidence {100 * conf:.1f}%"
              f"   (P[Present]={p_present:.3f}, P[Absent]={p_absent:.3f})")

        row = dict(file=name, seconds=round(secs, 1), murmur=verdict,
                   P_present=round(p_present, 4), P_absent=round(p_absent, 4),
                   timing="", systolic_pct="", diastolic_pct="", n_cycles="",
                   t_detect_s=round(t_detect, 3), t_segtiming_s="", t_total_s="")

        t_seg = 0.0
        if present:
            t1 = time.perf_counter()
            E, frac, n_cycles = analyze_timing(seg_model, fsst, path)
            t_seg = time.perf_counter() - t1
            e_sys, e_dia = E[1], E[3]
            tot = e_sys + e_dia + 1e-12
            sys_pct, dia_pct = 100 * e_sys / tot, 100 * e_dia / tot
            timing = "SYSTOLIC" if e_sys >= e_dia else "DIASTOLIC"

            print(f"  [2] Segmentation     : {n_cycles} cardiac cycles  "
                  f"(S1 {100*frac[0]:.0f}%, Sys {100*frac[1]:.0f}%, "
                  f"S2 {100*frac[2]:.0f}%, Dia {100*frac[3]:.0f}%)")
            print(f"  [3] Timing (murmur-band energy share):")
            mark_s = "  <-- murmur timing" if timing == "SYSTOLIC" else ""
            mark_d = "  <-- murmur timing" if timing == "DIASTOLIC" else ""
            print(f"        Systolic  : {sys_pct:5.1f}%{mark_s}")
            print(f"        Diastolic : {dia_pct:5.1f}%{mark_d}")

            row.update(timing=timing, systolic_pct=round(sys_pct, 1),
                       diastolic_pct=round(dia_pct, 1), n_cycles=n_cycles,
                       t_segtiming_s=round(t_seg, 3))
        else:
            print("      (no murmur -> segmentation/timing skipped)")

        t_total = t_detect + t_seg
        row["t_total_s"] = round(t_total, 3)
        seg_part = f" | segmentation+timing {t_seg:.2f}s" if present else ""
        print(f"  [runtime]            : detection {t_detect:.2f}s{seg_part}"
              f" | total {t_total:.2f}s  ({secs / t_total:.1f}x realtime)")
        print()
        rows.append(row)

    wall = time.perf_counter() - wall0
    print(f"Processed {len(rows)} file(s) in {wall:.1f}s "
          f"(avg {wall / max(len(rows), 1):.2f}s/file, excludes one-time model load).")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote results table -> {args.csv}")


if __name__ == "__main__":
    main()
