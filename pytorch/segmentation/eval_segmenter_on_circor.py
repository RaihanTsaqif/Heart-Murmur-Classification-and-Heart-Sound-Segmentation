"""
Cross-dataset segmentation test: fold-1 segmenter (trained on DavidSpringer HSS)
evaluated against CirCor's EXPERT .tsv segmentation labels, frame by frame.

CirCor .tsv: rows of (start_sec, end_sec, label) with label 1=S1, 2=Systole,
3=S2, 4=Diastole, 0=unannotated. Segmenter outputs 0=S1,1=Sys,2=S2,3=Dia, so
ground-truth label k maps to k-1; label 0 regions are excluded from scoring.

Usage (WSL pixi):  CIRCOR_DIR=/path/to/the-circor-...-1.0.3 \
                   pixi run python eval_segmenter_on_circor.py [N_sample]
"""
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch

from hss.model.lit_model_crf import LitModelCRF
from hss.transforms import FSST

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))


# CirCor training_data dir. Set $CIRCOR_DIR to the dataset root (the
# 'the-circor-digiscope-phonocardiogram-dataset-1.0.3' folder; PhysioNet download,
# not bundled here). Empty when unset so importing this module never fails.
def _seg_ckpt():
    """$SEG_CKPT, else a local training checkpoint, else this repo's shipped weights."""
    env = os.environ.get("SEG_CKPT")
    if env:
        return env
    local = sorted(glob.glob("lightning_logs/version_1/checkpoints/*.ckpt"))
    if local:
        return local[-1]
    return os.path.join(_REPO_ROOT, "models", "segmenter_finetuned_circor.pth")


CIR = os.path.join(os.environ["CIRCOR_DIR"], "training_data") if os.environ.get("CIRCOR_DIR") else ""
SEG_CKPT = _seg_ckpt()
OUT = "circor_segmentation_eval"
FS, FRAME, MIN_TAIL = 1000, 2000, 256
NAMES = ["S1", "Systole", "S2", "Diastole"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def load_segmenter():
    m = LitModelCRF(input_size=44, batch_size=1, device=DEV)
    ck = torch.load(SEG_CKPT, map_location=DEV, weights_only=False)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.endswith(("h0", "c0"))}
    m.load_state_dict(sd, strict=False)
    m.eval().to(DEV)
    return m


FSST_T = FSST(FS, window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
              truncate_freq=(25, 200), stack=True)


def segment(model, sig):
    out = []
    for s in range(0, len(sig), FRAME):
        seg = sig[s:s + FRAME]
        if len(seg) < MIN_TAIL:
            break
        feat = FSST_T(torch.tensor(seg, dtype=torch.float32)).unsqueeze(0).to(DEV)
        with torch.no_grad():
            d = model.model.decode(feat)[0]
        out.append(np.asarray(d.cpu().numpy() if isinstance(d, torch.Tensor) else d, np.int64))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def gt_from_tsv(tsv, n):
    """per-sample ground truth at 1 kHz; -1 = unannotated/excluded."""
    df = pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "label"])
    gt = np.full(n, -1, np.int64)
    for _, r in df.iterrows():
        lab = int(r["label"])
        if lab in (1, 2, 3, 4):
            a, b = int(round(r["start"] * FS)), int(round(r["end"] * FS))
            gt[max(0, a):min(n, b)] = lab - 1
    return gt


def main():
    if not CIR:
        raise SystemExit("Set $CIRCOR_DIR to the CirCor 1.0.3 dataset root "
                         "(PhysioNet download; not bundled in this repo).")
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    os.makedirs(OUT, exist_ok=True)
    model = load_segmenter()

    wavs = sorted(glob.glob(f"{CIR}/*.wav"))
    rng = np.random.default_rng(SEED)
    if n_sample and n_sample < len(wavs):
        wavs = list(rng.choice(wavs, size=n_sample, replace=False))
    print(f"segmenter: {os.path.basename(SEG_CKPT)} | scoring {len(wavs)} recordings")

    cm = np.zeros((4, 4), np.int64)        # cm[true, pred]
    rows = []
    t0 = time.time()
    for i, wav in enumerate(wavs):
        tsv = wav[:-4] + ".tsv"
        if not os.path.isfile(tsv):
            continue
        audio, fs = sf.read(wav)
        if audio.ndim > 1:
            audio = audio.mean(1)
        sig = scipy.signal.resample_poly(audio, FS, fs) if fs != FS else audio
        sig = sig / (np.std(sig) + 1e-9)

        pred = segment(model, sig)
        gt = gt_from_tsv(tsv, len(sig))
        m = min(len(pred), len(gt))
        pred, gt = pred[:m], gt[:m]
        mask = gt >= 0
        if mask.sum() == 0:
            continue
        p, g = pred[mask], gt[mask]
        for t, pr in zip(g, p):
            cm[t, pr] += 1
        acc = float((p == g).mean())
        rows.append(dict(record=os.path.basename(wav)[:-4], frames=int(mask.sum()),
                         accuracy=round(acc, 4)))
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(wavs)}  ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/per_recording.csv", index=False)

    total = cm.sum()
    overall_acc = float(np.trace(cm) / total)
    recall = np.array([cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else 0.0 for i in range(4)])
    prec = np.array([cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0 for i in range(4)])
    f1 = np.array([2 * p * r / (p + r) if (p + r) else 0.0 for p, r in zip(prec, recall)])
    bal_acc = float(recall.mean())

    metrics = dict(
        n_recordings=int(len(res)), n_frames=int(total),
        overall_accuracy=round(overall_acc, 4), balanced_accuracy=round(bal_acc, 4),
        per_class={NAMES[i]: dict(recall=round(float(recall[i]), 4),
                                  precision=round(float(prec[i]), 4),
                                  f1=round(float(f1[i]), 4)) for i in range(4)},
        confusion_matrix=cm.tolist(),
    )
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)

    lines = [f"=== fold-1 segmenter -> CirCor expert .tsv labels (frame-level) ===",
             f"recordings scored : {len(res)}   frames scored: {total:,}",
             f"overall accuracy  : {overall_acc*100:.2f}%",
             f"balanced accuracy : {bal_acc*100:.2f}%  (mean per-class recall)", ""]
    for i in range(4):
        lines.append(f"  {NAMES[i]:9s} recall {recall[i]*100:5.1f}%  "
                     f"precision {prec[i]*100:5.1f}%  F1 {f1[i]*100:5.1f}%")
    summary = "\n".join(lines)
    print("\n" + summary)
    open(f"{OUT}/results.txt", "w").write(summary + "\n")

    # confusion matrix (row-normalized)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(NAMES)
    ax.set_yticks(range(4)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (CirCor expert)")
    ax.set_title(f"Segmenter vs CirCor expert labels\noverall {overall_acc*100:.1f}%, "
                 f"balanced {bal_acc*100:.1f}%  ({len(res)} recordings)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]*100:.0f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)
    print(f"\nSaved -> {OUT}/")


if __name__ == "__main__":
    main()
