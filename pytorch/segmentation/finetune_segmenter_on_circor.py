"""
Fine-tune the fold-1 segmenter (trained on DavidSpringer) on CirCor's expert .tsv
segmentation labels -- to fix the pediatric Systole->Diastole leak.

NOT from scratch: starts from the existing checkpoint and keeps training.
Patient-grouped split (no patient in both train and test). FSST features are
precomputed ONCE (the bottleneck), then epochs are fast.

Reports baseline (un-fine-tuned) vs fine-tuned on the SAME held-out CirCor test
patients, frame-level. Saves the fine-tuned model + metrics + confusion matrices.

Usage (WSL pixi):  pixi run python finetune_segmenter_on_circor.py [n_train_rec n_test_rec epochs]
"""
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch.nn.utils import clip_grad_norm_

import eval_segmenter_on_circor as ev   # reuse FSST_T, gt_from_tsv, CIR, FS, FRAME, NAMES
from hss.model.lit_model_crf import LitModelCRF

OUT = "circor_finetune"
SEG_CKPT = ev.SEG_CKPT
FS, FRAME, STRIDE, MIN_TAIL = ev.FS, ev.FRAME, 1000, 256
NAMES = ev.NAMES
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def fsst(seg):
    return ev.FSST_T(torch.tensor(seg, dtype=torch.float32)).numpy().astype(np.float32)


def load_sig(wav):
    a, fs = sf.read(wav)
    if a.ndim > 1:
        a = a.mean(1)
    s = scipy.signal.resample_poly(a, FS, fs) if fs != FS else a
    return s / (np.std(s) + 1e-9)


def patient_split(n_train_rec, n_test_rec):
    by_pat = defaultdict(list)
    for w in sorted(glob.glob(f"{ev.CIR}/*.wav")):
        by_pat[os.path.basename(w).split("_")[0]].append(w)
    pats = sorted(by_pat)
    random.Random(SEED).shuffle(pats)
    n_test_pat = max(1, int(0.25 * len(pats)))
    test_pats, train_pats = pats[:n_test_pat], pats[n_test_pat:]
    tr = [w for p in train_pats for w in by_pat[p]][:n_train_rec]
    te = [w for p in test_pats for w in by_pat[p]][:n_test_rec]
    return tr, te


def build_train_windows(wavs):
    """fully-annotated 2 s windows -> (feat (2000,44), labels (2000,))."""
    out = []
    t0 = time.time()
    for i, w in enumerate(wavs):
        tsv = w[:-4] + ".tsv"
        if not os.path.isfile(tsv):
            continue
        sig = load_sig(w)
        gt = ev.gt_from_tsv(tsv, len(sig))
        for st in range(0, len(sig) - FRAME + 1, STRIDE):
            lab = gt[st:st + FRAME]
            if (lab < 0).any():           # skip windows with unannotated frames
                continue
            out.append((fsst(sig[st:st + FRAME]), lab.astype(np.int64)))
        if (i + 1) % 50 == 0:
            print(f"  train feat {i+1}/{len(wavs)}  windows={len(out)}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def build_test(wavs):
    """per recording: (list of window feats, gt_full) for frame-level eval."""
    out = []
    t0 = time.time()
    for i, w in enumerate(wavs):
        tsv = w[:-4] + ".tsv"
        if not os.path.isfile(tsv):
            continue
        sig = load_sig(w)
        gt = ev.gt_from_tsv(tsv, len(sig))
        if (gt >= 0).sum() == 0:
            continue
        feats = [fsst(sig[st:st + FRAME]) for st in range(0, len(sig), FRAME)
                 if len(sig[st:st + FRAME]) >= MIN_TAIL]
        out.append((feats, gt))
        if (i + 1) % 50 == 0:
            print(f"  test feat {i+1}/{len(wavs)}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def load_model(batch_size):
    m = LitModelCRF(input_size=44, batch_size=batch_size, device=DEV)
    ck = torch.load(SEG_CKPT, map_location=DEV, weights_only=False)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.endswith(("h0", "c0"))}
    m.load_state_dict(sd, strict=False)
    return m.to(DEV)


def finetune(windows, epochs, batch, lr):
    m = load_model(batch).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(windows)
        tot, nb = 0.0, 0
        for i in range(0, len(windows) - batch + 1, batch):
            chunk = windows[i:i + batch]
            x = torch.from_numpy(np.stack([c[0] for c in chunk])).to(DEV)
            y = torch.from_numpy(np.stack([c[1] for c in chunk])).to(DEV)
            loss = m.model.loss(x, y)
            opt.zero_grad(); loss.backward(); clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep+1:2d}/{epochs} | loss {tot/max(nb,1):.4f}", flush=True)
    return m


@torch.no_grad()
def evaluate(state_dict, test):
    m = load_model(1)
    m.load_state_dict({k: v for k, v in state_dict.items() if not k.endswith(("h0", "c0"))},
                      strict=False)
    m.eval()
    cm = np.zeros((4, 4), np.int64)
    for feats, gt in test:
        pred = []
        for f in feats:
            d = m.model.decode(torch.from_numpy(f[None]).to(DEV))[0]
            pred.append(np.asarray(d.cpu().numpy() if isinstance(d, torch.Tensor) else d, np.int64))
        pred = np.concatenate(pred) if pred else np.zeros(0, np.int64)
        mm = min(len(pred), len(gt)); p, g = pred[:mm], gt[:mm]
        mask = g >= 0
        for t, pr in zip(g[mask], p[mask]):
            cm[t, pr] += 1
    return cm


def cm_metrics(cm):
    rec = np.array([cm[i, i] / cm[i].sum() if cm[i].sum() else 0 for i in range(4)])
    prec = np.array([cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0 for i in range(4)])
    f1 = np.array([2*p*r/(p+r) if (p+r) else 0 for p, r in zip(prec, rec)])
    return dict(overall=float(np.trace(cm)/cm.sum()), balanced=float(rec.mean()),
                recall=rec.tolist(), precision=prec.tolist(), f1=f1.tolist())


def main():
    a = sys.argv[1:]
    n_train = int(a[0]) if len(a) > 0 else 300
    n_test = int(a[1]) if len(a) > 1 else 120
    epochs = int(a[2]) if len(a) > 2 else 8
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    tr_wavs, te_wavs = patient_split(n_train, n_test)
    print(f"train recordings: {len(tr_wavs)} | test recordings: {len(te_wavs)}")
    print("extracting FSST features (one-time)...")
    train_windows = build_train_windows(tr_wavs)
    test = build_test(te_wavs)
    print(f"train windows: {len(train_windows)} | test recordings usable: {len(test)}")

    base_ck = torch.load(SEG_CKPT, map_location=DEV, weights_only=False)
    cm_base = evaluate(base_ck["state_dict"], test)

    ft = finetune(train_windows, epochs, batch=32, lr=1e-3)
    cm_ft = evaluate(ft.state_dict(), test)

    mb, mf = cm_metrics(cm_base), cm_metrics(cm_ft)
    arch_note = "fold-1 segmenter fine-tuned on CirCor .tsv (patient-split, not from scratch)"
    torch.save({"state_dict": ft.state_dict(), "input_size": 44,
                "note": arch_note, "base_ckpt": SEG_CKPT,
                "test_overall": mf["overall"], "test_balanced": mf["balanced"]},
               f"{OUT}/segmenter_finetuned_circor.pth")

    summary = [f"=== Segmenter CirCor fine-tune (held-out test, {len(test)} recordings) ===",
               f"{'':12} {'overall':>8} {'balanced':>9}   per-class recall S1/Sys/S2/Dia",
               f"{'baseline':12} {mb['overall']*100:>7.1f}% {mb['balanced']*100:>8.1f}%   " +
               " ".join(f"{r*100:4.0f}" for r in mb['recall']),
               f"{'fine-tuned':12} {mf['overall']*100:>7.1f}% {mf['balanced']*100:>8.1f}%   " +
               " ".join(f"{r*100:4.0f}" for r in mf['recall'])]
    txt = "\n".join(summary)
    print("\n" + txt)
    open(f"{OUT}/results.txt", "w").write(txt + "\n")
    json.dump({"baseline": mb, "fine_tuned": mf,
               "cm_baseline": cm_base.tolist(), "cm_finetuned": cm_ft.tolist(),
               "n_train_windows": len(train_windows), "n_test_recordings": len(test)},
              open(f"{OUT}/metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, (cm, ttl) in enumerate([(cm_base, "baseline"), (cm_ft, "fine-tuned")]):
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax[k].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(NAMES)
        ax[k].set_yticks(range(4)); ax[k].set_yticklabels(NAMES)
        ax[k].set_xlabel("predicted"); ax[k].set_ylabel("true")
        m = cm_metrics(cm)
        ax[k].set_title(f"{ttl}: overall {m['overall']*100:.1f}%, bal {m['balanced']*100:.1f}%")
        for i in range(4):
            for j in range(4):
                ax[k].text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                           color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.suptitle("Segmenter on CirCor: baseline vs CirCor fine-tuned (held-out patients)")
    fig.tight_layout(); fig.savefig(f"{OUT}/finetune_comparison.png", dpi=130); plt.close(fig)
    print(f"\nSaved -> {OUT}/")


if __name__ == "__main__":
    main()
