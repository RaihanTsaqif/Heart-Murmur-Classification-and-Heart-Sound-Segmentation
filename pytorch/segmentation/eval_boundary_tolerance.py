"""
Boundary-tolerance diagnostic for the segmenter-vs-CirCor evaluation.

A frame's prediction is counted correct (at tolerance T ms) if its predicted
label appears anywhere in the expert labels within +/-T frames (1 frame = 1 ms
at 1 kHz). T=0 is the strict frame accuracy. If accuracy jumps a lot as T grows,
the errors are mostly boundary jitter; if it barely moves, they're genuine
whole-segment misclassifications.

Usage (WSL pixi):  pixi run python eval_boundary_tolerance.py [N_sample]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
import torch

import eval_segmenter_on_circor as ev

TOL_MS = [0, 10, 20, 40, 60]
OUT = "circor_segmentation_eval"
SEED = 0


def main():
    if not ev.CIR:
        raise SystemExit("Set $CIRCOR_DIR to the CirCor 1.0.3 dataset root "
                         "(PhysioNet download; not bundled in this repo).")
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model = ev.load_segmenter()
    wavs = sorted(glob.glob(f"{ev.CIR}/*.wav"))
    rng = np.random.default_rng(SEED)
    if n_sample and n_sample < len(wavs):
        wavs = list(rng.choice(wavs, size=n_sample, replace=False))
    print(f"boundary-tolerance check on {len(wavs)} recordings")

    # accumulators
    correct = {t: 0 for t in TOL_MS}                       # overall correct per tolerance
    cls_total = np.zeros(4, np.int64)
    cls_correct = {t: np.zeros(4, np.int64) for t in TOL_MS}
    total = 0
    t0 = time.time()
    for i, wav in enumerate(wavs):
        tsv = wav[:-4] + ".tsv"
        if not os.path.isfile(tsv):
            continue
        audio, fs = sf.read(wav)
        if audio.ndim > 1:
            audio = audio.mean(1)
        sig = scipy.signal.resample_poly(audio, ev.FS, fs) if fs != ev.FS else audio
        sig = sig / (np.std(sig) + 1e-9)

        pred = ev.segment(model, sig)
        gt = ev.gt_from_tsv(tsv, len(sig))
        m = min(len(pred), len(gt))
        pred, gt = pred[:m], gt[:m]
        mask = gt >= 0
        if mask.sum() == 0:
            continue
        idx = np.arange(m)
        for t in TOL_MS:
            if t == 0:
                ok = (pred == gt)
            else:
                size = 2 * t + 1
                # gt_near[c, i] = True if class c appears in gt[i-t : i+t]
                near = np.stack([scipy.ndimage.maximum_filter1d(
                    (gt == c).astype(np.uint8), size=size) for c in range(4)])
                ok = near[pred, idx] > 0
            ok_m = ok[mask]
            correct[t] += int(ok_m.sum())
            for c in range(4):
                cls_correct[t][c] += int(ok_m[gt[mask] == c].sum())
        g = gt[mask]
        for c in range(4):
            cls_total[c] += int((g == c).sum())
        total += int(mask.sum())
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(wavs)}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== Boundary-tolerance results (frames scored: "
          f"{total:,}) ===")
    print(f"{'tol(ms)':>8} {'overall':>9} {'balanced':>9}   "
          f"{'S1':>6} {'Sys':>6} {'S2':>6} {'Dia':>6}   (per-class recall)")
    rows = []
    for t in TOL_MS:
        overall = correct[t] / total
        rec = cls_correct[t] / cls_total.clip(min=1)
        bal = float(rec.mean())
        print(f"{t:>8} {overall*100:>8.1f}% {bal*100:>8.1f}%   "
              f"{rec[0]*100:>5.1f}% {rec[1]*100:>5.1f}% {rec[2]*100:>5.1f}% {rec[3]*100:>5.1f}%")
        rows.append(dict(tol_ms=t, overall=round(overall, 4), balanced=round(bal, 4),
                         recall_S1=round(float(rec[0]), 4), recall_Sys=round(float(rec[1]), 4),
                         recall_S2=round(float(rec[2]), 4), recall_Dia=round(float(rec[3]), 4)))
    json.dump({"n_recordings": len(wavs), "n_frames": total, "by_tolerance": rows},
              open(f"{OUT}/boundary_tolerance.json", "w"), indent=2)
    print(f"\nSaved -> {OUT}/boundary_tolerance.json")


if __name__ == "__main__":
    main()
