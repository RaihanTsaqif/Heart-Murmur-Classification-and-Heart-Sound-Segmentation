"""
Timing-localization test on CirCor using the fold-1 segmenter + murmur-band
energy pooling.

Pipeline per recording:
  1. load wav (4000 Hz) -> resample to 1000 Hz (segmenter rate)
  2. window into 2 s frames -> FSST (44-d) -> fold-1 LSTM+CRF -> per-sample labels
     (0=S1, 1=Systole, 2=S2, 3=Diastole)
  3. bandpass the 1000 Hz signal to the murmur band -> per-sample energy
  4. pool mean energy by segment label
  5. systolic_fraction = E_sys / (E_sys + E_dia); predict timing by argmax

We then compare against CirCor ground-truth timing (Systolic / Diastolic murmur
timing columns) for every Murmur=Present patient, using the most-audible-location
recording.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.signal
import torch

from hss.model.lit_model_crf import LitModelCRF
from hss.transforms import FSST


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sorted(glob.glob("lightning_logs/version_1/checkpoints/*.ckpt"))[-1]
CIR = "/mnt/c/Projects/Heart Sound AI Classifier/heart_sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
FS_SEG = 1000
FRAME = 2000          # 2 s windows (matches training sequence length)
MIN_TAIL = 256        # drop trailing window shorter than this
MURMUR_BAND = (150, 450)  # Hz, above S1/S2 fundamentals, within 1 kHz Nyquist

LABELS = {0: "S1", 1: "Systole", 2: "S2", 3: "Diastole"}


def load_model():
    model = LitModelCRF(input_size=44, batch_size=1, device=DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    sd = {k: v for k, v in ckpt["state_dict"].items()
          if not (k.endswith("h0") or k.endswith("c0"))}  # h0/c0 sized for batch 50
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert all(k.endswith(("h0", "c0")) for k in missing), missing
    model.eval().to(DEVICE)
    return model


FSST_T = FSST(
    1000,
    window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
    truncate_freq=(25, 200),
    stack=True,
)


def segment(model, sig_1k: np.ndarray) -> np.ndarray:
    """Return per-sample segment labels (0..3) for a 1000 Hz signal."""
    out = []
    for start in range(0, len(sig_1k), FRAME):
        seg = sig_1k[start:start + FRAME]
        if len(seg) < MIN_TAIL:
            break
        feat = FSST_T(torch.tensor(seg, dtype=torch.float32))      # (seq, 44)
        feat = feat.unsqueeze(0).to(DEVICE)                        # (1, seq, 44)
        with torch.no_grad():
            decoded = model.model.decode(feat)[0]                  # list/tensor len seq
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.cpu().numpy()
        out.append(np.asarray(decoded, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def murmur_energy(sig_1k: np.ndarray) -> np.ndarray:
    sos = scipy.signal.butter(4, MURMUR_BAND, btype="band", fs=FS_SEG, output="sos")
    return scipy.signal.sosfiltfilt(sos, sig_1k) ** 2


def process_recording(model, wav_path: str):
    import torchaudio
    sig, sr = torchaudio.load(wav_path)
    sig = sig[0].numpy().astype(np.float32)
    if sr != FS_SEG:
        sig = scipy.signal.resample_poly(sig, FS_SEG, sr)
    sig = sig / (np.std(sig) + 1e-9)

    labels = segment(model, sig)
    energy = murmur_energy(sig)
    m = min(len(labels), len(energy))
    labels, energy = labels[:m], energy[:m]

    E = {}
    frac = {}
    for k in range(4):
        mask = labels == k
        # median is robust to transient artifact spikes (a single loud click in
        # one segment would blow up a mean and flip the timing decision)
        E[k] = float(np.median(energy[mask])) if mask.any() else 0.0
        frac[k] = float(mask.mean())
    return E, frac, labels


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    model = load_model()

    df = pd.read_csv(f"{CIR}/training_data.csv")
    present = df[df["Murmur"] == "Present"].copy()
    if limit:
        present = present.head(limit)

    rows = []
    for i, (_, r) in enumerate(present.iterrows()):
        pid = int(r["Patient ID"])
        loc = r["Most audible location"]
        if not isinstance(loc, str):
            locs = str(r["Murmur locations"]).split("+")
            loc = locs[0] if locs and locs[0] != "nan" else None
        wav = f"{CIR}/training_data/{pid}_{loc}.wav" if loc else None
        if not wav or not os.path.isfile(wav):
            print(f"[skip] {pid} loc={loc} no wav", flush=True)
            continue

        E, frac, _ = process_recording(model, wav)
        e_sys, e_dia = E[1], E[3]
        sys_frac = e_sys / (e_sys + e_dia + 1e-12)
        pred = "Systolic" if e_sys >= e_dia else "Diastolic"

        gt_sys = pd.notna(r["Systolic murmur timing"])
        gt_dia = pd.notna(r["Diastolic murmur timing"])
        if gt_sys and gt_dia:
            truth = "Both"
        elif gt_sys:
            truth = "Systolic"
        else:
            truth = "Diastolic"

        rows.append(dict(
            pid=pid, loc=loc, truth=truth, pred=pred,
            E_S1=E[0], E_Sys=e_sys, E_S2=E[2], E_Dia=e_dia,
            sys_frac=sys_frac,
            sys_vs_s1s2=e_sys / (0.5 * (E[0] + E[2]) + 1e-12),
            frac_S1=frac[0], frac_Sys=frac[1], frac_S2=frac[2], frac_Dia=frac[3],
        ))
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(present)} processed", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("timing_results.csv", index=False)

    print("\n================ TIMING-LOCALIZATION RESULTS ================")
    print(f"recordings scored: {len(res)}")

    truth = res["truth"]
    sys_only = res[truth == "Systolic"]
    dia_any = res[truth.isin(["Diastolic", "Both"])]

    print("\n-- Systolic-timing cases (expect E_Sys > E_Dia) --")
    if len(sys_only):
        correct = (sys_only["pred"] == "Systolic").sum()
        print(f"  n={len(sys_only)}  predicted Systolic: {correct}/{len(sys_only)} "
              f"= {100 * correct / len(sys_only):.1f}%")
        print(f"  mean systolic_fraction = {sys_only['sys_frac'].mean():.3f} "
              f"(median {sys_only['sys_frac'].median():.3f})")

    print("\n-- Diastolic / Both cases (the hard, rare class) --")
    for _, x in dia_any.iterrows():
        print(f"  pid {x['pid']:>5} truth={x['truth']:<9} pred={x['pred']:<9} "
              f"E_Sys={x['E_Sys']:.3g} E_Dia={x['E_Dia']:.3g} sys_frac={x['sys_frac']:.3f}")

    print("\nSaved per-recording table -> timing_results.csv")


if __name__ == "__main__":
    main()
