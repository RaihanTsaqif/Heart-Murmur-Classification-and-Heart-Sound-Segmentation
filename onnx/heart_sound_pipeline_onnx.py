"""
Standalone ONNX heart-sound pipeline (no PyTorch needed at inference).

Per audio file:
  [1] Murmur detection  -> murmur_crnn_circor.onnx   (log-mel CRNN) -> Present/Absent + confidence
  [2] Segmentation      -> segmenter_emissions.onnx + Viterbi (CRF tables) -> S1/Sys/S2/Dia  (if Present)
  [3] Timing            -> murmur-band energy pooled by phase -> Systolic/Diastolic %

Feature extraction still runs here (ONNX only covers the neural nets):
  - murmur:  resample 2 kHz -> bandpass 25-400 -> log-mel 128 -> z-score -> +delta+delta2
  - segmenter: resample 1 kHz -> FSST (ssqueezepy, modulated=False) 25-200 Hz -> real/imag stack

Deps:  pip install onnxruntime numpy scipy soundfile librosa ssqueezepy
Usage: python heart_sound_pipeline_onnx.py <file_or_folder> [--threshold 0.5]
"""
import argparse
import glob
import os
import re
import time

import librosa
import numpy as np
import onnxruntime as ort
import scipy.signal
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
MURMUR_ONNX = os.path.join(HERE, "murmur_crnn_circor.onnx")
SEG_ONNX = os.path.join(HERE, "segmenter_emissions.onnx")
CRF_NPZ = os.path.join(HERE, "segmenter_crf_transitions.npz")

# murmur (must match training)
SR_M, HOP_MS, CYCLE = 2000, 15, 333          # 333 frames = 5 s window
# segmenter (must match training)
FS_S, FRAME, MIN_TAIL = 1000, 2000, 256
FSST_WIN = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
TRUNC, NFFT = (25, 200), 128
MURMUR_BAND = (150, 450)
SEG_NAMES = np.array(["S1", "Systole", "S2", "Diastole"])
AUDIO_EXTS = ("*.wav", "*.flac", "*.ogg")


def to_wsl_path(p):
    m = re.match(r"^([A-Za-z]):[\\/](.*)", p)
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}" if (m and os.path.isdir("/mnt")) else p


# ============================== [1] murmur ================================== #
def murmur_feature(path):
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if fs != SR_M:
        audio = librosa.resample(audio.astype(float), orig_sr=fs, target_sr=SR_M)
    b, a = scipy.signal.butter(5, [25, 400], btype="bandpass", fs=SR_M)
    audio = scipy.signal.lfilter(b, a, audio)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR_M, n_mels=128,
                                         hop_length=int(SR_M * HOP_MS / 1000),
                                         win_length=int(SR_M * 25 / 1000))
    feat = np.log(mel + 1e-8)
    feat = (feat - feat.mean()) / (feat.std() + 1e-12)
    d = librosa.feature.delta(feat); d2 = librosa.feature.delta(d)
    return np.concatenate([feat, d, d2], 0).astype(np.float32)   # (384, T)


def murmur_windows(feat):
    T = feat.shape[1]
    if T < CYCLE:
        return [np.pad(feat, ((0, 0), (0, CYCLE - T)), mode="wrap")]
    hop = CYCLE // 2
    starts = list(range(0, T - CYCLE + 1, hop))
    if starts[-1] != T - CYCLE:
        starts.append(T - CYCLE)
    return [feat[:, s:s + CYCLE] for s in starts]


def detect_murmur(sess, path):
    x = np.stack(murmur_windows(murmur_feature(path))).astype(np.float32)
    logits = sess.run(None, {"log_mel": x})[0]                   # (W, 2)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).mean(0)                    # mean softmax
    return float(p[0]), float(p[1])                             # P[Present], P[Absent]


# ===================== [2] segmentation + [3] timing ======================= #
def fsst_features(sig):
    import ssqueezepy as sq
    out = sq.ssq_stft(sig.astype(np.float64), window=FSST_WIN, n_fft=NFFT,
                      hop_len=1, fs=FS_S, modulated=False)
    Tx = np.asarray(out[0] if not hasattr(out[0], "cpu") else out[0].cpu().numpy())
    f = np.asarray(out[2]).squeeze()
    m = (f >= TRUNC[0]) & (f <= TRUNC[1])
    s = Tx[m, :]
    r, i = s.real, s.imag
    r = (r - r.mean()) / (r.std() + 1e-12)
    i = (i - i.mean()) / (i.std() + 1e-12)
    return np.concatenate([r, i], 0).T.astype(np.float32)       # (seq, 44)


def viterbi(em, start, end, trans):
    T, K = em.shape
    score = start + em[0]; back = np.zeros((T, K), int)
    for t in range(1, T):
        mm = score[:, None] + trans
        back[t] = mm.argmax(0); score = mm.max(0) + em[t]
    score = score + end
    last = int(score.argmax()); path = [last]
    for t in range(T - 1, 0, -1):
        last = int(back[t, last]); path.append(last)
    return np.array(path[::-1])


def segment_and_time(seg_sess, crf, path):
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    sig = scipy.signal.resample_poly(audio, FS_S, fs) if fs != FS_S else audio
    sig = sig / (np.std(sig) + 1e-9)

    labels = []
    for st in range(0, len(sig), FRAME):
        seg = sig[st:st + FRAME]
        if len(seg) < MIN_TAIL:
            break
        em = seg_sess.run(None, {"fsst": fsst_features(seg)[None]})[0][0]   # (seq, 4)
        labels.append(viterbi(em, crf["start_transitions"], crf["end_transitions"], crf["transitions"]))
    labels = np.concatenate(labels) if labels else np.zeros(0, int)

    sos = scipy.signal.butter(4, MURMUR_BAND, btype="band", fs=FS_S, output="sos")
    energy = scipy.signal.sosfiltfilt(sos, sig) ** 2
    m = min(len(labels), len(energy)); labels, energy = labels[:m], energy[:m]

    E = {k: (float(np.median(energy[labels == k])) if (labels == k).any() else 0.0) for k in range(4)}
    frac = {k: float((labels == k).mean()) for k in range(4)}
    n_cycles = int(np.sum((labels[1:] == 0) & (labels[:-1] != 0)))
    return E, frac, n_cycles


def collect(paths):
    out = []
    for p in paths:
        p = to_wsl_path(p)
        if os.path.isdir(p):
            for pat in AUDIO_EXTS:
                out += sorted(glob.glob(os.path.join(p, pat)))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    t_load = time.perf_counter()
    # Try GPU, fall back to CPU. (Models are tiny + FSST is CPU-bound, so the GPU
    # speedup for this pipeline is minimal -- CPU is perfectly fine.)
    try:
        ort.preload_dlls()   # load CUDA/cuDNN from nvidia pip packages (onnxruntime>=1.21)
    except Exception:
        pass
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    m_sess = ort.InferenceSession(MURMUR_ONNX, providers=prov)
    s_sess = ort.InferenceSession(SEG_ONNX, providers=prov)
    crf = np.load(CRF_NPZ)
    print(f"ONNX providers: {m_sess.get_providers()[0]}")
    print(f"(models loaded in {time.perf_counter() - t_load:.1f}s)\n")

    paths = collect(args.paths)
    wall0 = time.perf_counter()
    n = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: NOT FOUND\n"); continue
        n += 1
        name = os.path.basename(path)
        try:
            dur = sf.info(path).duration
        except Exception:
            dur = 0.0

        t0 = time.perf_counter()
        p_pres, p_abs = detect_murmur(m_sess, path)
        t_detect = time.perf_counter() - t0

        present = p_pres >= args.threshold
        verdict = "PRESENT" if present else "ABSENT"
        conf = p_pres if present else p_abs
        print(f"=== {name}  (~{dur:.1f}s) ===")
        print(f"  [1] Murmur detection : {verdict}   confidence {100*conf:.1f}%"
              f"   (P[Present]={p_pres:.3f}, P[Absent]={p_abs:.3f})")

        t_seg = 0.0
        if present:
            t1 = time.perf_counter()
            E, frac, n_cyc = segment_and_time(s_sess, crf, path)
            t_seg = time.perf_counter() - t1
            tot = E[1] + E[3] + 1e-12
            sys_pct, dia_pct = 100 * E[1] / tot, 100 * E[3] / tot
            timing = "SYSTOLIC" if E[1] >= E[3] else "DIASTOLIC"
            print(f"  [2] Segmentation     : {n_cyc} cardiac cycles  "
                  f"(S1 {100*frac[0]:.0f}%, Sys {100*frac[1]:.0f}%, S2 {100*frac[2]:.0f}%, Dia {100*frac[3]:.0f}%)")
            print(f"  [3] Timing           : Systolic {sys_pct:.1f}% / Diastolic {dia_pct:.1f}%"
                  f"  -> {timing}")
        else:
            print("      (no murmur -> segmentation/timing skipped)")

        t_total = t_detect + t_seg
        seg_part = f" | segmentation+timing {t_seg:.2f}s" if present else ""
        rt = f"  ({dur/t_total:.1f}x realtime)" if t_total > 0 and dur > 0 else ""
        print(f"  [runtime]            : detection {t_detect:.2f}s{seg_part} | total {t_total:.2f}s{rt}")
        print()

    wall = time.perf_counter() - wall0
    print(f"Processed {n} file(s) in {wall:.1f}s "
          f"(avg {wall/max(n,1):.2f}s/file, excludes one-time model load).")


if __name__ == "__main__":
    main()
