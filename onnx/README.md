# ONNX-converted models

Two PyTorch models from this project, converted to ONNX (opset 17, classic
TorchScript exporter). Both were verified against the original PyTorch outputs.

## 1. `murmur_crnn_circor.onnx`  (15.2 MB)
CirCor-trained murmur detector (CNN + BiLSTM).

- Trained on the
  [CirCor DigiScope phonocardiogram dataset](https://physionet.org/content/circor-heart-sound/1.0.3/training_data/#files-panel).
- **Input** `log_mel` : float32 `(batch, 384, 333)`
  - 384 = 3 channels (log-mel + delta + delta2) x 128 mel-bins
  - 333 = frames in a 5 s window @ 2 kHz, hop 15 ms  (**fixed**; batch is dynamic)
- **Output** `logits` : float32 `(batch, 2)` — index 0 = Present, 1 = Absent
  - apply softmax for probabilities; P[Present] = softmax(logits)[:,0]
- Preprocessing (must match): resample audio -> 2 kHz, bandpass 25-400 Hz,
  log-mel(128, hop 15 ms, win 25 ms), z-score, stack +delta+delta2.

## 2. `segmenter_emissions.onnx`  (7.8 MB)  + `segmenter_crf_transitions.npz`
Heart-sound segmenter (BiLSTM x2 -> Linear), fine-tuned on CirCor segmentation
labels. **Emission network only.**

- First trained on the
  [Springer heart-sound segmentation dataset](https://pub-db0cd070a4f94dabb9b58161850d4868.r2.dev/heart-sounds/springer_sounds.zip),
  then fine-tuned on CirCor expert segmentation labels.
- **Input** `fsst` : float32 `(1, seq, 44)`  (batch fixed at 1, seq dynamic)
  - 44 = FSST features (25-200 Hz, real+imag stacked) per 1 kHz sample
- **Output** `emissions` : float32 `(1, seq, 4)` — per-frame scores for
  0=S1, 1=Systole, 2=S2, 3=Diastole

### Important: the CRF decoder is NOT in the ONNX graph
The model's CRF layer does a Viterbi decode (loop-based) that cannot be exported
to ONNX. Two ways to get final labels from `emissions`:
- **Quick:** `labels = emissions.argmax(-1)` (ignores legal-transition constraints).
- **Faithful:** run a Viterbi decode in Python/C++ using the saved transition
  tables in `segmenter_crf_transitions.npz`
  (`start_transitions` [4], `end_transitions` [4], `transitions` [4x4]).

## `heart_sound_pipeline_onnx.py`  (ready-to-run, no PyTorch)
End-to-end pipeline using BOTH ONNX models via onnxruntime:
  1. murmur detection (Present/Absent + confidence)
  2. if Present: segmentation (emissions ONNX + Viterbi using the CRF tables)
  3. timing: murmur-band energy pooled by phase -> Systolic / Diastolic %

```
pip install onnxruntime numpy scipy soundfile librosa ssqueezepy
python heart_sound_pipeline_onnx.py path/to/audio.wav          # one file
python heart_sound_pipeline_onnx.py path/to/folder             # whole folder
python heart_sound_pipeline_onnx.py --threshold 0.5 a.wav
```
Feature extraction still happens in-script (ONNX only covers the neural nets):
log-mel (librosa) for the murmur model, FSST (ssqueezepy, modulated=False) for the
segmenter. Verified to reproduce the PyTorch pipeline's outputs.

## Performance

Murmur classifier on held-out CirCor patients (`n=133`, threshold `0.5`):

| True \ Pred | Present | Absent |
|---|---:|---:|
| Present | 25 | 3 |
| Absent | 1 | 104 |

| Accuracy | Sensitivity | Specificity | Balanced accuracy | F1 |
|---:|---:|---:|---:|---:|
| 96.99% | 89.29% | 99.05% | 94.17% | 92.59% |

CirCor-finetuned segmenter on held-out CirCor recordings (`n=120`, frame-level):

| True \ Pred | S1 | Systole | S2 | Diastole |
|---|---:|---:|---:|---:|
| S1 | 265375 | 15592 | 2604 | 22670 |
| Systole | 20602 | 291446 | 21385 | 9844 |
| S2 | 6482 | 17075 | 220146 | 25150 |
| Diastole | 20397 | 7851 | 18344 | 583367 |

| Class | Sensitivity / recall | Specificity (one-vs-rest) | Precision | F1 |
|---|---:|---:|---:|---:|
| S1 | 86.66% | 96.18% | 84.82% | 85.73% |
| Systole | 84.90% | 96.64% | 87.79% | 86.32% |
| S2 | 81.88% | 96.69% | 83.87% | 82.87% |
| Diastole | 92.60% | 93.72% | 91.00% | 91.80% |

Overall frame accuracy: `87.86%`; balanced frame accuracy: `86.51%`.

## Notes
- Conversion env: WSL pixi (torch 2.5.1), exporter `dynamo=False`.
- `segmenter_emissions.onnx` was exported from
  `models/segmenter_finetuned_circor.pth`; ONNX vs PyTorch emissions matched
  with max absolute difference `1.85e-06`, and dynamic sequence input was checked.
- Run a single model with onnxruntime: `ort.InferenceSession(path).run(None, {"<input>": arr})`.
