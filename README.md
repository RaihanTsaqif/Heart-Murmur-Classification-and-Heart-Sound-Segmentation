# Heart Murmur Classification and Heart Sound Segmentation

Deep-learning pipeline for heart-sound recordings (phonocardiograms / PCG): a
**CRNN murmur classifier** (Present / Absent) and an **LSTM-CRF segmenter** that
labels each moment of the cardiac cycle as **S1 / Systole / S2 / Diastole**, then
pools murmur-band energy per phase to localize **systolic vs diastolic** timing.
Provided in both **PyTorch** and **ONNX** runtimes.

## What it does

For an input `.wav` heart-sound recording:

1. **Murmur detection**: CRNN (log-mel + CNN-BiLSTM) → *murmur present / absent* + confidence.
2. **Segmentation** *(if a murmur is present)*: LSTM-CRF (FSST features) → per-frame
   S1 / Systole / S2 / Diastole, decoded with a CRF Viterbi pass.
3. **Timing**: murmur-band energy pooled per phase → *systolic vs diastolic* split.

## Quick start using ONNX

```bash
pip install -r requirements-onnx.txt
python onnx/heart_sound_pipeline_onnx.py path/to/recording.wav
```

## PyTorch version

The scripts in `pytorch/` reproduce training/evaluation/fine-tuning and the ONNX
export. They **import the two upstream model repositories** for the model class
definitions, so clone those and point the scripts at them (e.g. `--murmur-repo`,
`--seg-repo`, or `$MURMUR_REPO`). You run the scripts from this repo, not from
inside the upstream clones. See `pytorch/README.md`.

## Models

| File | Model |
|---|---|
| `models/murmur_crnn_circor.pth` | murmur CRNN, trained on CirCor |
| `models/segmenter_finetuned_circor.pth` | above, fine-tuned on CirCor (best on CirCor) |
| `onnx/murmur_crnn_circor.onnx` | murmur model, ONNX |
| `onnx/segmenter_emissions.onnx` + `segmenter_crf_transitions.npz` | CirCor-finetuned segmenter, ONNX emission network (+ CRF tables for Viterbi) |

## Training data provenance

- The murmur classifier was trained on the
  [CirCor DigiScope phonocardiogram dataset](https://physionet.org/content/circor-heart-sound/1.0.3/training_data/#files-panel).
- The segmentation model was first trained on the
  [Springer heart-sound segmentation dataset](https://pub-db0cd070a4f94dabb9b58161850d4868.r2.dev/heart-sounds/springer_sounds.zip),
  then fine-tuned on CirCor expert segmentation labels.

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

## Attribution / license

This project is released under the **MIT License** (see [LICENSE](LICENSE)).

It builds on two MIT-licensed projects — the murmur CRNN by **SiyuLou** and the
heart-sound segmenter by **Alvaro Gaona**. Their architectures and original
copyright/license notices are attributed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), as required by the MIT License.
Datasets (CirCor and Springer/DavidSpringer) are governed by their own terms
and are not included here.
