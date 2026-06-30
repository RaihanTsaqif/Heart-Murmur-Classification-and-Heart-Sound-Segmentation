# Trained models

All weights here are derived from the two MIT-licensed upstream architectures
(see `../THIRD_PARTY_NOTICES.md`). Training was done on public datasets (not
included; see notices).

## Training data provenance

- The murmur classifier weights were trained on the
  [CirCor DigiScope phonocardiogram dataset](https://physionet.org/content/circor-heart-sound/1.0.3/training_data/#files-panel).
- The segmentation weights were first trained on the
  [Springer heart-sound segmentation dataset](https://pub-db0cd070a4f94dabb9b58161850d4868.r2.dev/heart-sounds/springer_sounds.zip),
  then fine-tuned on CirCor expert segmentation labels.

| File | Architecture | Trained on | Notes |
|---|---|---|---|
| `murmur_crnn_circor.pth` | CRNN (SiyuLou) | CirCor murmur Present/Absent | main murmur model; ~94% MAcc on held-out CirCor test |
| `segmenter_finetuned_circor.pth` | BiLSTM-CRF (Gaona) | DavidSpringer → fine-tuned on CirCor `.tsv` | best on CirCor (~87% overall / ~86% balanced, held-out patients) |

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

## Loading

- **Murmur** (`crnn` arch in `../configs/config_crnn.json`): index 0 = Present, 1 = Absent.
  Input = log-mel (384 × 333).
- **Segmenter** (BiLSTM-CRF, input_size=44 FSST features): output classes
  0=S1, 1=Systole, 2=S2, 3=Diastole. Loading these requires the upstream
  segmentation repo's model code; drop the `h0`/`c0` buffers when changing batch size.

For a runtime that needs **no PyTorch and no upstream repos**, use the ONNX
versions in `../onnx/`. The ONNX segmenter is exported from
`segmenter_finetuned_circor.pth`.
