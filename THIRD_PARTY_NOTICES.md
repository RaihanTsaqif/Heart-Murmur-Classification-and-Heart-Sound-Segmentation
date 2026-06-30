# Third-Party Notices and Attributions

This project builds on two open-source projects, both released under the **MIT
License**. The neural-network **architectures** (and parts of the training /
evaluation code) for the two models in this repository are derived from them.
The trained weights and ONNX exports in this repository are derived works of
those architectures. Full license texts are reproduced below, as required by the
MIT License.

---

## 1. Murmur classification model — CRNN

- **Used for:** the heart-murmur (Present/Absent) CRNN architecture and the
  training/evaluation framework it is based on.
- **Files in this repo derived from it:** `models/murmur_crnn_circor.pth`,
  `onnx/murmur_crnn_circor.onnx`, and the scripts in `pytorch/murmur/`
  (which import that project's model code).
- **Original project:** SiyuLou / AutomaticHeartSoundClassification
  <https://github.com/SiyuLou/AutomaticHeartSoundClassification>
- **Note:** that project is itself based on the `pytorch-template` by victoresque
  (<https://github.com/victoresque/pytorch-template>, MIT License).

```
MIT License

Copyright (c) 2022 SiyuLou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Heart-sound segmentation model — LSTM + CRF

- **Used for:** the S1/Systole/S2/Diastole segmentation architecture (BiLSTM +
  CRF), the FSST feature transform, and the training/evaluation framework.
- **Files in this repo derived from it:** `models/segmenter_finetuned_circor.pth`,
  `onnx/segmenter_emissions.onnx`,
  `onnx/segmenter_crf_transitions.npz`, and the scripts in `pytorch/segmentation/`
  (which import that project's model code).
- **Original project:** alvgaona / heart-sounds-segmentation
  <https://github.com/alvgaona/heart-sounds-segmentation>

```
MIT License

Copyright (c) 2019 Alvaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Datasets (acknowledgement)

The models were trained / evaluated on the following public datasets, each
governed by its own data-use terms (not the MIT License above):

- **CirCor DigiScope Phonocardiogram Dataset** — Oliveira et al., via PhysioNet.
  <https://physionet.org/content/circor-heart-sound/>
- **David Springer heart-sound segmentation data** — used by the segmentation
  project above for training the segmenter.

Datasets are **not** included in this repository; download them from the sources
above under their respective licenses.

## Other dependencies

This project also uses standard open-source Python packages (PyTorch,
onnxruntime, NumPy, SciPy, librosa, ssqueezepy/ssq, scikit-learn, matplotlib,
soundfile), each distributed under its own license (MIT / BSD /
Apache-2.0). See each package's distribution for details.
