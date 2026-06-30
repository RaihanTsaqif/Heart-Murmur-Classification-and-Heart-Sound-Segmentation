# PyTorch scripts

These are the original PyTorch scripts used to run, evaluate, fine-tune, and
export the models. They **depend on the two upstream model repositories**, which
provide the model class definitions and helper functions.

## Upstream repositories

- Murmur CRNN: <https://github.com/SiyuLou/AutomaticHeartSoundClassification>
- Segmenter (LSTM-CRF + FSST): <https://github.com/alvgaona/heart-sounds-segmentation>

Both are MIT-licensed; see `../THIRD_PARTY_NOTICES.md`.

## Contents

### Root

- `heart_sound_pipeline.py` - end-to-end PyTorch pipeline: murmur detection,
  segmentation, and timing localization.

Example:

```bash
git clone https://github.com/RaihanTsaqif/Heart-Murmur-Classification-and-Heart-Sound-Segmentation.git
git clone https://github.com/alvgaona/heart-sounds-segmentation.git
git clone https://github.com/SiyuLou/AutomaticHeartSoundClassification.git

cd Heart-Murmur-Classification-and-Heart-Sound-Segmentation
python pytorch/heart_sound_pipeline.py path/to/audio.wav \
  --seg-repo ../heart-sounds-segmentation \
  --murmur-repo ../AutomaticHeartSoundClassification
```

The script is run from this repository. `--seg-repo` points to the cloned
segmentation repo so Python can import `hss.*`. `--murmur-repo` points to the
cloned murmur repo so Python can import `model.*` and `utils.*`. By default, the
trained weights are loaded from this repository's `models/` folder.

### `segmentation/`

These still mirror the original segmentation research workflow. Run them from
inside the upstream `heart-sounds-segmentation` repo unless the script says
otherwise.

- `timing_localization_test.py` - systolic/diastolic timing test on CirCor.
- `eval_segmenter_on_circor.py` - frame-level segmentation eval vs CirCor expert
  `.tsv` labels.
- `eval_boundary_tolerance.py` - boundary-tolerance diagnostic for segmentation.
- `finetune_segmenter_on_circor.py` - fine-tune the segmenter on CirCor.
- `export_segmenter_onnx.py` - export the segmenter emission network to ONNX.
- `plot_timing.py`, `plot_progress.py` - figures.

### `murmur/`

These still mirror the original murmur-model workflow. Run them from inside the
upstream `AutomaticHeartSoundClassification` repo unless the script says
otherwise.

- `export_murmur_onnx.py` - export the CirCor murmur model to ONNX.
- `predict_new_audio.py` - run new audio through the CirCor-trained murmur model.

## Feature extraction note

The segmentation models use the **FSST** transform. The upstream repo uses the
`ssq` package (Linux-only). For Windows/portable use, the ONNX pipeline uses
`ssqueezepy` instead (verified ~0.99 feature-equivalent). See the `onnx/` folder.
