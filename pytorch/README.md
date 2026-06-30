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

Research / eval scripts. They import the upstream segmenter's `hss` package (run
in its WSL pixi env, which provides the Linux-only FSST transform) and read the
**CirCor dataset** from `$CIRCOR_DIR` (PhysioNet download; not bundled). The
segmenter checkpoint defaults to this repo's
`models/segmenter_finetuned_circor.pth` (override with `$SEG_CKPT`). Example:

```bash
CIRCOR_DIR=/path/to/the-circor-...-1.0.3 \
  pixi run python pytorch/segmentation/eval_segmenter_on_circor.py
```

- `timing_localization_test.py` - systolic/diastolic timing test on CirCor.
- `eval_segmenter_on_circor.py` - frame-level segmentation eval vs CirCor expert
  `.tsv` labels.
- `eval_boundary_tolerance.py` - boundary-tolerance diagnostic for segmentation.
- `finetune_segmenter_on_circor.py` - fine-tune the segmenter on CirCor.
- `export_segmenter_onnx.py` - export the segmenter emission network to ONNX
  (writes to `onnx/` by default).
- `plot_timing.py`, `plot_progress.py` - figures.

### `murmur/`

Run these from anywhere; the murmur config/weights are resolved relative to this
repo (`configs/`, `models/`). They import `model`/`utils` from the upstream murmur
repo, so pass a clone of it with `--murmur-repo` (or set `$MURMUR_REPO`, or drop
the clone next to this repo as `../AutomaticHeartSoundClassification`).

- `predict_new_audio.py` - run new audio through the CirCor murmur model:

  ```bash
  python pytorch/murmur/predict_new_audio.py \
    --murmur-repo ../AutomaticHeartSoundClassification recording.wav
  ```

- `export_murmur_onnx.py` - re-export the CirCor murmur model to ONNX
  (`--murmur-repo ...`; writes to `onnx/` by default; needs the `onnx` package).

## Feature extraction note

The segmentation models use the **FSST** transform. The upstream repo uses the
`ssq` package (Linux-only). For Windows/portable use, the ONNX pipeline uses
`ssqueezepy` instead (verified ~0.99 feature-equivalent). See the `onnx/` folder.
