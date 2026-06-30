# PyTorch scripts

These are the original PyTorch scripts used to run, evaluate, fine-tune, and
export the models. They **depend on the two upstream model repositories**, which
they import. Clone those first and run the scripts from within the appropriate
upstream repo.

## Upstream repositories

- Murmur CRNN: <https://github.com/SiyuLou/AutomaticHeartSoundClassification>
- Segmenter (LSTM-CRF + FSST): <https://github.com/alvgaona/heart-sounds-segmentation>

Both are MIT-licensed; see `../THIRD_PARTY_NOTICES.md`.

## Contents

### Root

- `heart_sound_pipeline.py` - end-to-end PyTorch pipeline: murmur detection,
  segmentation, and timing localization. Run it from inside the upstream
  `heart-sounds-segmentation` repo so `hss.*` and `lightning_logs/...` resolve.

### `segmentation/`

Run these from inside the upstream `heart-sounds-segmentation` repo.

- `timing_localization_test.py` - systolic/diastolic timing test on CirCor.
- `eval_segmenter_on_circor.py` - frame-level segmentation eval vs CirCor expert
  `.tsv` labels.
- `eval_boundary_tolerance.py` - boundary-tolerance diagnostic for segmentation.
- `finetune_segmenter_on_circor.py` - fine-tune the segmenter on CirCor.
- `export_segmenter_onnx.py` - export the segmenter emission network to ONNX.
- `plot_timing.py`, `plot_progress.py` - figures.

### `murmur/`

Run these from inside the upstream `AutomaticHeartSoundClassification` repo.

- `export_murmur_onnx.py` - export the CirCor murmur model to ONNX.
- `predict_new_audio.py` - run new audio through the CirCor-trained murmur model.

## Feature extraction note

The segmentation models use the **FSST** transform. The upstream repo uses the
`ssq` package (Linux-only). For Windows/portable use, the ONNX pipeline uses
`ssqueezepy` instead (verified ~0.99 feature-equivalent). See the `onnx/` folder.
