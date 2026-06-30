"""Export the heart-sound segmenter emission network to ONNX.

The model is BiLSTM x2 -> Linear -> CRF. The LSTM/Linear part (the per-frame
"emission scores") exports cleanly; the CRF's Viterbi DECODE is loopy custom code
that does NOT export to ONNX, so we:
  - export the emission network  (input FSST (1, seq, 44) -> emissions (1, seq, 4))
  - save the CRF transition tables to .npz so Viterbi can be run in Python/C++
    on top of the ONNX emissions (or use plain argmax for a CRF-free decode).

Run in the WSL pixi env (needs the upstream `hss` package on PYTHONPATH):
  pixi run python pytorch/segmentation/export_segmenter_onnx.py
"""
import argparse
import glob
import os

import numpy as np
import torch

from hss.model.lit_model_crf import LitModelCRF

# Defaults resolved relative to THIS file -> this repo's models/ and onnx/.
DEFAULT_RELEASE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
DEFAULT_CKPT = os.path.join(DEFAULT_RELEASE, "models", "segmenter_finetuned_circor.pth")
DEFAULT_OUTDIR = os.path.join(DEFAULT_RELEASE, "onnx")


def default_base_ckpt():
    matches = sorted(glob.glob("lightning_logs/version_1/checkpoints/*.ckpt"))
    return matches[-1] if matches else None


def load_state_dict(path, cpu):
    ckpt = torch.load(path, map_location=cpu, weights_only=False)
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="fine-tuned .pth/.ckpt to export")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="directory for segmenter_emissions.onnx and CRF .npz")
    ap.add_argument("--base-checkpoint", default=default_base_ckpt(),
                    help="optional base fold checkpoint used only to initialize missing buffers")
    args = ap.parse_args()

    cpu = torch.device("cpu")
    model = LitModelCRF(input_size=44, batch_size=1, device=cpu)

    if args.base_checkpoint and os.path.exists(args.base_checkpoint):
        base_sd = {k: v for k, v in load_state_dict(args.base_checkpoint, cpu).items()
                   if not k.endswith(("h0", "c0"))}
        model.load_state_dict(base_sd, strict=False)

    sd = {k: v for k, v in load_state_dict(args.checkpoint, cpu).items()
          if not k.endswith(("h0", "c0"))}
    model.load_state_dict(sd, strict=False)
    seg = model.model.eval()                       # HeartSoundSegmenterCRF; forward -> emissions

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "segmenter_emissions.onnx")
    crf_out = os.path.join(args.outdir, "segmenter_crf_transitions.npz")

    dummy = torch.randn(1, 2000, 44)               # batch fixed at 1 (h0/c0), seq dynamic
    torch.onnx.export(
        seg, dummy, out,
        input_names=["fsst"], output_names=["emissions"],
        dynamic_axes={"fsst": {1: "seq"}, "emissions": {1: "seq"}},
        opset_version=17, dynamo=False,
    )

    # CRF transition tables -> for external Viterbi decode (labels: 0 S1,1 Sys,2 S2,3 Dia)
    crf = seg.crf
    np.savez(crf_out,
             start_transitions=crf.start_transitions.detach().numpy(),
             end_transitions=crf.end_transitions.detach().numpy(),
             transitions=crf.transitions.detach().numpy())

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(out))
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    o_onnx = sess.run(None, {"fsst": dummy.numpy()})[0]
    with torch.no_grad():
        o_torch = seg(dummy).numpy()
    print(f"emissions PyTorch vs ONNX max abs diff: {np.abs(o_onnx - o_torch).max():.2e}")
    d2 = torch.randn(1, 1234, 44)
    out2 = sess.run(None, {"fsst": d2.numpy()})[0]
    print(f"dynamic-seq check ok: input {tuple(d2.shape)} -> emissions {out2.shape}")
    print(f"checkpoint -> {args.checkpoint}")
    print(f"saved -> {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
    print(f"saved -> {crf_out}  (CRF tables for Viterbi)")


if __name__ == "__main__":
    main()
