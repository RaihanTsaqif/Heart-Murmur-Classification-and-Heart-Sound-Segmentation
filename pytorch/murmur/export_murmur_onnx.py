"""Export the CirCor-trained murmur CRNN (Present/Absent) to ONNX.

Input  : log-mel feature  (batch, 384, frames)   384 = 3 channels x 128 mel-bins
Output : logits           (batch, 2)              index 0 = Present, 1 = Absent

Usage (runs from any CWD; config/weights are resolved relative to THIS file):
  # `model` comes from the upstream murmur repo (SiyuLou); pass a clone of it.
  python pytorch/murmur/export_murmur_onnx.py --murmur-repo /path/to/AutomaticHeartSoundClassification
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths are anchored to THIS file, so the script runs from any CWD. ---------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))  # has configs/ models/ onnx/

module_arch = None  # bound from the upstream murmur repo in _load_upstream()


def _load_upstream(murmur_repo):
    """Add the upstream murmur repo to sys.path and import its `model` package."""
    global module_arch
    murmur_repo = os.path.abspath(murmur_repo)
    if not os.path.isdir(os.path.join(murmur_repo, "model")):
        raise SystemExit(
            f"murmur repo not found at: {murmur_repo}\n"
            "Clone SiyuLou/AutomaticHeartSoundClassification and pass it with "
            "--murmur-repo PATH (or set $MURMUR_REPO).")
    sys.path.insert(0, murmur_repo)
    import model.model as _ma
    module_arch = _ma


class ExportCRNN(nn.Module):
    """Wraps `crnn` so the mel-pooling kernel is a CONSTANT (mel dim after the CNN
    is fixed = 8 for 128-bin input). The original uses kernel=(x.size(-2),1), whose
    shape-derived value the classic ONNX exporter can't make static."""

    def __init__(self, crnn, mel_after):
        super().__init__()
        self.c = crnn
        self.mel_after = int(mel_after)

    def forward(self, x):
        c = self.c
        B, mel_bins, num_frames = x.size()
        x = x.view(B, c.in_channel, -1, num_frames).transpose(1, 2)
        x = c.bn0(x).transpose(1, 2)
        x = c.cnn(x)
        x = F.max_pool2d(x, kernel_size=(self.mel_after, 1))   # constant kernel
        x = x.squeeze(-2)
        return c.bilstm(x)

CFG = os.path.join(_REPO_ROOT, "configs", "config_crnn.json")
WEIGHTS = os.path.join(_REPO_ROOT, "models", "murmur_crnn_circor.pth")
DEFAULT_OUT = os.path.join(_REPO_ROOT, "onnx", "murmur_crnn_circor.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--murmur-repo", default=os.environ.get("MURMUR_REPO"),
                    help="clone of SiyuLou/AutomaticHeartSoundClassification (provides "
                         "the `model` package). Defaults to $MURMUR_REPO.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .onnx path")
    args = ap.parse_args()
    if not args.murmur_repo:
        raise SystemExit("Need --murmur-repo PATH (clone of "
                         "SiyuLou/AutomaticHeartSoundClassification) or set $MURMUR_REPO.")
    _load_upstream(args.murmur_repo)
    OUT = args.out
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

    arch = json.load(open(CFG))["arch"]
    net = getattr(module_arch, arch["type"])(**arch["args"])
    ckpt = torch.load(WEIGHTS, weights_only=False, map_location="cpu")
    net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    net.eval()

    dummy = torch.randn(1, 384, 333)            # 1 window: 384 feat x 333 frames (5 s)

    # determine the (constant) mel dim after the CNN, then wrap with a static kernel
    with torch.no_grad():
        t = dummy.view(1, net.in_channel, -1, 333).transpose(1, 2)
        t = net.bn0(t).transpose(1, 2)
        mel_after = net.cnn(t).shape[-2]
    export_net = ExportCRNN(net, mel_after).eval()

    torch.onnx.export(
        export_net, dummy, OUT,
        input_names=["log_mel"], output_names=["logits"],
        dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}},  # frames fixed at 333 (5 s)
        opset_version=17, dynamo=False,
    )

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(OUT))
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    o_onnx = sess.run(None, {"log_mel": dummy.numpy()})[0]
    with torch.no_grad():
        o_torch = net(dummy).numpy()
    print(f"PyTorch vs ONNX max abs diff: {np.abs(o_onnx - o_torch).max():.2e}")
    # sanity: dynamic batch (frames fixed at 333)
    d2 = torch.randn(4, 384, 333)
    out2 = sess.run(None, {"log_mel": d2.numpy()})[0]
    print(f"dynamic-batch check ok: input {tuple(d2.shape)} -> output {out2.shape}")
    print(f"saved -> {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
