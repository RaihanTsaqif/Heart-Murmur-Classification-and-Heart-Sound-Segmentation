"""Export the CirCor-trained murmur CRNN (Present/Absent) to ONNX.

Input  : log-mel feature  (batch, 384, frames)   384 = 3 channels x 128 mel-bins
Output : logits           (batch, 2)              index 0 = Present, 1 = Absent
Run from the repo root (so `model` imports):  python export_murmur_onnx.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.model as module_arch


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

CFG = "config/config_crnn.json"
WEIGHTS = ("saved/training with circor murmur using authors architecture and "
           "oversampling/best_model.pth")
_BASE = ("/mnt/c/Projects/Heart Sound AI Classifier/model converted using ONNX"
         if os.path.isdir("/mnt/c") else
         r"C:\Projects\Heart Sound AI Classifier\model converted using ONNX")
OUT = os.path.join(_BASE, "murmur_crnn_circor.onnx")


def main():
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
