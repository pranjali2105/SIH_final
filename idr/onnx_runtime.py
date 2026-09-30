"""Reference ONNX Runtime loader for the exported model (the app must mirror this).

onnxruntime's EXTENDED / ALL graph optimisations (ALL is the default) compute this model WRONGLY (operator-fusion
bug, onnxruntime 1.29: |diff| up to 0.6 on real windows). BASIC matches PyTorch to ~1e-6. So:
  1. the session is created with ORT_ENABLE_BASIC, and that is asserted at startup;
  2. a golden window (inputs + PyTorch outputs, written by idr.export) is run once at startup and must match.

    sess = make_session("idr/export/tft_ep2.onnx")          # raises if either check fails
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXPORT_DIR = Path(__file__).resolve().parent / "export"
OUTPUTS = ["u_hat", "p_stop", "e_hat", "b_hat"]
GOLDEN_TOL = 1e-4


def make_session(onnx_path, golden_path=None, check_golden=True):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, \
        "tft ONNX model must run with ORT_ENABLE_BASIC (EXTENDED/ALL fusions give wrong outputs)"
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    if check_golden:
        golden_path = Path(golden_path) if golden_path else Path(onnx_path).with_name(Path(onnx_path).stem + "_golden.npz")
        worst = golden_check(sess, golden_path)
        assert worst < GOLDEN_TOL, f"ONNX startup self-check failed: max |onnx - torch| = {worst:.3g} (> {GOLDEN_TOL})"
    return sess


def golden_check(sess, golden_path):
    g = np.load(golden_path)
    out = sess.run(None, {k: g[k] for k in ("seq", "static", "route", "u_raw")})
    return float(max(np.max(np.abs(o - g[f"out_{n}"])) for n, o in zip(OUTPUTS, out)))


def write_golden(path, inputs: dict, outputs: dict):
    np.savez_compressed(path, **{k: np.asarray(v, np.float32) for k, v in inputs.items()},
                        **{f"out_{k}": np.asarray(v, np.float32) for k, v in outputs.items()})
