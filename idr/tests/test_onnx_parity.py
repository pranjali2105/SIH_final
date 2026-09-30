"""CI: PyTorch checkpoint vs exported ONNX (ORT_ENABLE_BASIC) must agree.

    python -m pytest idr/tests/test_onnx_parity.py -q

Fails if the ONNX file is stale w.r.t. the checkpoint, if an export-time rewrite changed the maths, or if the
startup self-check (golden window) no longer passes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from idr.normalise import ARTIFACTS
from idr.onnx_runtime import EXPORT_DIR, GOLDEN_TOL, OUTPUTS, golden_check, make_session

TAG = "tft_ep2"
ONNX = EXPORT_DIR / f"{TAG}.onnx"
GOLDEN = EXPORT_DIR / f"{TAG}_golden.npz"
CKPT = ARTIFACTS / f"{TAG}.pt"

pytestmark = pytest.mark.skipif(not (ONNX.exists() and CKPT.exists()), reason="export or checkpoint missing")


@pytest.fixture(scope="module")
def torch_model():
    from idr.export import Wrapper
    from idr.model import ResidualTFT
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = ResidualTFT(d=ck["args"]["d"], use_tft=not ck["args"]["no_tft"]).eval()
    m.load_state_dict(ck["model"], strict=False)
    return Wrapper(m).eval()


@pytest.fixture(scope="module")
def sess():
    return make_session(ONNX, check_golden=False)


def _inputs(seed, B=4):
    from idr.dataset import CTX, HORIZON, N_SEQ, N_STATIC
    from idr.route_features import N_PTS
    g = np.random.default_rng(seed)
    seq = g.normal(0, 1, (B, CTX + HORIZON, N_SEQ)).astype(np.float32)
    seq[:, CTX:, :] *= (g.random((B, 1, 1)) > 0.5)          # some windows with zero-padded future
    static = g.normal(0, 0.5, (B, N_STATIC)).astype(np.float32)
    static[:, -1] = g.integers(0, 2, B)
    route = g.normal(0, 1, (B, N_PTS, 4)).astype(np.float32)
    route[..., 3] = (np.arange(N_PTS)[None] < g.integers(0, N_PTS, B)[:, None])
    u_raw = np.column_stack([g.normal(0, 0.2, (B, HORIZON)), g.normal(0, 1.0, (B, HORIZON)),
                             g.normal(0, 1.0, (B, HORIZON))]).reshape(B, 3, HORIZON).transpose(0, 2, 1)
    return {"seq": seq, "static": static, "route": route, "u_raw": np.ascontiguousarray(u_raw, np.float32)}


def _compare(torch_model, sess, x, tol):
    with torch.no_grad():
        t_out = torch_model(*(torch.from_numpy(x[k]) for k in ("seq", "static", "route", "u_raw")))
    o_out = sess.run(None, x)
    for n, t, o in zip(OUTPUTS, t_out, o_out):
        diff = float(np.max(np.abs(t.numpy() - o)))
        assert diff < tol, f"{n}: max |torch - onnx| = {diff:.3g}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_inputs(torch_model, sess, seed):
    _compare(torch_model, sess, _inputs(seed), 1e-4)


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden window missing (re-run idr.export)")
def test_golden_real_windows(torch_model, sess):
    g = np.load(GOLDEN)
    _compare(torch_model, sess, {k: g[k] for k in ("seq", "static", "route", "u_raw")}, 1e-4)
    assert golden_check(sess, GOLDEN) < GOLDEN_TOL


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden window missing (re-run idr.export)")
def test_startup_check_passes():
    make_session(ONNX, check_golden=True)


def test_batch_independence(sess):
    """Padding the batch must not change a sample's outputs."""
    x = _inputs(5, B=3)
    one = sess.run(None, {k: v[:1] for k, v in x.items()})
    full = sess.run(None, x)
    for a, b in zip(one, full):
        assert np.max(np.abs(a - b[:1])) < 1e-5
