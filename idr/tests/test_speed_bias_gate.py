"""Pre-outage speed-bias gate for the look-ahead prior. `python -m pytest idr/tests/test_speed_bias_gate.py -q`"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from idr.causal_calibration import CalibrationConfig
from idr.dataset import HORIZON, load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG
from idr.map_lookahead import LookaheadConfig, LookaheadPrior
from idr.map_matching import run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.route_input import RouteConfig
from idr.speed_bias_gate import recent_speed_bias

sys.path.insert(0, str(Path(__file__).parent))
from test_map_lookahead import PSEUDO, zigzag  # noqa: E402

CAL = CalibrationConfig()
GATE = {**PSEUDO, "enable_post_event_recovery": True, "enable_speed_bias_gate": True}


@pytest.fixture(scope="module")
def s2():
    drives, _ = load_drives(["S2"])
    specs = make_specs(drives, ["S2"], HORIZON, stride=600)[5:8]
    ck = torch.load(ARTIFACTS / "tft_ep2.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().eval()
    m.load_state_dict(ck["model"], strict=False)
    return drives, specs, m


def bias_of(drives, specs, m):
    return recent_speed_bias(m, drives, specs, torch.device("cpu"), CAL, RouteConfig(), NavConfig(**TFT_CFG), 30.0)


# (a) the estimate uses only GNSS strictly before outage start - guard
def test_a_bias_uses_only_pre_outage_gnss(s2):
    drives, specs, m = s2
    out = bias_of(drives, specs, m)
    assert all(o["available"] for o in out)
    for o, sp in zip(out, specs):
        assert o["window"][1] < sp["t0"] - CAL.guard_steps and o["n_samples"] == 300
        assert o["outage_start_index"] == sp["t0"]


# (b) poisoning every GNSS sample from the guard interval onwards cannot change the estimate
def test_b_gnss_during_outage_cannot_affect_bias(s2):
    drives, specs, m = s2
    for sp in specs:
        r = bias_of(drives, [sp], m)[0]                     # same batch size as the poisoned run (bit-exact)
        d, t0 = drives["S2"], sp["t0"] - CAL.guard_steps
        v, psi, pos = d.v.copy(), d.psi.copy(), d.pos.copy()
        dest = pos[-1].copy()
        v[t0:], psi[t0:], pos[t0:] = np.nan, np.nan, np.nan
        pos[-1] = dest                                     # user-entered destination: external input
        dp = {"S2": dataclasses.replace(d, v=v, psi=psi, pos=pos, grid_b=np.full_like(d.grid_b, np.nan))}
        dp["S2"].__dict__.pop("_starts_set", None)
        b = bias_of(dp, [sp], m)[0]
        for k in ("recent_ekf_speed_bias_mps", "recent_gnss_speed_mean_mps", "recent_ekf_speed_mean_mps", "window"):
            assert b[k] == r[k], k


def run(bias, extra=None, seed=0):
    rm, Pt, yaw, v = zigzag()
    info = {"available": True, "recent_ekf_speed_bias_mps": bias, "recent_gnss_speed_mean_mps": 14.0,
            "recent_ekf_speed_mean_mps": 14.0 + bias, "n_samples": 300}
    return run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(len(yaw)), seed=seed, return_trace=True,
                      lookahead={**GATE, **(extra or {})}, speed_bias_info=info)


# (c) a sufficiently negative pre-outage bias suppresses the prior
def test_c_negative_bias_suppresses():
    _, _, st, trc = run(-0.8)
    la = st["lookahead"]
    assert la["n_applications"] == 0 and la["total_speed_reduction_mps"] == 0.0
    assert la["bias_gate"]["suppressed_steps_by_reason"].get("bias_gate", 0) > 0
    assert all(x["delta_v_mps"] == 0.0 for x in trc["lookahead_trace"])


# (d) neutral or positive bias allows the prior
@pytest.mark.parametrize("b", [0.0, 0.4])
def test_d_neutral_or_positive_bias_allows(b):
    _, _, st, _ = run(b)
    assert st["lookahead"]["n_applications"] > 0


# (e) branch-proximity guard: negative-but-allowed bias is blocked only when a branch is close
def test_e_branch_guard():
    cfg = LookaheadConfig(enable_map_lookahead_speed_prior=True, enable_speed_bias_gate=True)
    pr = LookaheadPrior(cfg, {"available": True, "recent_ekf_speed_bias_mps": -0.1})
    pr.next_branch_m = 20.0
    assert pr.bias_gate() == (False, True, "branch_guard")
    pr.next_branch_m = 50.0
    assert pr.bias_gate()[0] is True
    pr.next_branch_m = None
    assert pr.bias_gate()[0] is True
    pr2 = LookaheadPrior(cfg, {"available": True, "recent_ekf_speed_bias_mps": 0.05})
    pr2.next_branch_m = 20.0
    assert pr2.bias_gate() == (True, True, None)
    pr3 = LookaheadPrior(cfg, {"available": False})
    assert pr3.bias_gate() == (False, False, "bias_unknown")
    assert LookaheadPrior(LookaheadConfig(enable_speed_bias_gate=True, bias_unknown_policy="allow"),
                          {"available": False}).bias_gate()[0] is True


# (f) gate disabled -> bit-identical to the ungated prior; gate enabled with a positive bias and no branches ->
#     identical too (the gate only ever removes negative updates)
def test_f_disabled_is_bit_identical():
    rm, Pt, yaw, v = zigzag()
    T = len(yaw)
    base = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=4,
                      lookahead={**PSEUDO, "enable_post_event_recovery": True})
    off = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=4,
                     lookahead={**PSEUDO, "enable_post_event_recovery": True, "enable_speed_bias_gate": False},
                     speed_bias_info={"available": True, "recent_ekf_speed_bias_mps": -5.0})
    np.testing.assert_array_equal(base[0], off[0])
    pos = run(0.5, seed=4)
    np.testing.assert_array_equal(base[0], pos[0])
    ctrl = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=4)
    supp = run(-0.8, seed=4)                                       # fully suppressed == control
    np.testing.assert_array_equal(ctrl[0], supp[0])


# (g) gate metadata and suppression reasons are recorded per potential event
def test_g_metadata_recorded():
    _, _, st, _ = run(-0.8)
    la = st["lookahead"]
    assert la["bias_gate"]["enabled"] and la["bias_gate"]["recent_ekf_speed_bias_mps"] == -0.8
    ev = la["events"]
    assert ev
    for e in ev:
        for k in ("recent_ekf_speed_bias_mps", "recent_gnss_speed_mean_mps", "recent_ekf_speed_mean_mps",
                  "bias_gate_passed", "branch_guard_active", "prior_applied", "prior_suppressed_reason"):
            assert k in e, k
    assert any(e["prior_suppressed_reason"] == "bias_gate" and e["prior_applied"] is False for e in ev)
    _, _, st2, _ = run(0.4)
    assert any(e["prior_applied"] and e["bias_gate_passed"] for e in st2["lookahead"]["events"])
