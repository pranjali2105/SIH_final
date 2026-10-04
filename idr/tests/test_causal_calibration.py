"""Causal per-outage calibration: leakage regression tests. `python -m pytest idr/tests/test_causal_calibration.py -q`

Uses real IO-VNBD drive S2 (validation). The "external plan" (user destination and the dataset's route-feature
source d.s / d.grid_b) is held fixed in the poison test: it is not calibration, and its oracle nature is a separate,
documented issue (see idr/causal_calibration.py and the task report).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from idr.causal_calibration import GNSS_COLS, CalibrationConfig, build_views, outage_calibration, segment_of
from idr.data import DT, load_all
from idr.dataset import CTX, HORIZON, OutageDataset, load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG, model_outputs
from idr.map_matching import _synthetic_map, run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS, _segments
from idr.osm_map import drive_roadmap

from idr.route_input import RouteConfig

ORACLE_ROUTE = RouteConfig("oracle_future_gnss", allow_oracle_diagnostic=True)
CAUSAL = CalibrationConfig()
ORACLE = CalibrationConfig(mode="offline_oracle")
REF = Path(__file__).parent / "data" / "oracle_reference.npz"
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def env():
    raw = load_all()
    drives, _ = load_drives(["S2", "Vtb5"])
    specs = make_specs(drives, ["S2"], HORIZON, stride=600)
    sp = specs[5]                                  # long prefix (full calibration)
    ck = torch.load(ARTIFACTS / "tft_ep2.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().eval()
    m.load_state_dict(ck["model"], strict=False)
    return {"raw": raw, "drives": drives, "specs": specs, "sp": sp, "model": m}


def _cal_json(cal, meta):
    return json.dumps(cal, sort_keys=True), json.dumps(meta, sort_keys=True, default=str)


# (a) GNSS inside the outage (and inside the guard) cannot change the causal calibration
def test_a_gnss_inside_outage_does_not_affect_causal_calibration(env):
    df, t0 = env["raw"]["S2"], env["sp"]["t0"]
    cal0, meta0 = outage_calibration(df, t0, CAUSAL)
    assert meta0["calibration_quality"] == "full"
    inj = df.copy()
    rows = inj.index[t0 - CAUSAL.guard_steps:]
    inj.loc[rows, "v_true"] = 55.5                  # distinctive values from the guard onwards
    inj.loc[rows, "yaw_rate_true"] = 3.3
    inj.loc[rows, "psi_true"] = 1.234
    cal1, meta1 = outage_calibration(inj, t0, CAUSAL)
    assert _cal_json(cal0, meta0) == _cal_json(cal1, meta1)
    assert meta0["calibration_end_time_s"] < t0 * DT - CAUSAL.guard_s
    # the oracle calibration DOES see it (proves the injection is real)
    assert _cal_json(*outage_calibration(df, t0, ORACLE))[0] != _cal_json(*outage_calibration(inj, t0, ORACLE))[0]


# (b) poison every GNSS sample at/after outage start: causal calibration, model inputs, model outputs and the
#     EKF + map track are unchanged
def test_b_poisoned_future_gnss_does_not_affect_causal_output(env):
    raw, drives, sp, m = env["raw"], env["drives"], env["sp"], env["model"]
    t0, d = sp["t0"], drives["S2"]
    from idr.route_input import annotate_specs
    sp = annotate_specs(drives, [sp], RouteConfig())[0][0]       # default OSM planned route (pre-outage inputs)
    assert "route_b" in sp
    raw_p = dict(raw)
    dfp = raw["S2"].copy()
    for c in GNSS_COLS:
        if c in dfp:
            dfp.loc[dfp.index[t0:], c] = np.nan
    raw_p["S2"] = dfp
    v, psi, pos = d.v.copy(), d.psi.copy(), d.pos.copy()
    v[t0:], psi[t0:] = np.nan, np.nan
    dest = pos[-1].copy()
    pos[t0:] = np.nan
    pos[-1] = dest                                  # user-entered destination: external input, kept
    drives_p = dict(drives)
    drives_p["S2"] = dataclasses.replace(d, v=v, psi=psi, pos=pos)
    va, sa, ma = build_views(drives, [sp], CAUSAL, raw)
    vb, sb, mb = build_views(drives_p, [sp], CAUSAL, raw_p)
    assert json.dumps(ma, sort_keys=True, default=str) == json.dumps(mb, sort_keys=True, default=str)
    A, B = va[sa[0]["drive"]], vb[sb[0]["drive"]]
    for k in ("raw_u", "feats", "psi_mag"):
        np.testing.assert_array_equal(getattr(A, k), getattr(B, k))
    items = []
    for views, specs in ((va, sa), (vb, sb)):
        ds = OutageDataset(views, list(views), 1, mode="eval")
        ds.fixed = specs
        items.append(ds[0])
    for k in ("seq", "static", "route", "u_raw"):
        assert torch.equal(items[0][k], items[1][k]), k
        assert torch.isfinite(items[0][k]).all(), k
    oa = model_outputs(m, drives, [sp], CPU, CAUSAL)
    # model_outputs loads the clean raw data itself, so the poisoned path is run through its views explicitly
    with torch.no_grad():
        outs = [m(it["seq"][None], it["static"][None], it["route"][None], it["u_raw"][None]) for it in items]
    for k in ("u_hat", "p_stop"):
        assert torch.equal(outs[0][k], outs[1][k])
    np.testing.assert_array_equal(oa["u_hat"][0], outs[0]["u_hat"][0].numpy())
    rm = drive_roadmap("S2")
    u, ps = outs[0]["u_hat"][0].numpy(), outs[0]["p_stop"][0].numpy()
    tr = [run_out(rm, dd, t0, u, ps) for dd in (d, drives_p["S2"])]
    np.testing.assert_array_equal(tr[0], tr[1])


def run_out(rm, d, t0, u, ps):
    return run_outage(rm, d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1]), u[:, 0], u[:, 1], u[:, 2],
                      p_stop=ps, nav_cfg=NavConfig(**TFT_CFG), seed=t0 * 7)[0]


# (c) changing pre-outage data may change the causal calibration
def test_c_prefix_changes_can_affect_causal_calibration(env):
    df, t0 = env["raw"]["S2"], env["sp"]["t0"]
    a, _ = segment_of(_segments(df), t0)
    mod = df.copy()
    rows = mod.index[a:t0 - CAUSAL.guard_steps]
    mod.loc[rows, "v_true"] = mod.loc[rows, "v_true"] * 1.2
    c0, _ = outage_calibration(df, t0, CAUSAL)
    c1, _ = outage_calibration(mod, t0, CAUSAL)
    assert not np.allclose(c0["c_fwd"], c1["c_fwd"])


# (d) modes are explicit, traceable and cannot be mixed
def test_d_modes_are_explicit_and_traceable(env):
    df, t0 = env["raw"]["S2"], env["sp"]["t0"]
    _, mc = outage_calibration(df, t0, CAUSAL)
    _, mo_ = outage_calibration(df, t0, ORACLE)
    assert mc["calibration_mode"] == "causal_pre_outage" and mc["uses_samples_after_outage_start"] is False
    assert mo_["calibration_mode"] == "offline_oracle" and mo_["uses_samples_after_outage_start"] is True
    assert mo_["calibration_end_time_s"] >= t0 * DT > mc["calibration_end_time_s"]
    with pytest.raises(ValueError):
        CalibrationConfig(mode="oracle")
    assert CalibrationConfig().mode == "causal_pre_outage"            # default is causal
    m = env["model"]
    sp = env["specs"][:2]
    for cfg in (CAUSAL, ORACLE):
        assert {x["calibration_mode"] for x in model_outputs(m, env["drives"], sp, CPU, cfg)["cal_meta"]} == {cfg.mode}
    # comparison tooling refuses mixed modes
    from idr import ablation_table
    assert ablation_table.run_mode({"calibration": {"calibration_mode": "causal_pre_outage"}}) == "causal_pre_outage"
    assert ablation_table.run_mode({"summary": {}}) == "offline_oracle"              # legacy files = oracle
    real_load = ablation_table.load
    try:
        ablation_table.load = lambda tag, td: {"drift": 9, "drift_rng": None, "wrong": 1, "cross": 1, "along": 1,
                                               "w25": 1, "first": None, "mode": tag,
                                               "route": "osm_planned_route"}
        with pytest.raises(ValueError):
            ablation_table.table([("a", "causal_pre_outage", None), ("b", "offline_oracle", None)])
    finally:
        ablation_table.load = real_load


# (e) offline_oracle reproduces the committed behaviour bit-for-bit (reference saved before this change)
def test_e_offline_oracle_is_bit_identical_to_committed(env):
    ref = np.load(REF)
    specs = json.loads(str(ref["specs"]))
    # the reference predates route sources: it used the true-track route (now an explicit diagnostic)
    mo = model_outputs(env["model"], env["drives"], specs, CPU, ORACLE, ORACLE_ROUTE)
    for k in ("u_hat", "p_stop", "u_raw"):
        np.testing.assert_array_equal(mo[k], ref[k])
    for i, sp in enumerate(specs):
        d, t0 = env["drives"][sp["drive"]], sp["t0"]
        tr = run_outage(drive_roadmap(sp["drive"]), d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1]),
                        mo["u_hat"][i][:, 0], mo["u_hat"][i][:, 1], mo["u_hat"][i][:, 2], vib=mo["vib"][i],
                        p_stop=mo["p_stop"][i], nav_cfg=NavConfig(**TFT_CFG), seed=t0 * 7)[0]
        np.testing.assert_array_equal(tr, ref["tracks"][i])


# (f) inference stays causal: model outputs up to step k ignore inputs after k; EKF + map track up to k ignores
#     IMU after k; causal views are NaN outside the outage window and nothing reads there
def test_f_model_and_ekf_inference_are_causal(env):
    m = env["model"]
    g = torch.Generator().manual_seed(0)
    seq = torch.randn(1, CTX + HORIZON, 17, generator=g)
    static, route, u = torch.randn(1, 5, generator=g), torch.randn(1, 100, 4, generator=g), torch.randn(1, HORIZON, 3, generator=g)
    route[..., 3] = 1.0
    k = 200
    seq2, u2 = seq.clone(), u.clone()
    seq2[:, CTX + k + 1:] += 5 * torch.randn_like(seq2[:, CTX + k + 1:])
    u2[:, k + 1:] += 5 * torch.randn_like(u2[:, k + 1:])
    with torch.no_grad():
        a, b = m(seq, static, route, u), m(seq2, static, route, u2)
    for key in ("u_hat", "p_stop"):
        torch.testing.assert_close(a[key][:, :k + 1], b[key][:, :k + 1], rtol=0, atol=1e-5)
    rmj = _synthetic_map([([1, 2], [[0, -200], [0, 500]], "secondary", 0), ([2, 3], [[0, 500], [600, 500]], "secondary", 0)])
    T = 500
    w, acc = np.zeros(T), np.zeros(T)
    w[300:330] = -np.pi / 2 / 3.0
    w2, acc2 = w.copy(), acc.copy()
    w2[k + 1:] += 0.3
    acc2[k + 1:] -= 2.0
    t1 = run_outage(rmj, np.array([0.0, 0.0]), 0.0, 10.0, w, acc, seed=1)[0]
    t2 = run_outage(rmj, np.array([0.0, 0.0]), 0.0, 10.0, w2, acc2, seed=1)[0]
    np.testing.assert_array_equal(t1[:k + 1], t2[:k + 1])
    views, vs, _ = build_views(env["drives"], [env["sp"]], CAUSAL, env["raw"])
    v = views[vs[0]["drive"]]
    lo, hi = vs[0]["t0"] - CTX, vs[0]["t0"] + HORIZON
    assert np.isnan(v.raw_u[:lo]).all() and np.isnan(v.feats[hi:]).all() and np.isfinite(v.raw_u[lo:hi]).all()


def test_cold_start_never_falls_back_to_oracle(env):
    df = env["raw"]["S2"]
    segs = _segments(df)
    a, b = segs[0]
    t0 = a + CTX                                  # 4.5 s prefix
    cal, meta = outage_calibration(df, t0, CAUSAL)
    oracle, _ = outage_calibration(df, t0, ORACLE)
    assert meta["calibration_quality"] == "cold_start" and meta["orientation_calibration_status"] == "prior"
    assert not np.allclose(cal["c_fwd"], oracle["c_fwd"])
    assert meta["prefix_duration_s"] == pytest.approx((t0 - CAUSAL.guard_steps - a) * DT)
