"""Export the validated model (tft_ep2) to ONNX + a manifest for the app. `python -m idr.export --tag tft_ep2`.

Fixed shapes (the app pads): seq (B, 650, 17) = 5 s context + 60 s outage at 10 Hz, static (B, 5), route (B, 100, 4),
u_raw (B, 600, 3). Outputs: u_hat (corrected [w_yaw, a_fwd, a_lat]), p_stop, e_hat, b_hat.
Verification: ONNX Runtime vs PyTorch on random / braking / constant-speed / roundabout validation windows, plus the
end-to-end open-loop drift difference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from idr import physics
from idr.dataset import CTX, HORIZON, N_SEQ, N_STATIC, SEQ_FEATURES, OutageDataset, attach_labels, load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG
from idr.model import E_SCALE, ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import OSM_DIR
from idr.route_features import N_PTS

EXPORT_DIR = Path(__file__).resolve().parent / "export"


class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, seq, static, route, u_raw):
        o = self.m(seq, static, route, u_raw)
        return o["u_hat"], o["p_stop"], o["e_hat"], o["b_hat"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def pick_windows(drives, names, n_each=16, seed=0):
    """Validation windows by category (by the dominant condition in the 60 s outage)."""
    rng = np.random.default_rng(seed)
    specs = {"random": [], "braking": [], "constant": [], "roundabout": []}
    for name in names:
        d = drives[name]
        for sp in make_specs(drives, [name], HORIZON, stride=150):
            t0 = sp["t0"]
            acc = d.u_true[t0:t0 + HORIZON, 1]
            st = d.states[t0:t0 + HORIZON] if d.states is not None else np.zeros(HORIZON, int)
            specs["random"].append(sp)
            if (acc < -1.0).mean() > 0.05:
                specs["braking"].append(sp)
            if np.abs(acc).mean() < 0.15 and d.v[t0:t0 + HORIZON].mean() > 8:
                specs["constant"].append(sp)
            if (st >= 2).any():
                specs["roundabout"].append(sp)
    out = {}
    for k, lst in specs.items():
        if lst:
            idx = rng.choice(len(lst), min(n_each, len(lst)), replace=False)
            out[k] = [lst[i] for i in idx]
    return out


def batch_of(drives, specs):
    ds = OutageDataset(drives, sorted({s["drive"] for s in specs}), len(specs), mode="eval")
    ds.fixed = specs
    items = [ds[i] for i in range(len(specs))]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def write_golden_file(wrapper, drives, cats, onnx_path):
    """One real window per category (braking / roundabout / constant / random) with PyTorch outputs, for the app's
    startup self-check (idr.onnx_runtime.make_session) and the CI parity test."""
    from idr.onnx_runtime import OUTPUTS, write_golden
    specs = [lst[0] for lst in cats.values() if lst]
    b = batch_of(drives, specs)
    with torch.no_grad():
        out = wrapper(b["seq"], b["static"], b["route"], b["u_raw"])
    path = onnx_path.with_name(onnx_path.stem + "_golden.npz")
    write_golden(path, {k: b[k].numpy() for k in ("seq", "static", "route", "u_raw")},
                 {n: o.numpy() for n, o in zip(OUTPUTS, out)})
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--val_drives", nargs="+", default=["S2", "Vtb5"])
    args = ap.parse_args()
    EXPORT_DIR.mkdir(exist_ok=True)
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    model = ResidualTFT(d=ck["args"]["d"], use_tft=not ck["args"]["no_tft"]).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    wrapper = Wrapper(model).eval()

    L = CTX + HORIZON
    ex = (torch.zeros(1, L, N_SEQ), torch.zeros(1, N_STATIC), torch.zeros(1, N_PTS, 4), torch.zeros(1, HORIZON, 3))
    onnx_path = EXPORT_DIR / f"{args.tag}.onnx"
    torch.onnx.export(wrapper, ex, str(onnx_path), input_names=["seq", "static", "route", "u_raw"],
                      output_names=["u_hat", "p_stop", "e_hat", "b_hat"],
                      dynamic_axes={n: {0: "batch"} for n in ["seq", "static", "route", "u_raw", "u_hat", "p_stop",
                                                               "e_hat", "b_hat"]},
                      opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(onnx_path))
    # ONNX Runtime's EXTENDED/ALL graph fusions compute this model WRONGLY (onnxruntime 1.29: |diff| up to 0.6 on
    # real windows); BASIC matches PyTorch to ~1e-6. The app must create its session with ORT_ENABLE_BASIC.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    drives, stats = load_drives(args.val_drives)
    from idr.roundabout_labels import all_labels
    attach_labels(drives, all_labels(drives, args.val_drives))
    cats = pick_windows(drives, args.val_drives)
    verify = {}
    for cat, specs in cats.items():
        b = batch_of(drives, specs)
        with torch.no_grad():
            t_out = wrapper(b["seq"], b["static"], b["route"], b["u_raw"])
        o_out = sess.run(None, {"seq": b["seq"].numpy(), "static": b["static"].numpy(),
                                "route": b["route"].numpy(), "u_raw": b["u_raw"].numpy()})
        diffs = {n: float(np.max(np.abs(t.numpy() - o))) for n, t, o in zip(["u_hat", "p_stop", "e_hat", "b_hat"],
                                                                              t_out, o_out)}
        # end-to-end open-loop drift with each backend
        drift = {}
        for name_, (u, ps) in {"torch": (t_out[0], t_out[1]), "onnx": (torch.tensor(o_out[0]), torch.tensor(o_out[1]))}.items():
            tr = physics.integrate(u[..., 0], u[..., 1], b["v0"], ps)
            err = (tr["pos"][:, -1] - b["pos_rel"][:, -1]).norm(dim=-1)
            dist = b["dist"][:, -1].clamp(min=1.0)
            drift[name_] = float(torch.median(err / dist * 100))
        verify[cat] = {"n": len(specs), "max_abs_diff": diffs, "median_open_loop_drift_pct": drift,
                       "drift_diff_pct_points": abs(drift["torch"] - drift["onnx"])}
        print(f"{cat:10s} n={len(specs):2d} max|torch-onnx| u_hat {diffs['u_hat']:.2e} p_stop {diffs['p_stop']:.2e} | "
              f"open-loop drift torch {drift['torch']:.3f}% onnx {drift['onnx']:.3f}%")
    worst = max(max(v["max_abs_diff"].values()) for v in verify.values())
    assert worst < 1e-3, f"ONNX mismatch {worst}"
    write_golden_file(wrapper, drives, cats, onnx_path)

    # manifest
    def load_json(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None
    step0 = {k: load_json(ARTIFACTS / f"{k}.json") for k in ("step0_YM", "step0_Vw4")}
    metrics = {}
    for k, v in step0.items():
        if v:
            metrics[k] = {var: {m_: v["summary"][var].get(m_) for m_ in ("median_drift_pct", "median_drift_pct_range",
                                                                           "pct_under_10pct", "wrong_road_pct",
                                                                           "median_along_m", "median_cross_m")}
                          for var in v["summary"]}
    valA = load_json(ARTIFACTS / "mm_eval_valA_S2_Vtb5_60s.json")
    if valA:
        metrics["validation_S2_Vtb5"] = {var: {m_: valA["summary"][var].get(m_) for m_ in ("median_drift_pct",
                                                                                          "median_drift_pct_range",
                                                                                          "pct_under_10pct",
                                                                                          "wrong_road_pct")}
                                         for var in valA["summary"]}
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": {"checkpoint": f"idr/artifacts/{args.tag}.pt", "sha256_16": sha(ARTIFACTS / f"{args.tag}.pt"),
                  "onnx": str(onnx_path.relative_to(onnx_path.parents[2])), "onnx_sha256_16": sha(onnx_path),
                  "opset": 17, "params": sum(p.numel() for p in model.parameters()),
                  "missing_keys_at_load": list(missing), "train_args": ck["args"]},
        "onnxruntime": {"required_graph_optimization_level": "ORT_ENABLE_BASIC",
                        "warning": "ORT_ENABLE_EXTENDED / ORT_ENABLE_ALL (the default) produce wrong outputs for this "
                                   "model with onnxruntime 1.29 (operator-fusion bug); verified basic == PyTorch.",
                        "verified_with": ort.__version__,
                        "startup_check": {"golden_file": f"{args.tag}_golden.npz", "tolerance": 1e-4,
                                          "reference": "idr/onnx_runtime.py make_session (asserts BASIC + golden)"},
                        "integration_checklist": "idr/export/APP_INTEGRATION.md",
                        "ci_test": "python -m pytest idr/tests/test_onnx_parity.py"},
        "model_rewrites_for_export": ["VSN einsum -> bmm over variables (memory-safe)", "bool causal mask -> additive -inf float mask",
                                      "route valid-mask in-place write -> torch.cat"],
        "sampling_rate_hz": 10, "dt_s": physics.DT, "context_steps": CTX, "horizon_steps": HORIZON,
        "inputs": {
            "seq": {"shape": ["B", L, N_SEQ], "feature_order": SEQ_FEATURES,
                    "notes": "steps 0..49 = 5 s before GNSS loss (gnss_flag=1), 50..649 = outage; tau=(k-49)*0.1/60; "
                             "sin/cos dpsi_* relative to last GNSS heading psi0; first 8 channels z-scored with "
                             "norm_stats (clip +-6 sigma); pad unused future steps with zeros"},
            "static": {"shape": ["B", N_STATIC], "order": ["v0/30", "log1p(dist_dest/100)/5", "sin(bearing_dest-psi0)",
                                                          "cos(bearing_dest-psi0)", "route_available"]},
            "route": {"shape": ["B", N_PTS, 4], "spec": "idr/route_features.py (20 m arc-length samples: sin, cos of "
                                                         "bearing-psi0, curvature*20, valid)"},
            "u_raw": {"shape": ["B", HORIZON, 3], "channels": ["w_yaw rad/s (CCW+)", "a_fwd m/s^2", "a_lat m/s^2"],
                      "notes": "phone IMU after GNSS-motion axis calibration (idr/normalise.calibrate) and gyro bias"},
        },
        "outputs": {"u_hat": "corrected [w_yaw, a_fwd, a_lat] per outage step", "p_stop": "stationary probability",
                    "e_hat": "per-step sensor-error residual", "b_hat": "per-outage bias",
                    "residual_scale": E_SCALE.tolist()},
        "normalisation": {"norm_stats": stats, "sha256_16": sha(ARTIFACTS / "norm_stats.json"),
                          "note": "stats used for every reported evaluation (train drives excl. VAL=S2,Vtb5,S3c,Vta1a);"
                                  " tft_ep2 was trained with S3c/Vta1a included in the stats - small known mismatch"},
        "downstream": {"integration": "idr/physics.py (d_psi=-cumsum(w)dt, v=relu(v0+cumsum(a)dt)*(1-p_stop))",
                       "ekf": {"class": "idr.nav_filter.NavEKF", "config": {**NavConfig(**TFT_CFG).__dict__}},
                       "map_matching": {"class": "idr.map_matching.MapMatcher", "defaults": "MMConfig() (Step-0 "
                                        "validated: full-state map updates, roundabout/landmark/calibration off)"},
                       "speed_calibration": None},
        "osm_cache": {f.name: {"bytes": f.stat().st_size,
                               "modified": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")}
                      for f in sorted(OSM_DIR.glob("*.osm.pbf"))},
        "evaluation_seeds": "particle filter seed = t0*7 + {0,1,2}",
        "metrics_60s_outages": metrics,
        "onnx_verification": verify,
    }
    (EXPORT_DIR / f"{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and manifest; worst |torch-onnx| {worst:.2e}")


if __name__ == "__main__":
    main()
