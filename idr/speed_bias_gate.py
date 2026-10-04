"""Causal pre-outage speed-bias estimate for the map look-ahead slowdown prior's bias gate.

Before an outage the app has GNSS, so the dead-reckoning pipeline's recent speed bias can be measured on a SHADOW
outage: the same TFT + EKF (no map) run over a window that ends strictly before outage_start - guard, compared with
the GNSS speed in that window.

    shadow start   t0s = t0 - guard_steps - W            (W = recent_bias_window_s * 10)
    shadow inputs  its own causal calibration (prefix before t0s - guard), its own OSM planned route, context GNSS
                   before t0s; IMU only afterwards (the model is causal, so outputs for the first W steps only see
                   IMU up to that step)
    bias           mean(ekf_speed[0:W] - gnss_speed[t0s:t0s+W])  -- every GNSS index used is < t0 - guard

Unavailable (not enough contiguous valid history in the same segment) -> available = False; the gate's
bias_unknown_policy decides (default: suppress).
"""
from __future__ import annotations

import numpy as np

from idr.dataset import CTX


def shadow_spec(d, sp, window_steps, guard_steps):
    t0 = int(sp["t0"])
    t0s = t0 - guard_steps - window_steps
    starts = getattr(d, "_starts_set", None)
    if starts is None:
        starts = set(int(x) for x in d.starts)
        d._starts_set = starts
    if t0s not in starts or int(d.run_end[t0s]) != int(d.run_end[t0]) or t0s - CTX < 0:
        return None
    return {**sp, "t0": t0s, "seed": int(t0s), "shadow_of": t0}


def recent_speed_bias(model, drives, specs, dev, calibration, route, nav_cfg, window_s=30.0, roadmaps=None):
    """Per spec: {"available", "recent_ekf_speed_bias_mps", "recent_gnss_speed_mean_mps", "recent_ekf_speed_mean_mps",
    "n_samples", "window": [first, last] row index, "outage_start_index"}."""
    from idr.evaluate_mm import model_outputs
    from idr.map_matching import run_outage
    from idr.osm_map import drive_roadmap
    roadmaps = roadmaps if roadmaps is not None else {}
    W = int(round(window_s * 10))
    guard = calibration.guard_steps
    out = [None] * len(specs)
    sh, idx = [], []
    for i, sp in enumerate(specs):
        s = shadow_spec(drives[sp["drive"]], sp, W, guard)
        if s is None:
            out[i] = {"available": False, "reason": "insufficient_pre_outage_history",
                      "recent_ekf_speed_bias_mps": None, "recent_gnss_speed_mean_mps": None,
                      "recent_ekf_speed_mean_mps": None, "n_samples": 0, "outage_start_index": int(sp["t0"])}
        else:
            sh.append(s)
            idx.append(i)
    if sh:
        mo = model_outputs(model, drives, sh, dev, calibration=calibration, route=route)
        for j, (i, s) in enumerate(zip(idx, sh)):
            d, t0s = drives[s["drive"]], s["t0"]
            name = s["drive"]
            if name not in roadmaps:
                roadmaps[name] = drive_roadmap(name)
            u = mo["u_hat"][j][:W]
            _, _, _, trc = run_outage(roadmaps[name], d.pos[t0s - 1], float(d.psi[t0s - 1]), float(d.v[t0s - 1]),
                                      u[:, 0], u[:, 1], u[:, 2], p_stop=mo["p_stop"][j][:W], nav_cfg=nav_cfg,
                                      use_map=False, return_trace=True)
            v_ekf = np.asarray(trc["speed_est"][:W], float)
            v_gnss = np.asarray(d.v[t0s:t0s + W], float)
            assert t0s + W <= int(specs[i]["t0"]) - guard            # strictly pre-outage GNSS
            out[i] = {"available": True, "recent_ekf_speed_bias_mps": float(np.mean(v_ekf - v_gnss)),
                      "recent_gnss_speed_mean_mps": float(np.mean(v_gnss)),
                      "recent_ekf_speed_mean_mps": float(np.mean(v_ekf)), "n_samples": int(W),
                      "window": [int(t0s), int(t0s + W - 1)], "outage_start_index": int(specs[i]["t0"]),
                      "window_s": float(window_s)}
    return out
