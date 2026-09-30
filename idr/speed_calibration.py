"""Speed calibration fitted on TRAINING drives only. `python -m idr.speed_calibration --tag tft_ep2`.

Collects (v_hat, v_GT) on 60 s outages from the training drives (never validation/test) and fits
  * affine   v_cal = clip(a + b*v_hat, 0, v_max)            (Huber regression)
  * isotonic v_cal = g(v_hat), monotone non-decreasing       (stored as knots, applied by linear interpolation)
  * lead     v_lead = v_cal + beta * (v_cal[t] - v_cal[t-1]) / dt   (causal lead compensator)
plus a per-phase lag analysis (all / braking / junction approach / roundabout entry / post-turn).
Calibration is a function of the MODEL OUTPUT only, so it is usable at inference time.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from idr.dataset import load_drives, split_names
from idr.evaluate import make_specs
from idr.evaluate_mm import model_outputs
from idr.model import ResidualTFT
from idr.normalise import ARTIFACTS
from idr.osm_map import OSM_DIR, drive_roadmap
from idr.roundabout_labels import label_drive
from idr.train import pick_device

DT = 0.1
H = 600
V_MAX = 45.0


def load_cal(path_or_tag):
    if path_or_tag is None:
        return None
    p = ARTIFACTS / (path_or_tag if str(path_or_tag).endswith(".json") else f"speed_cal_{path_or_tag}.json")
    return json.loads(p.read_text())


def apply_cal(v, cal, kind=None, v0=None):
    """Apply a fitted calibration to speeds v (array or scalar). kind in {'affine','isotonic'} (default cal['use']).
    If cal['anchor'] and v0 is given, only the IMU-driven change is calibrated: v0 + g(v) - g(v0), so the known GNSS
    speed at outage start is kept exactly (plain g(v) also rescales v0, which hurt constant/high-speed driving)."""
    if cal is None:
        return v
    if cal.get("anchor") and v0 is not None:
        c2 = {**cal, "anchor": False}
        return np.maximum(v0 + apply_cal(v, c2, kind) - apply_cal(v0, c2, kind), 0.0)
    kind = kind or cal.get("use", "isotonic")
    if kind == "affine":
        return np.clip(cal["affine"]["a"] + cal["affine"]["b"] * np.asarray(v), 0.0, V_MAX)
    xs, ys = np.asarray(cal["isotonic"]["x"]), np.asarray(cal["isotonic"]["y"])
    return np.interp(v, xs, ys)


def cal_slope(v, cal, kind=None):
    """Local derivative dg/dv (for the EKF Jacobian)."""
    if cal is None:
        return 1.0
    kind = kind or cal.get("use", "isotonic")
    if kind == "affine":
        return float(cal["affine"]["b"])
    xs, ys = np.asarray(cal["isotonic"]["x"]), np.asarray(cal["isotonic"]["y"])
    j = int(np.clip(np.searchsorted(xs, v) - 1, 0, len(xs) - 2))
    return float((ys[j + 1] - ys[j]) / max(xs[j + 1] - xs[j], 1e-6))


def collect(model, drives, names, dev, stride=300):
    """Per outage: v_hat (uncalibrated, stop-gated), v_GT, phase masks."""
    out = []
    for name in names:
        d = drives[name]
        has_map = (OSM_DIR / f"{name}.pkl").exists()
        rb = label_drive(d, drive_roadmap(name))[0] if has_map else None
        specs = make_specs(drives, [name], H, stride=stride)
        if not specs:
            continue
        mo = model_outputs(model, drives, specs, dev)
        for i, sp in enumerate(specs):
            t0 = sp["t0"]
            v = d.v[t0:t0 + H]
            if v.sum() * DT < 50:
                continue
            u, ps = mo["u_hat"][i], mo["p_stop"][i]
            vraw = np.maximum(float(d.v[t0 - 1]) + np.cumsum(u[:, 1]) * DT, 0.0)
            acc = d.u_true[t0:t0 + H, 1]
            yaw = np.abs(d.u_true[max(0, t0 - 100):t0 + H, 0]) > 0.1
            post = np.convolve(yaw.astype(float), np.ones(101), mode="full")[:len(yaw)][-H:] > 0
            st = rb[t0:t0 + H] if rb is not None else np.zeros(H, int)
            out.append({"drive": name, "vraw": vraw, "ps": ps, "v": v,
                        "phases": {"all": np.ones(H, bool), "braking": acc < -0.5, "post-turn": post,
                                   "roundabout entry": st == 2, "roundabout circulation": st == 3}})
    return out


def best_lag(items, phase, maxlag=30):
    best = (0, np.inf)
    for tau in range(-maxlag, maxlag + 1):
        se, n = 0.0, 0
        for it in items:
            vh, v, m = it["vcal"], it["v"], it["phases"][phase]
            idx = np.flatnonzero(m)
            idx = idx[(idx + tau >= 0) & (idx + tau < H)]
            if len(idx) == 0:
                continue
            se += np.sum((vh[idx + tau] - v[idx]) ** 2)
            n += len(idx)
        if n and se / n < best[1]:
            best = (tau, se / n)
    return best[0] * DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    args = ap.parse_args()
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives()
    train = split_names(list(drives))["train"]
    items = collect(m, drives, train, dev)
    gate = lambda it: (1 - it["ps"])
    X = np.concatenate([it["vraw"] for it in items])
    Y = np.concatenate([it["v"] / np.maximum(gate(it), 1e-3) * (gate(it) > 0.05) for it in items])
    keep = np.concatenate([gate(it) > 0.05 for it in items])        # fit where the stop gate is open
    X, Y = X[keep], Y[keep]
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import HuberRegressor
    hub = HuberRegressor(epsilon=1.35, max_iter=500).fit(X[::5, None], Y[::5])
    iso = IsotonicRegression(y_min=0.0, y_max=V_MAX, increasing=True, out_of_bounds="clip").fit(X, Y)
    grid = np.unique(np.concatenate([np.linspace(0, 40, 81), [V_MAX]]))
    cal = {"tag": args.tag, "train_drives": train, "n_samples": int(len(X)),
           "affine": {"a": float(hub.intercept_), "b": float(hub.coef_[0])},
           "isotonic": {"x": grid.tolist(), "y": iso.predict(grid).tolist()}, "use": "isotonic"}

    # lead compensator beta (causal) on top of the isotonic calibration, and per-phase lags
    for it in items:
        it["vcal"] = apply_cal(it["vraw"], cal) * (1 - it["ps"])
    dv = np.concatenate([np.r_[0.0, np.diff(it["vcal"])] / DT for it in items])
    vc = np.concatenate([it["vcal"] for it in items])
    vt = np.concatenate([it["v"] for it in items])
    beta = float(np.sum(dv * (vt - vc)) / max(np.sum(dv ** 2), 1e-9))
    cal["lead_beta"] = float(np.clip(beta, 0.0, 1.5))
    cal["lag_s_by_phase_after_isotonic"] = {p: best_lag(items, p) for p in items[0]["phases"]}
    for it in items:
        it["vcal"] = it["vraw"] * (1 - it["ps"])
    cal["lag_s_by_phase_uncalibrated"] = {p: best_lag(items, p) for p in items[0]["phases"]}

    def dist_err(fn):
        errs = [abs(np.sum(fn(it)) - np.sum(it["v"])) / np.sum(it["v"]) * 100 for it in items]
        sgn = [(np.sum(fn(it)) - np.sum(it["v"])) / np.sum(it["v"]) * 100 for it in items]
        return float(np.median(errs)), float(np.median(sgn))
    cal["train_fit_dist_err_pct"] = {
        "uncalibrated": dist_err(lambda it: it["vraw"] * (1 - it["ps"])),
        "affine": dist_err(lambda it: apply_cal(it["vraw"], cal, "affine") * (1 - it["ps"])),
        "isotonic": dist_err(lambda it: apply_cal(it["vraw"], cal, "isotonic") * (1 - it["ps"])),
    }
    out = ARTIFACTS / f"speed_cal_{args.tag}.json"
    out.write_text(json.dumps(cal, indent=1))
    print(json.dumps({k: v for k, v in cal.items() if k not in ("isotonic", "train_drives")}, indent=1))
    print("isotonic g(v): " + ", ".join(f"{x:.0f}->{y:.1f}" for x, y in zip(grid[::8], np.array(cal['isotonic']['y'])[::8])))
    print("saved", out)


if __name__ == "__main__":
    main()
