"""Roundabout state detector: causal features over the last 80 m of travel -> {straight, ordinary_turn, entry,
circulating, exit}. Rule detector (always available) and a gradient-boosted classifier (kept only if it beats the rules
under leave-one-drive-out CV). `python -m idr.roundabout_detector` trains + reports per-class F1.

Inputs are what the phone has during an outage: yaw rate (CCW+), speed (EKF), lateral accel, dead-reckoned position,
and optionally the OSM rings (map feature).
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from idr.circle_fit import kasa, refine
from idr.data import DT
from idr.normalise import ARTIFACTS
from idr.roundabout_labels import STATES

WIN_M = 40.0          # short enough that a ~100 m roundabout traversal dominates the window
MAX_WIN = 300
FEATURES = ["dpsi", "abs_dpsi", "yaw_mean", "yaw_std", "kappa_mean", "kappa_std", "kappa_cv", "int_abs_yaw", "v_mean",
            "alat_abs", "sign_consist", "fit_r", "fit_rms", "dist_ring", "arc_len", "arc_per_rad"]


def window_features(w, v, a_lat, pos, t, ring_dist=np.nan):
    """Features at step t from the causal window ending at t (w: yaw CCW+, v: speed, pos: (N,2) DR positions)."""
    d = np.cumsum(v[max(0, t - MAX_WIN):t + 1][::-1]) * DT
    n = int(np.searchsorted(d, WIN_M)) + 1
    lo = max(0, t - min(n, MAX_WIN) + 1)
    ww, vv = w[lo:t + 1], v[lo:t + 1]
    dpsi = float(np.sum(ww) * DT)
    kap = -ww / (vv + 0.5)                          # clockwise-positive curvature, rad/m
    km, ks = float(np.mean(kap)), float(np.std(kap))
    sgn = np.sign(dpsi) if dpsi != 0 else 1.0
    consist = float(np.mean(np.sign(ww) == sgn))
    fit_r, fit_rms = 500.0, 50.0
    p = pos[lo:t + 1]
    if len(p) >= 8 and np.ptp(p, axis=0).max() > 5:
        try:
            (cx, cy, r), _, rms = refine(p, kasa(p), iters=5)
            fit_r, fit_rms = float(min(r, 500.0)), float(min(rms, 50.0))
        except Exception:
            pass
    return [dpsi, abs(dpsi), float(np.mean(ww)), float(np.std(ww)), km, ks, ks / (abs(km) + 1e-3),
            float(np.sum(np.abs(ww)) * DT), float(np.mean(vv)), float(np.mean(np.abs(a_lat[lo:t + 1]))), consist,
            fit_r, fit_rms, float(ring_dist), float(np.sum(vv) * DT),
            float(np.sum(vv) * DT / max(abs(dpsi), 1e-2))]     # cumulative arc; arc per radian ~ turning radius


def ring_distance(rm, xy) -> float:
    """Distance from a point to the nearest ring's circle (|dist to center - r|); nan if no map."""
    if rm is None or not rm.rings:
        return np.nan
    c = np.array([[r["cx"], r["cy"]] for r in rm.rings])
    rad = np.array([r["r"] for r in rm.rings])
    return float(np.min(np.abs(np.hypot(*(c - xy).T) - rad)))


def rule_predict(F: np.ndarray, use_map=True) -> np.ndarray:
    """Hand-set rules on the feature matrix (rows = time, 1 Hz)."""
    f = {k: F[:, i] for i, k in enumerate(FEATURES)}
    circ_shape = (f["sign_consist"] > 0.85) & (f["kappa_cv"] < 0.6) & (f["fit_r"] > 8) & (f["fit_r"] < 45) & \
                 (f["fit_rms"] < 3.0) & (f["abs_dpsi"] > np.radians(45))
    near = f["dist_ring"] < 25 if use_map else np.ones(len(F), bool)
    near = np.where(np.isnan(f["dist_ring"]), True, near)
    out = np.where(np.abs(f["yaw_mean"]) > 0.1, 1, 0)
    circ = circ_shape & near
    out[circ] = 3
    if use_map:
        approach = (f["dist_ring"] < 40) & ~circ
        out[approach] = 2
    # exit: within 3 samples after circulating, once curvature has decayed
    last_circ = -10
    for i in range(len(out)):
        if out[i] == 3:
            last_circ = i
        elif i - last_circ <= 3 and abs(f["kappa_mean"][i]) < 0.03:
            out[i] = 4
    return out


def build_dataset(drives, labels, names, stride=10, map_feature=True, rng=None):
    """Features at 1 Hz using phone yaw (raw aligned), GT speed and a dead-reckoned egocentric track."""
    from idr.osm_map import drive_roadmap
    rng = rng or np.random.default_rng(0)
    X, y, g = [], [], []
    for n in names:
        if n not in labels:
            continue
        d, st = drives[n], labels[n]["states"]
        rm = drive_roadmap(n) if map_feature else None
        w = d.raw_u[:, 0].astype(float)
        a_lat = d.raw_u[:, 2].astype(float)
        psi = np.cumsum(-w) * DT
        pos = np.cumsum(np.column_stack([d.v * np.sin(psi), d.v * np.cos(psi)]), 0) * DT
        noise = rng.normal(0, 10.0, (len(d.v), 2))            # DR position error in deployment
        for t in range(50, len(d.v), stride):
            if not d.valid[max(0, t - 50):t + 1].all() or d.v[t] < 1.0:
                continue
            rd = ring_distance(rm, d.pos[t] + noise[t]) if map_feature else np.nan
            X.append(window_features(w, d.v, a_lat, pos, t, rd))
            y.append(int(st[t]))
            g.append(n)
    return np.array(X, dtype=float), np.array(y), np.array(g)


def per_class_f1(y, p):
    out = {}
    for c, name in enumerate(STATES):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[name] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3),
                     "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 3), "support": int(np.sum(y == c))}
    out["macro_f1"] = round(float(np.mean([out[s]["f1"] for s in STATES if out[s]["support"] > 0])), 3)
    return out


def train_gbdt(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    Xf = np.nan_to_num(X, nan=999.0)
    clf.fit(Xf, y)
    return clf


def lodo_gbdt(X, y, g):
    """Leave-one-drive-out predictions (drives with no roundabouts still serve as negatives)."""
    pred = np.zeros_like(y)
    for n in np.unique(g):
        tr, te = g != n, g == n
        if len(np.unique(y[tr])) < 3:
            pred[te] = 0
            continue
        pred[te] = train_gbdt(X[tr], y[tr]).predict(np.nan_to_num(X[te], nan=999.0))
    return pred


if __name__ == "__main__":
    from idr.dataset import load_drives, split_names
    from idr.roundabout_labels import all_labels
    drives, _ = load_drives()
    sp = split_names(list(drives))
    dev_names = sp["train"] + sp["val"]
    labels = all_labels(drives, dev_names + sp["test"] + sp["scenario"])
    report = {}
    for use_map in (True, False):
        X, y, g = build_dataset(drives, labels, dev_names, map_feature=use_map)
        key = "map" if use_map else "map_free"
        rules = per_class_f1(y, rule_predict(X, use_map))
        gb = per_class_f1(y, lodo_gbdt(X, y, g))
        report[key] = {"dev_samples": int(len(y)), "class_counts": {s: int(np.sum(y == i)) for i, s in enumerate(STATES)},
                       "rules_lodo": rules, "gbdt_lodo": gb}
        best = "gbdt" if gb["macro_f1"] > rules["macro_f1"] else "rules"
        report[key]["chosen"] = best
        if best == "gbdt":
            clf = train_gbdt(X, y)
            (ARTIFACTS / f"rb_detector_{key}.pkl").write_bytes(pickle.dumps(clf))
        # held-out test drives (never used above)
        Xt, yt, _ = build_dataset(drives, labels, sp["test"] + sp["scenario"], map_feature=use_map)
        pt = train_gbdt(X, y).predict(np.nan_to_num(Xt, nan=999.0)) if best == "gbdt" else rule_predict(Xt, use_map)
        report[key]["test"] = per_class_f1(yt, pt)
        fp_circ = np.sum((pt == 3) & (yt != 3) & (yt != 2) & (yt != 4))
        report[key]["test_false_circulating_per_hour"] = round(float(fp_circ / max(len(yt) / 3600, 1e-9)), 2)
        print(f"[{key}] dev n={len(y)} counts={report[key]['class_counts']}")
        print(f"   LODO macro-F1: rules {rules['macro_f1']}  gbdt {gb['macro_f1']}  -> chosen {best}")
        print(f"   TEST circulating F1 {report[key]['test']['circulating']}  macro-F1 {report[key]['test']['macro_f1']}  "
              f"false circulating/h {report[key]['test_false_circulating_per_hour']}")
    (ARTIFACTS / "rb_detector_report.json").write_text(json.dumps(report, indent=1))
