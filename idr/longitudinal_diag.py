"""Longitudinal (along-track) diagnostics — log only, no behaviour change. `python -m idr.longitudinal_diag`.

Per outage (TFT + NHC-EKF + map + planned route, seed 0):
  * signed along-track error over time: EKF position projected onto the true path (arc coordinate) minus true arc
  * dead-reckoned vs true cumulative distance, EKF speed-scale estimate over time
  * junction-crossing time error dt_junction = t_est - t_ref for every junction (degree >= 3 node) on the true path:
    t_ref = when the GNSS track passes the node, t_est = when the dominant particle's matched edge changes at that node
  * wrong-road outages classified: SPEED-driven (along-track error / dt_junction grows steadily before the wrong branch)
    vs TOPOLOGY-driven (small timing error, wrong successor chosen at a junction)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from idr.dataset import load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG, model_outputs
from idr.map_matching import run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import drive_roadmap, planned_route
from idr.train import pick_device

FIGS = Path(__file__).resolve().parent / "figures"
DT = 0.1


def arc_projection(path: np.ndarray, pts: np.ndarray):
    """Arc coordinate (m along `path`) of the nearest path point for each of `pts`, and the lateral distance."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    from scipy.spatial import cKDTree
    d, j = cKDTree(path).query(pts)
    return s[j], d, s


def junction_times(rm, gt, trace, t_offset_steps=0):
    """Reference and estimated crossing times (s) for junction nodes on the true path."""
    node_xy = {}
    for e in range(len(rm.e_u)):
        node_xy.setdefault(int(rm.e_v[e]), rm.pts[rm.e_start[e] + rm.e_n[e] - 1])
    nodes = [n for n, deg in rm.node_deg.items() if deg >= 6 and n in node_xy]   # deg counts in+out of both dirs
    if not nodes:
        return []
    xy = np.array([node_xy[n] for n in nodes])
    from scipy.spatial import cKDTree
    d, j = cKDTree(gt).query(xy)
    on_path = [(nodes[i], int(j[i])) for i in range(len(nodes)) if d[i] < 8.0]
    # estimated: first trace sample whose dominant edge starts at the node (edge transition through it)
    est = {}
    prev = None
    for x in trace:
        e = x["edge"]
        if prev is not None and e != prev and int(rm.e_u[e]) == int(rm.e_v[prev]):
            est.setdefault(int(rm.e_u[e]), x["k"])
        prev = e
    out = []
    for n, k_ref in on_path:
        if n in est:
            out.append({"node": n, "t_ref": k_ref * DT, "t_est": est[n] * DT, "dt": (est[n] - k_ref) * DT})
    return sorted(out, key=lambda r: r["t_ref"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="+", default=["Y1", "M"])
    ap.add_argument("--stride", type=int, default=600)
    ap.add_argument("--plots", type=int, default=4)
    args = ap.parse_args()
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives(args.drives)
    H = 600
    recs, keep = [], []
    for name in args.drives:
        d = drives[name]
        rm = drive_roadmap(name)
        specs = make_specs(drives, [name], H, stride=args.stride)
        mo = model_outputs(m, drives, specs, dev)
        for i, sp in enumerate(specs):
            t0 = sp["t0"]
            gt = d.pos[t0:t0 + H]
            true_dist = np.cumsum(d.v[t0:t0 + H]) * DT
            if true_dist[-1] < 50 or np.mean(rm.nearest(gt[::10])[0] < 15) < 0.9:
                continue
            p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
            u, ps = mo["u_hat"][i], mo["p_stop"][i]
            tr, snap, st, trc = run_outage(rm, p0, psi0, v0, u[:, 0], u[:, 1], u[:, 2], p_stop=ps,
                                           nav_cfg=NavConfig(**TFT_CFG), seed=t0 * 7,
                                           route_edges=planned_route(rm, p0, psi0, d.pos[-1]), return_trace=True)
            path = d.pos[t0:t0 + H + 300]
            s_proj, lat, s_path = arc_projection(path, tr)
            along_err = s_proj - s_path[:H]                       # signed: + = estimate ahead of truth
            off_road_end = float(np.min(np.hypot(*(path - snap[-1]).T)))
            jt = junction_times(rm, gt, trc["mm_trace"])
            wrong = off_road_end > 15.0
            # classification of wrong-road outages
            cls = None
            if wrong:
                k_off = int(np.argmax(lat > 15.0)) if (lat > 15.0).any() else H - 1
                pre = along_err[max(0, k_off - 200):k_off + 1]
                growth = abs(pre[-1]) - abs(pre[0]) if len(pre) > 1 else 0.0
                big_along = abs(along_err[k_off]) > 30.0
                cls = "speed" if (big_along and growth > 10.0) else "topology"
            recs.append({"drive": name, "t0": t0, "wrong_road": wrong, "class": cls,
                         "along_end": float(along_err[-1]), "abs_along_end": float(abs(along_err[-1])),
                         "dist_ratio": float(trc["dist"][-1] / true_dist[-1]),
                         "speed_scale_end": float(trc["speed_scale"][-1]),
                         "dt_junction": [r["dt"] for r in jt], "n_junctions": len(jt)})
            keep.append((name, t0, along_err, trc, true_dist, jt, wrong, cls))
        print(name, "done", flush=True)

    dts = np.concatenate([np.abs(r["dt_junction"]) for r in recs if r["dt_junction"]]) if recs else np.array([])
    wr = [r for r in recs if r["wrong_road"]]
    summary = {
        "n_outages": len(recs),
        "median_abs_along_end_m": float(np.median([r["abs_along_end"] for r in recs])),
        "worst25_abs_along_end_m": float(np.percentile([r["abs_along_end"] for r in recs], 75)),
        "share_estimate_ahead_pct": float(np.mean([r["along_end"] > 0 for r in recs]) * 100),
        "median_dist_ratio": float(np.median([r["dist_ratio"] for r in recs])),
        "median_abs_dist_err_pct": float(np.median([abs(r["dist_ratio"] - 1) for r in recs]) * 100),
        "median_speed_scale_end": float(np.median([r["speed_scale_end"] for r in recs])),
        "junction_crossings": int(len(dts)),
        "median_abs_dt_junction_s": float(np.median(dts)) if len(dts) else None,
        "p90_abs_dt_junction_s": float(np.percentile(dts, 90)) if len(dts) else None,
        "wrong_road_outages": len(wr),
        "wrong_road_speed_driven": sum(r["class"] == "speed" for r in wr),
        "wrong_road_topology_driven": sum(r["class"] == "topology" for r in wr),
    }
    (ARTIFACTS / "longitudinal_diag.json").write_text(json.dumps({"summary": summary, "rows": recs}, indent=1))
    print(json.dumps(summary, indent=1))

    # plots
    FIGS.mkdir(exist_ok=True)
    order = sorted(range(len(keep)), key=lambda i: -abs(keep[i][2][-1]))
    picks = [i for i in order if keep[i][6]][:2] + [i for i in order if not keep[i][6]][:2]
    fig, ax = plt.subplots(len(picks), 3, figsize=(15, 3.2 * len(picks)))
    for row, i in enumerate(picks[:args.plots]):
        name, t0, along_err, trc, true_dist, jt, wrong, cls = keep[i]
        tt = np.arange(len(along_err)) * DT
        ax[row, 0].plot(tt, along_err)
        ax[row, 0].axhline(0, color="k", lw=0.5)
        for r in jt:
            ax[row, 0].axvline(r["t_ref"], color="g", lw=0.5)
        ax[row, 0].set_title(f"{name} t0={t0 / 10:.0f}s along-track error (m); wrong={wrong} ({cls})", fontsize=8)
        ax[row, 1].plot(tt, trc["dist"], label="dead-reckoned")
        ax[row, 1].plot(tt, true_dist, "k--", label="true")
        ax[row, 1].legend(fontsize=7)
        ax[row, 1].set_title("cumulative distance (m)", fontsize=8)
        ax[row, 2].plot(tt, trc["speed_scale"], label="EKF speed scale")
        if jt:
            ax2 = ax[row, 2].twinx()
            ax2.plot([r["t_ref"] for r in jt], [r["dt"] for r in jt], "ro", ms=3, label="dt_junction (s)")
            ax2.set_ylabel("dt_junction s", fontsize=7)
        ax[row, 2].set_title("speed scale / junction timing error", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "long_diag.png", dpi=110)
    print("saved", ARTIFACTS / "longitudinal_diag.json", FIGS / "long_diag.png")


if __name__ == "__main__":
    main()
