"""Tail diagnosis of the map-aided pipeline (validation drives). `python -m idr.tail_diag --drives S2 Vtb5`.

For every 60 s outage (tft_ep2, validated defaults, seed 0) log, per second: dominant edge, its arc position, route
entropy, ESS, speed scale; derive
  t_along  = first time |signed along-track error| > 30 m (EKF projected on the true path)
  t_wrong  = first time the dominant particle road is off the true path (> 15 m from GNSS track within +-20 s)
and classify the worst-25% (by final |along-track|) + all wrong-road outages into
  1 speed-scale drift before junction   (along diverges first, then a wrong branch at a junction)
  2 missed / delayed junction           (right road, large along error; the junction crossing is late/early by > 3 s)
  3 wrong successor selected            (wrong edge first, starting at a junction, along still small)
  4 correct successor, wrong progress   (never off-road, large along error, junction timing ok)
  5 map geometry mismatch               (GNSS track > 15 m from any road for > 10% of the outage)
  6 temporary route ambiguity           (off-road for a while but ends on the true path)
  7 particle impoverishment             (wrong edge selected while route entropy ~ 0 / ESS tiny, and it persists)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch
from scipy.spatial import cKDTree

from idr.causal_calibration import add_cli, config_from_args, nav_cfg_for
from idr.route_input import add_cli as add_route_cli
from idr.route_input import config_from_args as route_config_from_args
from idr.dataset import load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG, model_outputs
from idr.longitudinal_diag import arc_projection, junction_times
from idr.map_matching import run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import drive_roadmap, planned_route
from idr.train import pick_device

DT = 0.1
H = 600
CLASSES = {1: "speed-scale drift before junction", 2: "missed/delayed junction", 3: "wrong successor selected",
           4: "correct successor, wrong progress", 5: "map geometry mismatch", 6: "temporary route ambiguity",
           7: "particle impoverishment"}


def edge_off_path(rm, e, gt_window):
    pts = rm.pts[rm.e_start[e]:rm.e_start[e] + rm.e_n[e]]
    d, _ = cKDTree(gt_window).query(pts)
    return float(np.min(d))


def classify(r):
    if r["map_mismatch_frac"] > 0.10:
        return 5
    if r["wrong_end"]:
        if r["min_entropy_at_wrong"] is not None and r["min_entropy_at_wrong"] < 0.05 and r["wrong_persist_s"] > 10:
            return 7
        if r["t_along"] is not None and (r["t_wrong"] is None or r["t_along"] < r["t_wrong"] - 1):
            return 1
        return 3
    if r["t_wrong"] is not None:
        return 6
    if r["max_abs_dt_junction"] is not None and r["max_abs_dt_junction"] > 3:
        return 2
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="+", default=["S2", "Vtb5"])
    ap.add_argument("--stride", type=int, default=600)
    ap.add_argument("--mm_overrides", default="{}", help="JSON MMConfig overrides (e.g. '{\"along_landmark\": true}')")
    ap.add_argument("--seeds", type=int, default=1, help="particle-filter seeds t0*7+s, s < seeds (tail classified per seed)")
    ap.add_argument("--out", default="tail_diag.json")
    ap.add_argument("--lookahead", default=None, help="JSON LookaheadConfig (idr/map_lookahead.py), off when omitted")
    add_cli(ap)
    add_route_cli(ap)
    args = ap.parse_args()
    calcfg = config_from_args(args)
    routecfg = route_config_from_args(args)
    mm_over = json.loads(args.mm_overrides)
    la_cfg = json.loads(args.lookahead) if args.lookahead else None
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives(args.drives)
    rows = []
    for name in args.drives:
        d = drives[name]
        rm = drive_roadmap(name)
        specs = make_specs(drives, [name], H, stride=args.stride)
        mo = model_outputs(m, drives, specs, dev, calibration=calcfg, route=routecfg)
        for i, sp in enumerate(specs):
            for sd in range(args.seeds):
                t0 = sp["t0"]
                gt = d.pos[t0:t0 + H]
                dist = np.cumsum(d.v[t0:t0 + H]) * DT
                if dist[-1] < 50:
                    continue
                p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
                u, ps = mo["u_hat"][i], mo["p_stop"][i]
                route = planned_route(rm, p0, psi0, d.pos[-1])
                tr, snap, st, trc = run_outage(rm, p0, psi0, v0, u[:, 0], u[:, 1], u[:, 2], p_stop=ps,
                                               nav_cfg=nav_cfg_for(mo["cal_meta"][i], NavConfig(**TFT_CFG), calcfg),
                                               seed=t0 * 7 + sd, route_edges=route,
                                               return_trace=True, mm_overrides=mm_over,
                                               tft_resid=np.linalg.norm(mo["u_raw"][i, :H] - u, axis=1),
                                               lookahead=la_cfg)
                path = d.pos[t0:t0 + H + 300]
                s_proj, lat, s_path = arc_projection(path, tr)
                along = s_proj - s_path[:H]
                t_along = next((k * DT for k in range(H) if abs(along[k]) > 30), None)
                mt = trc["mm_trace"]
                wrong_flags = []
                for x in mt:
                    kk = int(x["k"])
                    lo, hi = max(0, kk - 200), min(H, kk + 200)
                    wrong_flags.append(edge_off_path(rm, x["edge"], gt[lo:hi]) > 15.0)
                t_wrong = next((mt[j]["k"] * DT for j in range(len(mt)) if wrong_flags[j]), None)
                j_w = next((j for j in range(len(mt)) if wrong_flags[j]), None)
                persist = 0.0
                if j_w is not None:
                    run_ = 0
                    for f in wrong_flags[j_w:]:
                        if not f:
                            break
                        run_ += 1
                    persist = run_ * 1.0
                ent_at = min((mt[j]["entropy"] for j in range(max(0, (j_w or 0) - 3), (j_w or 0) + 1)), default=None) \
                    if j_w is not None else None
                dmap, _ = rm.nearest(gt[::10])
                jt = junction_times(rm, gt, mt)
                wrong_end = bool(np.min(np.hypot(*(path - snap[-1]).T)) > 15.0)
                ev = trc.get("along_events", [])
                r = {"drive": name, "t0": t0, "seed": sd, "calibration": mo["cal_meta"][i], "route": mo["route_meta"][i],
                     "conf_stats": {k: v for k, v in st.items() if k.startswith("conf") or k.startswith("alm")}, "along_trace_m": along[::10].round(1).tolist(),
                     "cross_end_m": float(abs(lat[H - 1])) if np.ndim(lat) else None,
                     "jct_dt_s": [float(x["dt"]) for x in jt],
                     "alm_events": len(ev), "alm_applied": sum(bool(e_["applied"]) for e_ in ev),
                     "alm_skips": dict(Counter(e_.get("skip") for e_ in ev if not e_["applied"])),
                     "alm_r_med": [e_.get("r_med") for e_ in ev if e_["applied"]],
                     "ess_prior_med": float(np.median([x["ess_prior"] for x in mt])) if mt else None,
                     "ess_post_med": float(np.median([x["ess_post"] for x in mt])) if mt else None,
                     "ess_resampled_med": float(np.median([x["ess"] for x in mt])) if mt else None,
                     "n_roads_med": float(np.median([x["n_roads"] for x in mt])) if mt else None,
                     "dist_m": float(dist[-1]), "along_end_m": float(along[-1]),
                     "abs_along_end_m": float(abs(along[-1])), "drift_pct": float(np.hypot(*(tr[-1] - gt[-1])) / dist[-1] * 100),
                     "t_along": t_along, "t_wrong": t_wrong, "wrong_end": wrong_end, "wrong_persist_s": persist,
                     "min_entropy_at_wrong": ent_at, "map_mismatch_frac": float(np.mean(dmap > 15)),
                     "max_abs_dt_junction": float(max(abs(x["dt"]) for x in jt)) if jt else None,
                     "n_junctions": len(jt), "speed_scale_end": float(trc["speed_scale"][-1]),
                     "planned_route_edges": len(route), "min_ess": float(min(x["ess"] for x in mt)) if mt else None}
                rows.append(r)
        print(name, "done", flush=True)
    q = np.percentile([r["abs_along_end_m"] for r in rows if r["seed"] == 0], 75)
    tail = [r for r in rows if r["abs_along_end_m"] >= q or r["wrong_end"]]
    for r in tail:
        r["class"] = classify(r)
    cnt = Counter(r["class"] for r in tail)
    both = [r for r in tail if r["t_along"] is not None and r["t_wrong"] is not None]
    lead = [r["t_wrong"] - r["t_along"] for r in both]
    summary = {"calibration_mode": calcfg.mode, "route_input_source": routecfg.source, "mm_overrides": mm_over, "seeds": args.seeds, "n_outages": len(rows), "tail_threshold_along_m": float(q), "n_tail": len(tail),
               "classes": {CLASSES[k]: {"n": v, "pct_of_tail": round(100 * v / len(tail), 1),
                                        "median_abs_along_end_m": float(np.median([r["abs_along_end_m"] for r in tail
                                                                                   if r["class"] == k]))}
                           for k, v in sorted(cnt.items())},
               "tail_with_both_times": len(both),
               "along_diverges_first_pct": float(np.mean([x > 1 for x in lead]) * 100) if lead else None,
               "wrong_edge_first_pct": float(np.mean([x < -1 for x in lead]) * 100) if lead else None,
               "median_t_wrong_minus_t_along_s": float(np.median(lead)) if lead else None}
    (ARTIFACTS / args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
