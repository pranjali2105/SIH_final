"""Per-manoeuvre roundabout evaluation: `python -m idr.evaluate_roundabouts --tag tft_ep2`.

For every OSM-labelled roundabout traversal (valid phone IMU) in the test + scenario drives, outages start 5 s and 15 s
before ring entry and run until 20 s after leaving the ring (max 60 s). Variants on identical outages:
  pf           particle filter, roundabout mode OFF
  pf_rb        roundabout mode ON (exit hypotheses, ring residuals, soft radial, state-adaptive NHC)
  untagged_rb  map has the roads but no roundabout tags (rings removed) + map-free circle mode
  no_map       no map at all, map-free circle mode only
Metrics per manoeuvre (1st/2nd/3rd+ exit, straight, left/right, u_turn, full_circle): wrong-exit rate (EKF track >15 m
from the path actually driven, 10 s after exit), error 10 s after exit, drift % at the end.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from idr.dataset import HORIZON, load_drives, split_names
from idr.evaluate_mm import TFT_CFG, model_outputs
from idr.map_matching import run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import OSM_DIR, drive_roadmap, planned_route
from idr.roundabout_labels import label_drive
from idr.train import pick_device

FIGS = Path(__file__).resolve().parent / "figures"
POST_EXIT = 200
LEADS = (50, 150)


def untagged(rm):
    r2 = copy.copy(rm)
    r2.rings = []
    r2.e_ring = np.full_like(rm.e_ring, -1)
    return r2


def manoeuvre_key(t):
    if t["manoeuvre"] in ("full_circle", "u_turn"):
        return t["manoeuvre"]
    return f"exit_{min(t['exit_no'], 3)}{'+' if t['exit_no'] >= 3 else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="*", default=None)
    ap.add_argument("--plots", type=int, default=3)
    args = ap.parse_args()
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    model = ResidualTFT(d=ck["args"]["d"], use_tft=not ck["args"]["no_tft"]).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives()
    sp = split_names(list(drives))
    names = args.drives or [n for n in sp["test"] + sp["scenario"] if (OSM_DIR / f"{n}.pkl").exists()]

    rows, traces, det_pairs = [], [], []
    for name in names:
        rm = drive_roadmap(name)
        d = drives[name]
        states_all, trav = label_drive(d, rm)
        specs, meta = [], []
        for t in trav:
            for lead in LEADS:
                t0 = t["start"] - lead
                H = min(HORIZON, t["end"] - t0 + POST_EXIT)
                if t0 < 51 or t["end"] + POST_EXIT >= len(d.v) or not d.valid[t0 - 50:t0 + H].all() \
                        or d.run_end[t0] - t0 < H or d.v[t0 - 1] < 1.0:
                    continue
                specs.append({"drive": name, "t0": int(t0), "td": len(d.v) - 1, "route_mode": "none",
                              "seed": int(t0), "h_long": H})
                meta.append((t, H, lead))
        if not specs:
            print(f"{name}: no usable traversals")
            continue
        mo = model_outputs(model, drives, specs, dev)
        rm_u = untagged(rm)
        for i, (sp_, (t, H, lead)) in enumerate(zip(specs, meta)):
            t0 = sp_["t0"]
            gt = d.pos[t0:t0 + H]
            dist = np.cumsum(d.v[t0:t0 + H]) * 0.1
            p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
            u, ps = mo["u_hat"][i, :H], mo["p_stop"][i, :H]
            kw = dict(p0=p0, psi0=psi0, v0=v0, w=u[:, 0], a=u[:, 1], a_lat=u[:, 2], p_stop=ps,
                      nav_cfg=NavConfig(**TFT_CFG), seed=t0)
            route = planned_route(rm, p0, psi0, d.pos[-1])
            res = {}
            res["pf"], _, _, tr_pf = run_outage(rm, roundabout=False, map_free_circle=False, route_edges=route,
                                                detect=True, return_trace=True, **kw)
            lab = states_all[t0:t0 + H]
            for j, st_pred in enumerate(tr_pf["states"]):
                k = 10 * j + 9
                if k < len(lab):
                    det_pairs.append((int(lab[k]), int(st_pred)))
            res["pf_rb"], _, st, tr = run_outage(rm, roundabout=True, route_edges=route, return_trace=True, **kw)
            res["untagged_rb"], _, _ = run_outage(rm_u, roundabout=True, map_free_circle=True, **kw)
            res["no_map"], _, _ = run_outage(None, use_map=False, roundabout=True, map_free_circle=True, **kw)
            k_exit = t["end"] - t0
            k10 = min(H - 1, k_exit + 100)
            path_after = d.pos[t["end"]:t["end"] + POST_EXIT + 100]
            for key, trk in res.items():
                off = float(np.min(np.hypot(*(path_after - trk[k10]).T)))
                rows.append({"drive": name, "t0": t0, "lead_s": lead / 10, "manoeuvre": manoeuvre_key(t),
                             "turn_deg": t["turn_deg"], "exit_no": t["exit_no"], "radius": t["radius"],
                             "variant": key, "wrong_exit": bool(off > 15.0),
                             "err_10s_after_exit": float(np.hypot(*(trk[k10] - gt[k10]))),
                             "drift_pct": float(np.hypot(*(trk[-1] - gt[-1])) / max(dist[-1], 1.0) * 100)})
            traces.append((name, t0, t, gt, res, tr, u, rm, k_exit))
        print(f"{name}: {len(specs)} roundabout outages", flush=True)

    # --- summary tables
    variants = ["pf", "pf_rb", "untagged_rb", "no_map"]
    mans = sorted({r["manoeuvre"] for r in rows})
    summary = {}
    for v in variants:
        summary[v] = {}
        for m in mans + ["ALL"]:
            sel = [r for r in rows if r["variant"] == v and (m == "ALL" or r["manoeuvre"] == m)]
            if not sel:
                continue
            summary[v][m] = {"n": len(sel), "wrong_exit_pct": 100 * float(np.mean([r["wrong_exit"] for r in sel])),
                             "median_err_10s_after_exit_m": float(np.median([r["err_10s_after_exit"] for r in sel])),
                             "median_drift_pct": float(np.median([r["drift_pct"] for r in sel]))}
    from idr.roundabout_detector import per_class_f1
    yl, yp = np.array([a for a, _ in det_pairs]), np.array([b for _, b in det_pairs])
    det_online = per_class_f1(yl, yp) if len(yl) else {}
    out = {"summary": summary, "rows": rows, "detector_online_rules_map": det_online}
    det = ARTIFACTS / "rb_detector_report.json"
    if det.exists():
        out["detector"] = json.loads(det.read_text())
    (ARTIFACTS / f"rb_eval_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\n{'variant':12s} {'manoeuvre':12s} {'n':>4s} {'wrong-exit%':>11s} {'err 10s after exit':>19s} {'drift%':>7s}")
    for v in variants:
        for m, x in summary[v].items():
            print(f"{v:12s} {m:12s} {x['n']:4d} {x['wrong_exit_pct']:11.1f} {x['median_err_10s_after_exit_m']:17.1f} m "
                  f"{x['median_drift_pct']:7.1f}")

    # --- debug figures: correct exit, wrong exit, full circle / largest turn
    FIGS.mkdir(exist_ok=True)
    rb_rows = [r for r in rows if r["variant"] == "pf_rb"]
    picks = []
    ok = [i for i, r in enumerate(rb_rows) if not r["wrong_exit"]]
    bad = [i for i, r in enumerate(rb_rows) if r["wrong_exit"]]
    circ = sorted(range(len(rb_rows)), key=lambda i: -abs(rb_rows[i]["turn_deg"]))
    for pool in (ok, bad, circ):
        for i in pool:
            if i not in picks:
                picks.append(i)
                break
    for j, i in enumerate(picks[:args.plots]):
        name, t0, t, gt, res, tr, u, rm, k_exit = traces[i]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        a = ax[0, 0]
        ring = rm.rings[t["ring"]]
        near = rm.tree.query_ball_point([ring["cx"], ring["cy"]], 250)
        a.plot(rm.pts[near, 0], rm.pts[near, 1], ".", color="0.85", ms=1)
        a.plot(gt[:, 0], gt[:, 1], "k-", label="GNSS truth")
        for key, sty in [("pf", "C1--"), ("pf_rb", "C2-"), ("no_map", "C4:")]:
            a.plot(res[key][:, 0], res[key][:, 1], sty, label=key)
        a.set_aspect("equal"); a.legend(fontsize=7)
        a.set_title(f"{name} {t['manoeuvre']} exit {t['exit_no']}/{t['n_exits']} r={t['radius']:.0f} m")
        tt = np.arange(len(u)) * 0.1
        ax[0, 1].plot(tt, u[:, 0], label="yaw rate (TFT)")
        ax[0, 1].plot(tt, np.unwrap(-np.cumsum(u[:, 0]) * 0.1), label="unwrapped dpsi")
        ax[0, 1].axvspan(t["start"] - t0 and (t["start"] - t0) / 10, k_exit / 10, color="y", alpha=0.2)
        ax[0, 1].legend(fontsize=7); ax[0, 1].set_title("yaw")
        v = np.maximum(np.cumsum(u[:, 1]) * 0.1 + 0, 0)
        ax[0, 2].plot(tt, -u[:, 0] / (np.abs(v) + 5.0), label="kappa (approx)")
        ax[0, 2].axhline(1 / ring["r"], color="k", ls=":", label="1/r ring")
        ax[0, 2].legend(fontsize=7); ax[0, 2].set_title("curvature")
        mt = tr["mm_trace"]
        ks = np.array([x["k_now"] for x in mt]) / 10
        dc = [np.hypot(res["pf_rb"][min(x["k_now"], len(gt) - 1)][0] - ring["cx"],
                       res["pf_rb"][min(x["k_now"], len(gt) - 1)][1] - ring["cy"]) for x in mt]
        ax[1, 0].plot(ks, dc, label="EKF dist to ring center")
        ax[1, 0].axhline(ring["r"], color="k", ls=":")
        ax[1, 0].set_title("distance to center"); ax[1, 0].legend(fontsize=7)
        exits = sorted({e for x in mt for e in x["exit_probs"]})
        for e in exits:
            ax[1, 1].plot(ks, [x["exit_probs"].get(e, 0) for x in mt], label=f"exit edge {e}")
        ax[1, 1].set_title("exit hypothesis probabilities"); ax[1, 1].legend(fontsize=6)
        ax[1, 2].step(np.arange(len(tr["states"])), tr["states"], where="post")
        ax[1, 2].set_yticks(range(5)); ax[1, 2].set_yticklabels(["straight", "turn", "entry", "circ", "exit"])
        ax[1, 2].set_title("detector state (1 Hz)")
        fig.tight_layout()
        fig.savefig(FIGS / f"rb_{args.tag}_{j}.png", dpi=110)
        plt.close(fig)
    if det_online:
        print("\nonline rule detector (map) on these outages, per state P/R/F1:",
              {k: (v["precision"], v["recall"], v["f1"]) for k, v in det_online.items() if isinstance(v, dict)},
              " macro-F1", det_online["macro_f1"])
    print("saved", ARTIFACTS / f"rb_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
