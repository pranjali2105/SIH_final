"""Controlled roundabout experiment: same outages, N seeds, particle filter with vs without ring mode.

`python -m idr.rb_seed_compare --seeds 3`. Only the ring logic (exit hypotheses + heading-driven arc + ring residuals)
is toggled; state-adaptive NHC and map-free circle stay off in both arms. Reports wrong-exit % per manoeuvre, post-exit
error (10 s after leaving the ring) and drift, as mean [min-max] over seeds.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from idr.dataset import HORIZON, load_drives
from idr.evaluate_mm import TFT_CFG, model_outputs
from idr.evaluate_roundabouts import LEADS, POST_EXIT, manoeuvre_key
from idr.map_matching import MMConfig, run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import drive_roadmap, planned_route
from idr.roundabout_labels import label_drive
from idr.train import pick_device

MANS = ["exit_1", "exit_2", "exit_3+", "full_circle", "u_turn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="+", default=["M", "Vw4", "Y1"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="rb_seed_compare")
    args = ap.parse_args()
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives(args.drives)
    rows = []
    for name in args.drives:
        d = drives[name]
        rm = drive_roadmap(name)
        _, trav = label_drive(d, rm)
        specs, meta = [], []
        for t in trav:
            for lead in LEADS:
                t0 = t["start"] - lead
                H = min(HORIZON, t["end"] - t0 + POST_EXIT)
                if t0 < 51 or t["end"] + POST_EXIT >= len(d.v) or not d.valid[t0 - 50:t0 + H].all() \
                        or d.run_end[t0] - t0 < H or d.v[t0 - 1] < 1:
                    continue
                specs.append({"drive": name, "t0": int(t0), "td": len(d.v) - 1, "route_mode": "none",
                              "seed": int(t0), "h_long": H})
                meta.append((t, H))
        mo = model_outputs(m, drives, specs, dev)
        for i, (sp, (t, H)) in enumerate(zip(specs, meta)):
            t0 = sp["t0"]
            u, ps = mo["u_hat"][i, :H], mo["p_stop"][i, :H]
            p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
            route = planned_route(rm, p0, psi0, d.pos[-1])
            k10 = min(H - 1, t["end"] - t0 + 100)
            after = d.pos[t["end"]:t["end"] + 300]
            gt = d.pos[t0:t0 + H]
            dist = np.cumsum(d.v[t0:t0 + H]) * 0.1
            for arm in ("pf", "ring"):
                for sd in range(args.seeds):
                    tr, _, _ = run_outage(rm, p0, psi0, v0, u[:, 0], u[:, 1], u[:, 2], p_stop=ps,
                                          nav_cfg=NavConfig(**TFT_CFG), seed=t0 * 7 + sd, route_edges=route,
                                          roundabout=False, ring_mode=(arm == "ring"), map_free_circle=False)
                    rows.append({"arm": arm, "seed": sd, "drive": name, "t0": t0, "man": manoeuvre_key(t),
                                 "wrong_exit": bool(np.min(np.hypot(*(after - tr[k10]).T)) > 15),
                                 "post_exit_err": float(np.hypot(*(tr[k10] - gt[k10]))),
                                 "drift": float(np.hypot(*(tr[-1] - gt[-1])) / max(dist[-1], 1) * 100)})
        print(name, "done", flush=True)
    out = {}
    for arm in ("pf", "ring"):
        out[arm] = {}
        for man in MANS + ["ALL"]:
            vals = {"wrong_exit": [], "post_exit_err": [], "drift": []}
            for sd in range(args.seeds):
                sel = [r for r in rows if r["arm"] == arm and r["seed"] == sd and (man == "ALL" or r["man"] == man)]
                if not sel:
                    continue
                vals["wrong_exit"].append(100 * np.mean([r["wrong_exit"] for r in sel]))
                vals["post_exit_err"].append(np.median([r["post_exit_err"] for r in sel]))
                vals["drift"].append(np.median([r["drift"] for r in sel]))
            if vals["wrong_exit"]:
                n = sum(1 for r in rows if r["arm"] == arm and r["seed"] == 0 and (man == "ALL" or r["man"] == man))
                out[arm][man] = {"n": n, **{k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
                                            for k, v in vals.items()}}
    (ARTIFACTS / f"{args.out}.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(f"\n{'arm':5s} {'manoeuvre':12s} {'n':>4s}  {'wrong-exit % mean [min-max]':28s} {'post-exit err m':16s} drift %")
    for man in MANS + ["ALL"]:
        for arm in ("pf", "ring"):
            if man in out[arm]:
                x = out[arm][man]
                print(f"{arm:5s} {man:12s} {x['n']:4d}  {x['wrong_exit'][0]:5.1f} [{x['wrong_exit'][1]:4.1f}-{x['wrong_exit'][2]:4.1f}]"
                      f"{'':12s} {x['post_exit_err'][0]:6.1f}{'':10s} {x['drift'][0]:5.1f}")


if __name__ == "__main__":
    main()
