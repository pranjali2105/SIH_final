"""Map-aided evaluation on held-out outages: `python -m idr.evaluate_mm --tag tft_ep2 --drives Y1 M`.

Variants on identical outages (stride 60 s): raw physics / raw+EKF / raw+EKF+map / TFT / TFT+EKF / TFT+EKF+map /
TFT+EKF+map+planned route / TFT+EKF+map with FIXED NHC covariance. Only the IMU (raw or TFT-corrected), the last GNSS
fix and OSM are used during the outage; GNSS truth is used for scoring only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader

from idr import physics
from idr.dataset import HORIZON, OutageDataset, load_drives
from idr.evaluate import make_specs
from idr.map_matching import run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS, STAT_COLS
from idr.osm_map import drive_roadmap, planned_route
from idr.train import pick_device, to_dev

FIGS = Path(__file__).resolve().parent / "figures"
RAW_CFG = dict(sig_w=0.086, sig_a=0.76)      # residual std of raw aligned phone IMU (training drives)
TFT_CFG = dict(sig_w=0.05, sig_a=0.5)


@torch.no_grad()
def model_outputs(model, drives, specs, dev):
    ds = OutageDataset(drives, sorted({s["drive"] for s in specs}), len(specs), mode="eval")
    ds.fixed = specs
    outs = []
    for batch in DataLoader(ds, batch_size=64):
        b = to_dev(batch, dev)
        o = model(b["seq"], b["static"], b["route"], b["u_raw"])
        outs.append({"u_hat": o["u_hat"].cpu().numpy(), "p_stop": o["p_stop"].cpu().numpy(),
                     "u_raw": b["u_raw"].cpu().numpy()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="+", default=["Y1", "M"])
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--stride", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="particle-filter seeds per outage (report mean and spread)")
    ap.add_argument("--core", action="store_true", help="only tft, tft_ekf_map, tft_ekf_map_route (faster)")
    ap.add_argument("--mm_kwargs", default="{}", help="JSON dict of extra run_outage kwargs for the map variants")
    ap.add_argument("--nav_kwargs", default="{}", help="JSON dict of NavConfig overrides for the TFT variants")
    ap.add_argument("--speed_cal", default=None, help="speed calibration tag/file (idr.speed_calibration)")
    ap.add_argument("--cal_kind", default=None, choices=[None, "affine", "isotonic"])
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--anchor", action="store_true")
    args = ap.parse_args()
    extra = json.loads(args.mm_kwargs)
    from idr.speed_calibration import apply_cal, load_cal
    cal = load_cal(args.speed_cal)
    if cal is not None and args.cal_kind:
        cal["use"] = args.cal_kind
    if cal is not None:
        cal["anchor"] = args.anchor
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    model = ResidualTFT(d=ck["args"]["d"], use_tft=not ck["args"]["no_tft"]).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives()
    H = args.horizon
    iv = STAT_COLS.index("a_up")

    rows = {}
    examples = []
    t_start = time.time()
    for name in args.drives:
        rm = drive_roadmap(name)
        d = drives[name]
        specs = make_specs(drives, [name], H, stride=args.stride)
        if args.limit:
            specs = specs[:args.limit]
        mo = model_outputs(model, drives, specs, dev)
        vib_all = np.clip(uniform_filter1d(d.feats[:, iv] ** 2, 10) ** 0.5 - 1.0, 0, 3)
        for i, sp in enumerate(specs):
            t0 = sp["t0"]
            gt = d.pos[t0:t0 + H]
            dist = np.cumsum(d.v[t0:t0 + H]) * 0.1
            # skip outages without road coverage (GT off-map)
            dd, _ = rm.nearest(gt[::10])
            if np.mean(dd < 15) < 0.9 or dist[-1] < 50:
                continue
            p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
            route = planned_route(rm, p0, psi0, d.pos[-1])
            vib = vib_all[t0:t0 + H]
            u_raw, u_hat, ps = mo["u_raw"][i, :H], mo["u_hat"][i, :H], mo["p_stop"][i, :H]
            res = {}
            # open-loop physics (no filter): same equations as training
            for key, u, pst in [("raw", u_raw, None), ("tft", u_hat, ps)]:
                tr = physics.integrate(torch.tensor(u[None, :, 0]), torch.tensor(u[None, :, 1]),
                                       torch.tensor([v0]), None if pst is None else torch.tensor(pst[None]))
                rel = tr["pos"][0].numpy()
                if key == "tft" and cal is not None:
                    vb = np.maximum(v0 + np.cumsum(u[:, 1]) * 0.1, 0.0)
                    spd = apply_cal(vb, cal, v0=v0) * (1 - pst)
                    dps = tr["d_psi"][0].numpy()
                    rel = np.cumsum(np.column_stack([spd * np.cos(dps), spd * np.sin(dps)]), 0) * 0.1
                f = np.array([np.sin(psi0), np.cos(psi0)])
                r = np.array([np.cos(psi0), -np.sin(psi0)])
                res[key] = p0 + rel[:, :1] * f + rel[:, 1:] * r
            common = dict(rm=rm, p0=p0, psi0=psi0, v0=v0, vib=vib)
            raw_cfg = NavConfig(**RAW_CFG)
            tft_cfg = NavConfig(**{**TFT_CFG, **json.loads(args.nav_kwargs)}, speed_cal=cal)
            tft_in = dict(w=u_hat[:, 0], a=u_hat[:, 1], a_lat=u_hat[:, 2], p_stop=ps,
                          tft_resid=np.linalg.norm(u_raw - u_hat, axis=1))
            # direction of travel at the end (for along/cross-track decomposition)
            g = gt[-1] - gt[max(0, H - 30)]
            g = g / np.linalg.norm(g) if np.linalg.norm(g) > 1 else np.array([np.sin(d.psi[t0 + H - 1]),
                                                                              np.cos(d.psi[t0 + H - 1])])
            n_ = np.array([g[1], -g[0]])
            path = d.pos[t0:t0 + H + 300]

            def record(key, tr, snap=None, st=None, seed=0):
                err = np.hypot(*(tr - gt).T)
                e = tr[-1] - gt[-1]
                rec = {"drive": name, "t0": t0, "seed": seed, "err_end": float(err[-1]), "dist": float(dist[-1]),
                       "err30": float(err[299]), "err10": float(err[99]), "err20": float(err[199]),
                       "along": float(abs(e @ g)), "cross": float(abs(e @ n_)), "stats": st}
                if snap is not None:
                    rec["wrong_road"] = bool(np.min(np.hypot(*(path - snap[-1]).T)) > 15.0)
                rows.setdefault(key, []).append(rec)

            record("tft", res["tft"])
            if not args.core:
                record("raw", res["raw"])
                record("raw_ekf", run_outage(w=u_raw[:, 0], a=u_raw[:, 1], a_lat=u_raw[:, 2], nav_cfg=raw_cfg,
                                             use_map=False, seed=t0, **common)[0])
                record("tft_ekf", run_outage(nav_cfg=tft_cfg, use_map=False, seed=t0, **tft_in, **common)[0])
            for sd in range(args.seeds):
                seed = t0 * 7 + sd
                if not args.core:
                    record("raw_ekf_map", run_outage(w=u_raw[:, 0], a=u_raw[:, 1], a_lat=u_raw[:, 2],
                                                     nav_cfg=raw_cfg, seed=seed, **common)[0], seed=sd)
                tr, snap, st = run_outage(nav_cfg=tft_cfg, seed=seed, **tft_in, **common, **extra)
                record("tft_ekf_map", tr, snap, st, seed=sd)
                tr_r, snap_r, st_r = run_outage(nav_cfg=tft_cfg, route_edges=route, seed=seed, **tft_in, **common,
                                                **extra)
                record("tft_ekf_map_route", tr_r, snap_r, st_r, seed=sd)
                res["tft_ekf_map_route"] = tr_r
            examples.append((name, t0, gt, res, rm))
        print(f"{name}: {len(rows.get('tft', []))} outages so far ({time.time() - t_start:.0f} s)", flush=True)

    def stats_of(lst):
        e = np.array([r["err_end"] for r in lst])
        di = np.array([r["dist"] for r in lst])
        drift = e / di * 100
        along = np.array([r["along"] for r in lst])
        worst = drift >= np.percentile(drift, 75)
        out_ = {"n": len(lst), "median_drift_pct": float(np.median(drift)),
                "p90_drift_pct": float(np.percentile(drift, 90)), "pct_under_10pct": float((drift < 10).mean() * 100),
                "median_err_end_m": float(np.median(e)), "median_dist_m": float(np.median(di)),
                "median_along_m": float(np.median(along)), "median_cross_m": float(np.median([r["cross"] for r in lst])),
                "worst25_along_m": float(np.median(along[worst])),
                "median_err_m_10_20_30s": [float(np.median([r[f"err{s_}"] for r in lst])) for s_ in (10, 20, 30)]}
        if "wrong_road" in lst[0]:
            out_["wrong_road_pct"] = float(np.mean([r["wrong_road"] for r in lst]) * 100)
        return out_

    summary = {}
    for key, lst in rows.items():
        seeds = sorted({r["seed"] for r in lst})
        per = [stats_of([r for r in lst if r["seed"] == sd]) for sd in seeds]
        agg = {"n": per[0]["n"], "seeds": len(seeds)}
        for k in per[0]:
            if k == "n":
                continue
            vals = np.array([p[k] for p in per], dtype=float)
            agg[k] = vals.mean(0).tolist() if vals.ndim > 1 else float(vals.mean())
            if vals.ndim == 1 and len(seeds) > 1:
                agg[k + "_range"] = [float(vals.min()), float(vals.max())]
        summary[key] = agg
    out = ARTIFACTS / f"mm_eval_{args.out_tag or args.tag}_{'_'.join(args.drives)}_{H // 10}s.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    rng_ = lambda v, k: (f" [{v[k + '_range'][0]:.1f}-{v[k + '_range'][1]:.1f}]" if k + "_range" in v else "")
    for k, v in summary.items():
        print(f"{k:22s} n={v['n']:4d} x{v['seeds']} | median drift {v['median_drift_pct']:5.1f}%{rng_(v, 'median_drift_pct')}"
              f"  p90 {v['p90_drift_pct']:5.1f}%  <10%: {v['pct_under_10pct']:5.1f}%"
              + (f"  wrong-road {v['wrong_road_pct']:4.1f}%{rng_(v, 'wrong_road_pct')}" if "wrong_road_pct" in v else "")
              + f"  along/cross {v['median_along_m']:.0f}/{v['median_cross_m']:.0f} m  worst25% along {v['worst25_along_m']:.0f} m",
              flush=True)

    # plots: 4 examples with the largest improvement + 1 failure
    FIGS.mkdir(exist_ok=True)
    route0 = [r for r in rows["tft_ekf_map_route"] if r["seed"] == 0]
    gain = [r0["err_end"] - r1["err_end"] for r0, r1 in zip(rows["tft"], route0)]
    order = np.argsort(gain)
    picks = list(order[-3:][::-1]) + [order[len(order) // 2], order[0]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 4.6))
    for ax, i in zip(axes, picks):
        name, t0, gt, res, rm = examples[i]
        c = gt.mean(0)
        span = max(np.ptp(gt, axis=0).max(), 200) * 0.8
        near = rm.tree.query_ball_point(c, span * 1.2)
        ax.plot(rm.pts[near, 0], rm.pts[near, 1], ".", color="0.85", ms=1)
        for key, lab, sty in [(None, "GNSS truth", "k-"), ("raw", "raw IMU", "C1:"), ("tft", "TFT", "C0--"),
                              ("tft_ekf_map_route", "TFT+NHC-EKF+map", "C2-")]:
            p = gt if key is None else res[key]
            ax.plot(p[:, 0], p[:, 1], sty, lw=1.6, label=lab)
        ax.plot(*gt[0], "go")
        ax.set_xlim(c[0] - span, c[0] + span)
        ax.set_ylim(c[1] - span, c[1] + span)
        ax.set_aspect("equal")
        e_t = rows["tft"][i]["err_end"]
        e_m = route0[i]["err_end"]
        ax.set_title(f"{name} t0={t0 / 10:.0f}s, {rows['tft'][i]['dist']:.0f} m\nTFT {e_t:.0f} m -> +map {e_m:.0f} m",
                     fontsize=9)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / f"mm_{args.tag}_{'_'.join(args.drives)}.png", dpi=120)
    print("saved", out, "and figure")


if __name__ == "__main__":
    main()
