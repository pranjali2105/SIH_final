"""Speed-error slicing on held-out drives (analysis only). `python -m idr.speed_slices [--split val|test]`.

For 60 s outages (every 30 s) the open-loop TFT speed  v_hat = relu(v0 + sum a_hat*dt) * (1 - p_stop)  is compared with
the vehicle speed. Every 10 Hz sample gets one label per slicing dimension:
  speed regime, motion phase (single label, priority ordered so phases partition the samples), road class, curvature,
  signal quality (tertiles), drive and driver/vehicle group.
Per slice: sample count, number of 60 s outages containing it, MAE, signed bias, RMSE, p95 |e|, and the slice's
integrated distance contribution  sum(e_v*dt)  per outage. Per outage: distance error after 10/30/60 s and open-loop
along-track error. Per drive: speed scale lambda = sum(v_hat*v)/sum(v_hat^2) and lag tau minimising MSE(v_hat(t+tau), v).
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

from idr.data import load_all
from idr.dataset import TEST, VAL, load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import model_outputs
from idr.longitudinal_diag import arc_projection
from idr.model import E_SCALE, ResidualTFT
from idr.normalise import ARTIFACTS, STAT_COLS
from idr.osm_map import OSM_DIR, drive_roadmap
from idr.roundabout_labels import label_drive
from idr.train import pick_device

FIGS = Path(__file__).resolve().parent / "figures"
DT = 0.1
H = 600
SPEED_BINS = [(0, 0.5, "stationary 0-0.5"), (0.5, 3, "very low 0.5-3"), (3, 8, "urban low 3-8"),
              (8, 15, "urban medium 8-15"), (15, 99, "high >15")]
PHASES = ["roundabout entry", "roundabout circulation", "roundabout exit", "junction crossing", "junction approach",
          "turning", "post-turn recovery", "stationary", "start from rest", "stop-and-go", "braking", "acceleration",
          "constant speed"]
ROAD_CLASS = {4.75: "motorway", 4.5: "trunk", 4.0: "primary", 3.75: "secondary", 3.5: "tertiary", 3.0: "minor/residential"}
GROUPS = ["Vfa", "Vta", "Vtb", "Vw", "S", "M", "Y"]


def group_of(name):
    return next(g for g in GROUPS if name.startswith(g))


def phase_labels(d, t0, rb_states, rm, s_gt_drive, node_arcs):
    """Single motion-phase label per sample of the outage (priority order = PHASES order)."""
    sl = slice(t0, t0 + H)
    v = d.v[sl]
    acc = d.u_true[sl, 1]
    yaw = np.abs(d.u_true[sl, 0])
    lab = np.full(H, "constant speed", dtype=object)
    stopped = d.v < 0.5
    # stop-and-go: >= 2 stop onsets within +-15 s
    onset = np.flatnonzero(np.diff(stopped.astype(int)) == 1) + 1
    for k in range(H):
        t = t0 + k
        if ((onset > t - 150) & (onset < t + 150)).sum() >= 2:
            lab[k] = "stop-and-go"
    lab[acc > 0.5] = np.where(lab[acc > 0.5] == "stop-and-go", "stop-and-go", "acceleration")
    lab[acc < -0.5] = np.where(lab[acc < -0.5] == "stop-and-go", "stop-and-go", "braking")
    # start from rest: within 5 s after a stop, speed < 5 m/s
    for k in range(H):
        t = t0 + k
        if v[k] < 5 and v[k] >= 0.5 and stopped[max(0, t - 50):t].any():
            lab[k] = "start from rest"
    lab[v < 0.5] = "stationary"
    # turning and post-turn recovery
    turning = yaw > 0.1
    post = np.zeros(H, bool)
    for k in np.flatnonzero(turning):
        post[k + 1:k + 101] = True
    lab[post & ~turning] = "post-turn recovery"
    lab[turning] = "turning"
    # junctions on the true path (arc distance to the next junction node)
    s_now = s_gt_drive[t0:t0 + H]
    if len(node_arcs):
        for k in range(H):
            ahead = node_arcs - s_now[k]
            dmin = np.min(np.abs(ahead))
            if dmin <= 10:
                lab[k] = "junction crossing"
            else:
                nxt = ahead[ahead > 0]
                if len(nxt) and nxt.min() <= 30:
                    lab[k] = "junction approach"
    if rb_states is not None:
        st = rb_states[sl]
        lab[st == 2] = "roundabout entry"
        lab[st == 3] = "roundabout circulation"
        lab[st == 4] = "roundabout exit"
    return lab


def stats(e, dt_contrib=None):
    a = np.abs(e)
    out = {"count": int(len(e)), "MAE": float(a.mean()), "bias": float(e.mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
           "p95_abs": float(np.percentile(a, 95))}
    if dt_contrib is not None and len(dt_contrib):
        c = np.array(dt_contrib)
        out.update({"outages": int(len(c)), "dist_contrib_mean_m": float(c.mean()),
                    "dist_contrib_abs_mean_m": float(np.abs(c).mean()), "dist_contrib_abs_p90_m": float(np.percentile(np.abs(c), 90))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--stride", type=int, default=300)
    ap.add_argument("--drives", nargs="*", default=None, help="override the drive list (e.g. clean val: S2 Vtb5)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed_cal", default=None)
    ap.add_argument("--cal_kind", default=None)
    ap.add_argument("--anchor", action="store_true")
    args = ap.parse_args()
    names = args.drives or (VAL if args.split == "val" else TEST)
    from idr.speed_calibration import apply_cal, load_cal
    cal = load_cal(args.speed_cal)
    if cal is not None and args.cal_kind:
        cal["use"] = args.cal_kind
    if cal is not None:
        cal["anchor"] = args.anchor
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, stats_norm = load_drives(names)
    raw_all = load_all()
    iv = STAT_COLS.index("a_up")

    S = {k: [] for k in ["e", "v", "vhat", "acc", "yaw", "outage", "drive", "group", "speed", "phase", "road", "curv",
                         "q_acc", "q_jerk", "q_gyro", "q_vib", "q_resid", "q_nhc", "q_ts"]}
    outages = []
    tracks = []
    for name in names:
        d = drives[name]
        has_map = (OSM_DIR / f"{name}.pkl").exists()
        rm = drive_roadmap(name) if has_map else None
        rb_states = label_drive(d, rm)[0] if has_map else None
        s_gt = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(d.pos, axis=0).T))])
        node_arcs = np.array([])
        if has_map:
            node_xy = {}
            for e_ in range(len(rm.e_u)):
                node_xy.setdefault(int(rm.e_v[e_]), rm.pts[rm.e_start[e_] + rm.e_n[e_] - 1])
            nodes = [n_ for n_, deg in rm.node_deg.items() if deg >= 6 and n_ in node_xy]
            if nodes:
                from scipy.spatial import cKDTree
                xy = np.array([node_xy[n_] for n_ in nodes])
                dd, jj = cKDTree(d.pos).query(xy)
                node_arcs = np.sort(s_gt[jj[dd < 8.0]])
        ts = raw_all[name]["s_t_ms"].to_numpy()
        ts_irreg = np.r_[False, np.abs(np.diff(ts) - 100) > 20]
        specs = make_specs(drives, [name], H, stride=args.stride)
        mo = model_outputs(m, drives, specs, dev)
        for i, sp in enumerate(specs):
            t0 = sp["t0"]
            v = d.v[t0:t0 + H]
            if np.sum(v) * DT < 50:
                continue
            u, ps, ur = mo["u_hat"][i], mo["p_stop"][i], mo["u_raw"][i]
            v0 = float(d.v[t0 - 1])
            vb = np.maximum(v0 + np.cumsum(u[:, 1]) * DT, 0)
            vhat = (apply_cal(vb, cal, v0=v0) if cal is not None else vb) * (1 - ps)
            e = vhat - v
            oid = len(outages)
            # open-loop track for along-track error
            psi = d.psi[t0 - 1] - np.cumsum(u[:, 0]) * DT
            tr = d.pos[t0 - 1] + np.cumsum(np.column_stack([vhat * np.sin(psi), vhat * np.cos(psi)]), 0) * DT
            path = d.pos[t0:t0 + H + 300]
            s_proj, _, s_path = arc_projection(path, tr)
            along = s_proj - s_path[:H]
            dist = np.cumsum(v) * DT
            es = np.cumsum(e) * DT
            outages.append({"drive": name, "group": group_of(name), "t0": int(t0), "mean_speed": float(v.mean()),
                            "speed_bias": float(e.mean()), "speed_RMSE": float(np.sqrt((e ** 2).mean())),
                            "e_s_10": float(es[99]), "e_s_30": float(es[299]), "e_s_60": float(es[-1]),
                            "dist_60": float(dist[-1]), "dist_err_pct": float(es[-1] / dist[-1] * 100),
                            "along_end_m": float(along[-1]),
                            "max_acc": float(np.max(np.abs(d.u_true[t0:t0 + H, 1]))),
                            "max_jerk": float(np.max(np.abs(np.diff(d.u_true[t0:t0 + H, 1]))) / DT),
                            "mean_abs_yaw": float(np.mean(np.abs(d.u_true[t0:t0 + H, 0]))),
                            "roundabout_steps": int(np.sum(rb_states[t0:t0 + H] >= 2)) if rb_states is not None else 0,
                            "junctions": int(np.sum((node_arcs >= s_gt[t0]) & (node_arcs <= s_gt[t0 + H - 1]))),
                            "resid_norm": float(np.mean(np.linalg.norm((ur - u) / E_SCALE.numpy(), axis=1))),
                            "nhc_proxy": float(np.mean(np.abs(u[:, 2] - vhat * u[:, 0])))})
            tracks.append((name, t0, e, es, v, vhat))
            # per-sample labels
            S["e"].append(e); S["v"].append(v); S["vhat"].append(vhat)
            S["acc"].append(d.u_true[t0:t0 + H, 1]); S["yaw"].append(d.u_true[t0:t0 + H, 0])
            S["outage"].append(np.full(H, oid)); S["drive"].append(np.full(H, name, dtype=object))
            S["group"].append(np.full(H, group_of(name), dtype=object))
            S["speed"].append(np.array([next(lab for lo, hi, lab in SPEED_BINS if lo <= x < hi) for x in v], dtype=object))
            S["phase"].append(phase_labels(d, t0, rb_states, rm, s_gt, node_arcs))
            if has_map:
                _, j = rm.tree.query(d.pos[t0:t0 + H])
                cls = [ROAD_CLASS.get(round(float(rm.e_width[rm.edge[jj_]]), 2), "minor/residential") for jj_ in j]
                S["road"].append(np.array(cls, dtype=object))
            else:
                S["road"].append(np.full(H, "no map", dtype=object))
            kap = np.abs(d.u_true[t0:t0 + H, 0]) / np.maximum(v, 1.0)
            S["curv"].append(np.where(kap < 0.005, "straight |k|<0.005", np.where(kap < 0.02, "gentle 0.005-0.02",
                                                                                   "sharp >=0.02")).astype(object))
            a2 = ur[:, 1:3]
            S["q_acc"].append(np.linalg.norm(a2, axis=1))
            S["q_jerk"].append(np.r_[0, np.linalg.norm(np.diff(a2, axis=0), axis=1)] / DT)
            S["q_gyro"].append(np.abs(ur[:, 0]))
            vib = d.feats[t0:t0 + H, iv]
            S["q_vib"].append(np.sqrt(np.convolve(vib ** 2, np.ones(10) / 10, mode="same")))
            S["q_resid"].append(np.linalg.norm((ur - u) / E_SCALE.numpy(), axis=1))
            S["q_nhc"].append(np.abs(u[:, 2] - vhat * u[:, 0]))
            S["q_ts"].append(np.where(ts_irreg[t0:t0 + H], "irregular timestamp", "regular timestamp").astype(object))
        print(f"{name}: {sum(o['drive'] == name for o in outages)} outages", flush=True)

    A = {k: np.concatenate(v_) for k, v_ in S.items()}
    for q in ["q_acc", "q_jerk", "q_gyro", "q_vib", "q_resid", "q_nhc"]:
        t1, t2 = np.percentile(A[q], [33.3, 66.7])
        A[q] = np.where(A[q] < t1, "low", np.where(A[q] < t2, "mid", "high")).astype(object)
    n_out = len(outages)

    def slice_table(dim):
        rows = {}
        for c in sorted(set(A[dim])):
            msk = A[dim] == c
            contrib = [float(A["e"][msk & (A["outage"] == o)].sum() * DT) for o in np.unique(A["outage"][msk])]
            rows[c] = stats(A["e"][msk], contrib)
        return rows

    dims = {"speed regime": "speed", "motion phase": "phase", "road class": "road", "curvature": "curv",
            "acc magnitude": "q_acc", "jerk": "q_jerk", "gyro magnitude": "q_gyro", "vibration": "q_vib",
            "TFT residual norm": "q_resid", "NHC proxy |a_lat - v*w|": "q_nhc", "timestamps": "q_ts",
            "drive": "drive", "group": "group"}
    report = {"split": args.split, "n_outages": n_out, "n_samples": int(len(A["e"])), "overall": stats(A["e"]),
              "slices": {k: slice_table(v_) for k, v_ in dims.items()}}

    # per drive: speed scale and lag
    per_drive = {}
    for name in names:
        ids = [k for k, o in enumerate(outages) if o["drive"] == name]
        if not ids:
            continue
        vh = np.concatenate([tracks[k][5] for k in ids])
        vt = np.concatenate([tracks[k][4] for k in ids])
        lam = float(np.sum(vh * vt) / np.sum(vh ** 2))
        best = (0, np.inf)
        for tau in range(-30, 31):
            mse = []
            for k in ids:
                a, b = tracks[k][5], tracks[k][4]
                if tau >= 0:
                    mse.append(np.mean((a[tau:] - b[:H - tau]) ** 2))
                else:
                    mse.append(np.mean((a[:tau] - b[-tau:]) ** 2))
            if np.mean(mse) < best[1]:
                best = (tau, float(np.mean(mse)))
        de = np.array([outages[k]["dist_err_pct"] for k in ids])
        per_drive[name] = {"outages": len(ids), "speed_scale_lambda": lam, "lag_s": best[0] * DT,
                           "median_abs_dist_err_pct": float(np.median(np.abs(de))),
                           "median_signed_dist_err_pct": float(np.median(de))}
    report["per_drive"] = per_drive
    de_all = np.array([o["dist_err_pct"] for o in outages])
    report["outage_level"] = {"median_abs_dist_err_60s_pct": float(np.median(np.abs(de_all))),
                              "median_signed_dist_err_60s_pct": float(np.median(de_all)),
                              "median_abs_e_s_m": {f"{t}s": float(np.median(np.abs([o[f'e_s_{t}'] for o in outages])))
                                                   for t in (10, 30, 60)},
                              "median_abs_along_end_m": float(np.median(np.abs([o["along_end_m"] for o in outages])))}
    order = np.argsort([-abs(o["e_s_60"]) for o in outages])
    report["worst_25pct"] = [outages[k] for k in order[:max(1, n_out // 4)]]

    # sanity: per-outage phase contributions sum to e_s(60)
    k0 = 0
    tot = sum(A["e"][(A["outage"] == k0) & (A["phase"] == p)].sum() * DT for p in set(A["phase"][A["outage"] == k0]))
    assert abs(tot - outages[k0]["e_s_60"]) < 1e-6 * max(1, abs(tot)) + 1e-6
    assert sum(r["count"] for r in report["slices"]["motion phase"].values()) == len(A["e"])

    out = ARTIFACTS / f"{args.out or 'speed_slices_' + args.split}.json"
    out.write_text(json.dumps(report, indent=1, default=str))

    # markdown summary
    print(f"\n{args.split}: {n_out} outages x 60 s, {len(A['e'])} samples; overall speed MAE {report['overall']['MAE']:.2f} "
          f"bias {report['overall']['bias']:+.2f} RMSE {report['overall']['RMSE']:.2f} m/s; median |distance error| "
          f"{report['outage_level']['median_abs_dist_err_60s_pct']:.1f}% (signed {report['outage_level']['median_signed_dist_err_60s_pct']:+.1f}%)")
    for dim in ["speed regime", "motion phase", "road class", "curvature", "acc magnitude", "jerk", "gyro magnitude",
                "vibration", "TFT residual norm", "NHC proxy |a_lat - v*w|", "timestamps", "group", "drive"]:
        rows = report["slices"][dim]
        print(f"\n| {dim} | samples | outages | MAE m/s | bias m/s | RMSE m/s | p95 m/s | dist contrib mean (abs) m |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c, r in sorted(rows.items(), key=lambda kv: -kv[1].get("dist_contrib_abs_mean_m", 0)):
            print(f"| {c} | {r['count']} | {r.get('outages', 0)} | {r['MAE']:.2f} | {r['bias']:+.2f} | {r['RMSE']:.2f} | "
                  f"{r['p95_abs']:.2f} | {r.get('dist_contrib_mean_m', 0):+.1f} ({r.get('dist_contrib_abs_mean_m', 0):.1f}) |")
    print("\n| drive | outages | speed scale lambda | lag s | median abs dist err % | median signed % |")
    print("|---|---:|---:|---:|---:|---:|")
    for n_, r in per_drive.items():
        print(f"| {n_} | {r['outages']} | {r['speed_scale_lambda']:.3f} | {r['lag_s']:+.1f} | "
              f"{r['median_abs_dist_err_pct']:.1f} | {r['median_signed_dist_err_pct']:+.1f} |")

    # plots
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 5, figsize=(24, 9))
    ax = ax.ravel()
    sub = np.random.default_rng(0).choice(len(A["e"]), min(40000, len(A["e"])), replace=False)
    for dn in names:
        msk = A["drive"][sub] == dn
        ax[0].scatter(A["v"][sub][msk], A["vhat"][sub][msk], s=1, alpha=0.3, label=dn)
    ax[0].plot([0, 35], [0, 35], "k--"); ax[0].set_xlabel("GT speed"); ax[0].set_ylabel("predicted"); ax[0].legend(markerscale=8)
    ax[0].set_title("1 predicted vs GT speed")
    for (dn, t0, e, es, v, vh) in [tracks[k] for k in order[:3]] + [tracks[k] for k in order[len(order) // 2:len(order) // 2 + 2]]:
        ax[1].plot(np.arange(H) * DT, e, lw=0.8, label=f"{dn}@{t0 // 10}s")
        ax[2].plot(np.arange(H) * DT, es, lw=0.8)
    ax[1].set_title("2 signed speed error (3 worst + 2 median)"); ax[1].legend(fontsize=6)
    ax[2].set_title("3 cumulative distance error (m)")
    for k_, (x, lab) in enumerate([(A["v"], "GT speed"), (A["acc"], "GT accel"), (np.abs(A["yaw"]), "|yaw rate|")]):
        a_ = ax[3 + k_]
        bins = np.percentile(x, np.linspace(0, 100, 21))
        idx = np.clip(np.digitize(x, bins) - 1, 0, 19)
        med = [np.median(A["e"][idx == b]) for b in range(20)]
        q1 = [np.percentile(A["e"][idx == b], 25) for b in range(20)]
        q3 = [np.percentile(A["e"][idx == b], 75) for b in range(20)]
        cx = [(bins[b] + bins[b + 1]) / 2 for b in range(20)]
        a_.plot(cx, med, "o-"); a_.fill_between(cx, q1, q3, alpha=0.3); a_.axhline(0, color="k", lw=0.5)
        a_.set_title(f"{4 + k_} speed error vs {lab}")
    ph = [p for p in PHASES if p in set(A["phase"])]
    ax[6].boxplot([A["e"][A["phase"] == p] for p in ph], showfliers=False)
    ax[6].set_xticks(range(1, len(ph) + 1)); ax[6].set_xticklabels(ph, rotation=70, fontsize=7)
    ax[6].set_title("7 speed error per manoeuvre"); ax[6].axhline(0, color="k", lw=0.5)
    for k_, (q, lab) in enumerate([("q_resid", "TFT residual norm tertile"), ("q_nhc", "NHC proxy tertile")]):
        cats = ["low", "mid", "high"]
        ax[7 + k_].boxplot([A["e"][A[q] == c] for c in cats], showfliers=False)
        ax[7 + k_].set_xticks([1, 2, 3]); ax[7 + k_].set_xticklabels(cats)
        ax[7 + k_].set_title(f"{8 + k_} speed error vs {lab}"); ax[7 + k_].axhline(0, color="k", lw=0.5)
    # 10: error aligned on junction crossings
    segs = []
    for o_id in range(n_out):
        m_ = A["outage"] == o_id
        ph_o, e_o = A["phase"][m_], A["e"][m_]
        starts = np.flatnonzero((ph_o == "junction crossing") & np.r_[True, ph_o[:-1] != "junction crossing"])
        for s_ in starts:
            if 100 <= s_ < H - 100:
                segs.append(e_o[s_ - 100:s_ + 100])
    if segs:
        segs = np.array(segs)
        tt = np.arange(-100, 100) * DT
        ax[9].plot(tt, np.median(segs, 0)); ax[9].fill_between(tt, np.percentile(segs, 25, 0), np.percentile(segs, 75, 0), alpha=0.3)
        ax[9].axvline(0, color="k", lw=0.5); ax[9].axhline(0, color="k", lw=0.5)
    ax[9].set_title(f"10 speed error around junction crossings (n={len(segs)})")
    fig.tight_layout()
    fig.savefig(FIGS / f"{args.out or 'speed_slices_' + args.split}.png", dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
