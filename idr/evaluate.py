"""Evaluate a trained residual model on held-out drives: `python -m idr.evaluate --tag tft`.

Outages start every 30 s through each test drive (fully valid phone data required). The destination is the drive's
real end point (the trip's destination). Writes idr/artifacts/<tag>_eval.json and trajectory plots to idr/figures/.
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
from torch.utils.data import DataLoader

from idr import physics
from idr.dataset import OutageDataset, load_drives, split_names
from idr.losses import angle_weights, cos_angle, true_velocity
from idr.model import ResidualTFT
from idr.normalise import ARTIFACTS
from idr.train import pick_device, to_dev

FIGS = Path(__file__).resolve().parent / "figures"


def make_specs(drives, names, horizon_steps, stride=300, route_mode="route"):
    specs = []
    for n in names:
        d = drives[n]
        last_t = -10**9
        for t0 in d.starts:
            if t0 - last_t < stride or d.run_end[t0] - t0 < horizon_steps or d.v[t0 - 1] < 1.0:
                continue
            specs.append({"drive": n, "t0": int(t0), "td": len(d.v) - 1, "route_mode": route_mode,
                          "seed": int(t0), "h_long": horizon_steps})
            last_t = t0
    return specs


def _batches(drives, specs, calibration):
    """Eval batches; causal_pre_outage (default) builds per-outage prefix-calibrated views chunk by chunk."""
    from idr.causal_calibration import CalibrationConfig, build_views
    cfg = calibration or CalibrationConfig()
    if cfg.mode == "offline_oracle":
        ds = OutageDataset(drives, sorted({s["drive"] for s in specs}), len(specs), mode="eval")
        ds.fixed = specs
        yield from DataLoader(ds, batch_size=64)
        return
    for k in range(0, len(specs), 64):
        views, vspecs, _ = build_views(drives, specs[k:k + 64], cfg)
        ds = OutageDataset(views, sorted({s["drive"] for s in vspecs}), len(vspecs), mode="eval")
        ds.fixed = vspecs
        yield from DataLoader(ds, batch_size=64)


@torch.no_grad()
def run(model, drives, specs, dev, horizon_steps, calibration=None, route=None):
    from idr.route_input import RouteConfig, annotate_specs
    specs, _ = annotate_specs(drives, specs, route or RouteConfig(), horizon=horizon_steps)
    res = {"model": [], "physics": []}
    keep = []
    for batch in _batches(drives, specs, calibration):
        b = to_dev(batch, dev)
        out = model(b["seq"], b["static"], b["route"], b["u_raw"])
        for key, u, ps in [("model", out["u_hat"], out["p_stop"]), ("physics", b["u_raw"], None)]:
            traj = physics.integrate(u[..., 0], u[..., 1], b["v0"], ps)
            err = (traj["pos"] - b["pos_rel"]).norm(dim=-1)[:, :horizon_steps]
            dist = b["dist"][:, :horizon_steps]
            aw = angle_weights(b["v_true"], b["hvalid"], b["mask"])[:, :horizon_steps]
            deg = torch.rad2deg(torch.acos(cos_angle(traj["vel"], true_velocity(b)).clamp(-1, 1)))[:, :horizon_steps]
            res[key].append({"err": err.cpu().numpy(), "dist": dist.cpu().numpy(),
                             "angle": ((deg * aw).sum(1) / aw.sum(1).clamp(min=1e-3)).cpu().numpy(),
                             "pos": traj["pos"][:, :horizon_steps].cpu().numpy()})
        keep.append({"pos_rel": b["pos_rel"][:, :horizon_steps].cpu().numpy(), "psi0": b["psi0"].cpu().numpy()})
    cat = lambda lst, k: np.concatenate([x[k] for x in lst])
    return {k: {kk: cat(v, kk) for kk in v[0]} for k, v in res.items()}, \
        {"pos_rel": cat(keep, "pos_rel"), "psi0": cat(keep, "psi0")}


def summarise(r, horizon_steps):
    fe, fd = r["err"][:, -1], r["dist"][:, -1]
    ok = fd > 50
    drift = fe[ok] / fd[ok] * 100
    return {"n": int(ok.sum()), "median_drift_pct": float(np.median(drift)), "p90_drift_pct": float(np.percentile(drift, 90)),
            "pct_outages_under_10pct_drift": float((drift < 10).mean() * 100),
            "median_err_m": float(np.median(fe)), "median_dist_m": float(np.median(fd)),
            "mean_angle_deg": float(np.mean(r["angle"]))}


def crse_max(r):
    """Onyekpe et al.: cumulative root squared error over a 10 s outage, sampled each second; max over windows."""
    e = r["err"][:, 9:100:10]
    return float(np.max(e.sum(1)))


def plot_examples(truth, res, specs, tag, n=4):
    FIGS.mkdir(exist_ok=True)
    turn = np.abs(np.degrees(np.arctan2(truth["pos_rel"][:, -1, 1], truth["pos_rel"][:, -1, 0])))
    dist = res["model"]["dist"][:, -1]
    picks = []
    for order in (np.argsort(-turn), np.argsort(-dist)):          # sharpest turning, longest (fast) outages
        for i in order:
            if i not in picks and dist[i] > 200:
                picks.append(int(i))
                break
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(dist)):
        if len(picks) >= n:
            break
        if dist[i] > 200 and int(i) not in picks:
            picks.append(int(i))
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.2))
    for ax, i in zip(np.atleast_1d(axes), picks):
        for key, lab, st in [(None, "GNSS truth", "k-"), ("physics", "raw IMU (physics)", "C1--"), ("model", "residual TFT", "C0-")]:
            p = truth["pos_rel"][i] if key is None else res[key]["pos"][i]
            ax.plot(p[:, 1], p[:, 0], st, label=lab, lw=1.8)
        ax.plot(0, 0, "go")
        ax.set_aspect("equal", "datalim")
        s = specs[i]
        e_m, e_p = res["model"]["err"][i, -1], res["physics"]["err"][i, -1]
        ax.set_title(f"{s['drive']} t0={s['t0'] / 10:.0f}s  dist {dist[i]:.0f} m\nerr: TFT {e_m:.1f} m / raw {e_p:.1f} m", fontsize=9)
        ax.set_xlabel("right of initial heading (m)")
        ax.set_ylabel("forward (m)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / f"{tag}_trajectories.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft")
    from idr.causal_calibration import add_cli, config_from_args
    add_cli(ap)
    from idr.route_input import add_cli as add_route_cli
    from idr.route_input import config_from_args as route_config_from_args
    add_route_cli(ap)
    args = ap.parse_args()
    calcfg = config_from_args(args)
    routecfg = route_config_from_args(args)
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ResidualTFT(d=a["d"], use_tft=not a["no_tft"]).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives()
    sp = split_names(list(drives))
    report = {"checkpoint_val": ck["val"], "baseline_val": ck["baseline_val"]}

    for H in (300, 600):
        specs = make_specs(drives, sp["test"], H)
        res, truth = run(model, drives, specs, dev, H, calcfg, routecfg)
        report[f"test_{H // 10}s"] = {k: summarise(v, H) for k, v in res.items()}
        if H == 600:
            report["trajectory_plot"] = plot_examples(truth, res, specs, args.tag)
        # route ablation on the same outages
        abl = {}
        for mode, label in [("none", "destination_only"), ("route", "destination_plus_route"),
                            ("wrong", "destination_plus_wrong_route")]:
            sp_m = [{**s, "route_mode": mode} for s in specs]
            abl[label] = summarise(run(model, drives, sp_m, dev, H, calcfg, routecfg)[0]["model"], H)
        report[f"test_{H // 10}s_route_ablation"] = abl

    # MARK-V comparable protocol on (re-synced) Y1: median error at 10/20/30/60 s
    specs = make_specs(drives, ["Y1"], 600)
    res, _ = run(model, drives, specs, dev, 600, calcfg, routecfg)
    report["Y1_marky_protocol_median_err_m"] = {
        k: {f"{s}s": float(np.median(v["err"][:, s * 10 - 1])) for s in (10, 20, 30, 60)} for k, v in res.items()}

    # scenario sets (Onyekpe et al. 2021): 10 s outages every 5 s, max CRSE + drift
    for name in sp["scenario"]:
        specs = make_specs(drives, [name], 100, stride=50)
        if not specs:
            continue
        res, _ = run(model, drives, specs, dev, 100, calcfg, routecfg)
        report[f"scenario_{name}_10s"] = {k: {"max_crse_m": crse_max(v), **summarise(v, 100)} for k, v in res.items()}

    out = ARTIFACTS / f"{args.tag}_eval.json"
    report["calibration_mode"] = calcfg.mode
    report["route_input_source"] = routecfg.source
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
