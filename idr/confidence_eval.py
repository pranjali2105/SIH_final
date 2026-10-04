"""Confidence evaluation and threshold selection (validation drives only). `python -m idr.confidence_eval`.

Runs the v2 turn landmark UNGATED with confidence logging (`along_landmark=True, confidence_log=True`). In that mode
every proposed correction is applied, so each one's real outcome can be set against the confidence it carried.

Labels (from GNSS ground truth):
  junction landmark  correct  = located junction within 15 m of the true position at the gyro-turn midpoint
                     success  = position error 10 s later < 20 m and no wrong road within 10 s
  1 Hz map update    success  = same definition, for strong/soft EKF updates from _decide

Outputs: idr/artifacts/confidence_eval.json, idr/figures/confidence_{reliability,precision_coverage}.png
  - reliability diagram + ECE of C_junction as a probability of "landmark correct"
  - precision vs coverage of the gate rule (action == hard AND C_junction >= t AND every factor >= cfactor_min)
  - chosen threshold: smallest t with precision >= target on >= min_n accepted events
  - per-action success / wrong-road rate of the 1 Hz updates (does the action ordering mean anything?)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from idr.causal_calibration import add_cli, config_from_args, nav_cfg_for
from idr.route_input import add_cli as add_route_cli
from idr.route_input import config_from_args as route_config_from_args
from idr.dataset import load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import FIGS, TFT_CFG, model_outputs
from idr.map_matching import MMConfig, run_outage
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import drive_roadmap, planned_route
from idr.tail_diag import edge_off_path
from idr.train import pick_device

H = 600
ACTIONS = ("hard_gated_update", "soft_map_update", "preserve_hypotheses", "expand_support")
FACTORS = ("C_edge", "C_turn", "C_timing", "C_cloud")


def wrong_within(rm, mt, gt, t, horizon=100):
    for x in mt:
        if t < x["k_now"] - 1 <= t + horizon:
            kk = int(x["k"])
            lo, hi = max(0, kk - 200), min(H, kk + 200)
            if edge_off_path(rm, x["edge"], gt[lo:hi]) > 15.0:
                return True
    return False


def collect(args, overrides, calcfg, routecfg):
    dev = pick_device()
    ck = torch.load(ARTIFACTS / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives(args.drives)
    jev, upd = [], []
    for name in args.drives:
        d = drives[name]
        rm = drive_roadmap(name)
        specs = make_specs(drives, [name], H, stride=args.stride)
        mo = model_outputs(m, drives, specs, dev, calibration=calcfg, route=routecfg)
        for i, sp in enumerate(specs):
            t0 = sp["t0"]
            gt = d.pos[t0:t0 + H]
            if np.sum(d.v[t0:t0 + H]) * 0.1 < 50:
                continue
            p0, psi0, v0 = d.pos[t0 - 1], float(d.psi[t0 - 1]), float(d.v[t0 - 1])
            u, ps = mo["u_hat"][i], mo["p_stop"][i]
            route = planned_route(rm, p0, psi0, d.pos[-1])
            for sd in range(args.seeds):
                tr, _, _, trc = run_outage(rm, p0, psi0, v0, u[:, 0], u[:, 1], u[:, 2], p_stop=ps,
                                           nav_cfg=nav_cfg_for(mo["cal_meta"][i], NavConfig(**TFT_CFG), calcfg),
                                           seed=t0 * 7 + sd, route_edges=route,
                                           return_trace=True, mm_overrides=overrides,
                                           tft_resid=np.linalg.norm(mo["u_raw"][i, :H] - u, axis=1))
                err = np.hypot(*(tr - gt).T)
                mt = trc["mm_trace"]
                base = {"drive": name, "t0": int(t0), "seed": sd}
                for ev in trc["along_events"]:
                    if "C_junction" not in ev:
                        continue
                    c = int(np.clip(ev["k"], 0, H - 1))
                    t = int(ev["k_now"]) - 1
                    t10 = min(t + 100, H - 1)
                    lm = np.array(ev["landmark"])
                    jev.append({**base, "k": c, "t": t, **{f: ev[f] for f in FACTORS}, "C_junction": ev["C_junction"],
                                "action": ev["action"], "r_med": ev["r_med"], "applied": ev["applied"],
                                "lm_err_m": float(np.hypot(*(lm - gt[c]))),
                                "correct": bool(np.hypot(*(lm - gt[c])) < 15.0),
                                "err_before": float(err[max(t - 1, 0)]), "err_10s": float(err[t10]),
                                "success": bool(err[t10] < 20.0 and not wrong_within(rm, mt, gt, t)),
                                **{f"cr_{k}": ev["cr"][k] for k in ("best_route_probability", "route_margin",
                                                                    "ess_ratio", "pos_sigma_m", "cloud_sigma_m")}})
                for x in mt:
                    if x.get("applied") not in ("strong", "soft") or "cr" not in x:
                        continue
                    t = int(x["k_now"]) - 1
                    t10 = min(t + 100, H - 1)
                    upd.append({**base, "t": t, "applied": x["applied"], "action": x["cr"]["action"],
                                "route_margin": x["cr"]["route_margin"], "p1": x["cr"]["best_route_probability"],
                                "err_now": float(err[t]), "err_10s": float(err[t10]),
                                "success": bool(err[t10] < 20.0 and not wrong_within(rm, mt, gt, t)),
                                "wrong_10s": wrong_within(rm, mt, gt, t)})
        print(name, "done", len(jev), "junction events", len(upd), "updates", flush=True)
    return jev, upd


def reliability(score, label, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out, ece, n = [], 0.0, len(score)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (score >= a) & ((score < b) if b < 1 else (score <= b))
        if m.any():
            conf, acc = float(score[m].mean()), float(label[m].mean())
            out.append({"bin": [round(a, 2), round(b, 2)], "n": int(m.sum()), "mean_conf": conf, "frac_correct": acc})
            ece += m.sum() / n * abs(conf - acc)
    return out, float(ece)


def precision_coverage(jev, label_key, n_all, fmin, grid):
    C = np.array([e["C_junction"] for e in jev])
    lab = np.array([e[label_key] for e in jev], dtype=float)
    hard = np.array([e["action"] == "hard_gated_update" for e in jev], dtype=bool)
    fok = np.array([min(e[f] for f in FACTORS) >= fmin for e in jev], dtype=bool)
    rows = []
    for t in grid:
        acc = hard & fok & (C >= t)
        rows.append({"threshold": round(float(t), 3), "n_accepted": int(acc.sum()),
                     "precision": float(lab[acc].mean()) if acc.any() else None,
                     "coverage_of_scored": float(acc.mean()) if len(acc) else 0.0,
                     "coverage_of_all_turns": float(acc.sum() / max(n_all, 1))})
    return rows


def choose(rows, target, min_n):
    ok = [r for r in rows if r["precision"] is not None and r["n_accepted"] >= min_n and r["precision"] >= target]
    if ok:
        return {"threshold": ok[0]["threshold"], "met_target": True, **ok[0]}
    cand = [r for r in rows if r["precision"] is not None and r["n_accepted"] >= min_n]
    best = max(cand, key=lambda r: (r["precision"], r["n_accepted"])) if cand else None
    return {"threshold": best["threshold"] if best else None, "met_target": False, **(best or {})}


def plots(rel, pc, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="#9aa3ad", lw=1, ls="--", label="perfect calibration")
    ax.plot([b["mean_conf"] for b in rel], [b["frac_correct"] for b in rel], "o-", color="#2a78d6", lw=2,
            label="C_junction")
    for b in rel:
        ax.annotate(str(b["n"]), (b["mean_conf"], b["frac_correct"]), textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color="#4a525c")
    ax.set(xlabel="mean C_junction in bin", ylabel="fraction of landmarks correct", xlim=(0, 1), ylim=(0, 1.02),
           title="Junction-landmark reliability")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "confidence_reliability.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    r = [x for x in pc if x["precision"] is not None]
    ax.plot([x["coverage_of_all_turns"] for x in r], [x["precision"] for x in r], "o-", color="#2a78d6", lw=2, ms=4)
    if chosen.get("threshold") is not None:
        ax.plot([chosen["coverage_of_all_turns"]], [chosen["precision"]], "o", ms=10, mfc="none", mec="#eb6834",
                mew=2, label=f"chosen t = {chosen['threshold']}")
    ax.axhline(0.95, color="#1a7f4b", ls="--", lw=1, label="95% precision")
    ax.set(xlabel="coverage (accepted / all gyro turns)", ylabel="precision (landmark correct)",
           title="Gate rule: hard action AND C_junction ≥ t", ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "confidence_precision_coverage.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tft_ep2")
    ap.add_argument("--drives", nargs="+", default=["S2", "Vtb5"])
    ap.add_argument("--stride", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--out", default="confidence_eval.json")
    add_cli(ap)
    add_route_cli(ap)
    args = ap.parse_args()
    calcfg = config_from_args(args)
    routecfg = route_config_from_args(args)
    overrides = {"along_landmark": True, "along_landmark_mode": "map", "confidence_log": True}
    jev, upd = collect(args, overrides, calcfg, routecfg)
    fmin = MMConfig().cfactor_min
    n_turns = len(jev)
    C = np.array([e["C_junction"] for e in jev])
    lab = np.array([e["correct"] for e in jev], dtype=float)
    rel, ece = reliability(C, lab)
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    pc = precision_coverage(jev, "correct", n_turns, fmin, grid)
    pc_succ = precision_coverage(jev, "success", n_turns, fmin, grid)
    chosen = choose(pc, args.target, args.min_n)
    per_action = {}
    for a in ACTIONS:
        u = [x for x in upd if x["action"] == a]
        j = [e for e in jev if e["action"] == a]
        per_action[a] = {"updates_n": len(u),
                         "update_success": float(np.mean([x["success"] for x in u])) if u else None,
                         "update_wrong_road_10s": float(np.mean([x["wrong_10s"] for x in u])) if u else None,
                         "junction_n": len(j),
                         "junction_correct": float(np.mean([e["correct"] for e in j])) if j else None}
    factor_fail = {f: float(np.mean([e[f] < fmin for e in jev])) for f in FACTORS} if jev else {}
    summary = {"calibration_mode": calcfg.mode, "route_input_source": routecfg.source, "overrides": overrides, "seeds": args.seeds, "drives": args.drives,
               "junction_events_scored": n_turns,
               "landmark_correct_rate": float(lab.mean()) if n_turns else None,
               "landmark_success_rate": float(np.mean([e["success"] for e in jev])) if n_turns else None,
               "ece_C_junction": ece, "reliability": rel,
               "factor_below_min_rate": factor_fail,
               "precision_coverage_correct": pc, "precision_coverage_success": pc_succ,
               "chosen": chosen, "current_cjunction_min": MMConfig().cjunction_min, "per_action": per_action}
    (ARTIFACTS / args.out).write_text(json.dumps({"summary": summary, "junction_events": jev, "updates": upd},
                                                 indent=1, default=str))
    plots(rel, pc, chosen)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("precision_coverage_success",)}, indent=1))


if __name__ == "__main__":
    main()
