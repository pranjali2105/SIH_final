"""Select the look-ahead bias-gate threshold on S2 + Vtb5 ONLY. `python -m idr.bias_gate_tune`

PREDECLARED RULE (written before any result was looked at, 2026-10-04):
  grid tau in {-1.0, -0.75, -0.5, -0.25, 0.0, +0.25} (minimum_allowed_bias_mps)
  choose the MOST CONSERVATIVE (highest) tau such that, on S2 + Vtb5:
    (1) mean worst-quarter along-track gain vs control >= 50% of the ungated gain (B vs A),
    (2) worst-quarter along-track < control in every seed,
    (3) wrong-road not worse than control by > 1 pt in any seed;
  if no tau qualifies, use the default -0.25.

Emulation is exact for the outage-level gate: an outage whose pre-outage bias is below tau gets NO prior, and a run
with the prior fully suppressed is bit-identical to the control (test_speed_bias_gate::test_f), so its rows are taken
from the control run A; outages at/above tau take the ungated run B. The branch-proximity guard is per event and
cannot be emulated; the chosen tau is therefore confirmed by a real run (C) afterwards.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from idr.causal_calibration import CalibrationConfig
from idr.dataset import load_drives
from idr.evaluate import make_specs
from idr.evaluate_mm import TFT_CFG
from idr.lookahead_report import per_seed
from idr.model import ResidualTFT
from idr.nav_filter import NavConfig
from idr.normalise import ARTIFACTS
from idr.osm_map import drive_roadmap
from idr.route_input import RouteConfig
from idr.speed_bias_gate import recent_speed_bias
from idr.train import pick_device

GRID = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25]


def shadow_biases(drives_names, window_s=30.0):
    dev = pick_device()
    ck = torch.load(ARTIFACTS / "tft_ep2.pt", map_location="cpu", weights_only=False)
    m = ResidualTFT().to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    drives, _ = load_drives(drives_names)
    out = {}
    for name in drives_names:                       # per drive, all specs: same batching as evaluate_mm
        specs = make_specs(drives, [name], 600, stride=600)
        b = recent_speed_bias(m, drives, specs, dev, CalibrationConfig(), RouteConfig(), NavConfig(**TFT_CFG),
                              window_s, roadmaps={name: drive_roadmap(name)})
        for sp, x in zip(specs, b):
            out[(name, sp["t0"])] = x
    return out


def rows_of(tag, drives_sfx):
    doc = json.loads((ARTIFACTS / f"mm_eval_{tag}_{drives_sfx}_60s.json").read_text())
    return [r for r in doc["rows"]["tft_ekf_map"] if r["calibration"]["calibration_quality"] != "cold_start"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="ctrlPM")
    ap.add_argument("--ungated", default="l15Brec")
    ap.add_argument("--out", default="bias_gate_tune.json")
    args = ap.parse_args()
    bias = shadow_biases(["S2", "Vtb5"])
    A = {(r["drive"], r["t0"], r["seed"]): r for r in rows_of(args.control, "S2_Vtb5")}
    B = {(r["drive"], r["t0"], r["seed"]): r for r in rows_of(args.ungated, "S2_Vtb5")}
    assert set(A) == set(B)
    vals = np.array([v["recent_ekf_speed_bias_mps"] for (n, t0), v in bias.items() if v["available"]
                     and any(k[:2] == (n, t0) for k in A)])
    n_unavail = sum(1 for (n, t0), v in bias.items() if not v["available"] and any(k[:2] == (n, t0) for k in A))
    pa, pb = per_seed(list(A.values())), per_seed(list(B.values()))
    seeds = sorted(pa)
    gain_b = np.mean([pa[s]["w25"] - pb[s]["w25"] for s in seeds])
    print(f"S2+Vtb5 pre-outage EKF speed bias over {len(vals)} outages (+{n_unavail} unavailable -> suppressed): "
          f"mean {vals.mean():+.2f}, quartiles {np.round(np.percentile(vals, [25, 50, 75]), 2)} m/s")
    print(f"ungated worst-quarter gain {gain_b:.1f} m (control {np.mean([pa[s]['w25'] for s in seeds]):.0f} m)\n")
    res, chosen = [], None
    print("| tau | gated-in outages | median drift | wrong-road | cross | worst-quarter | per-seed w25 delta | per-seed wrong-road delta | qualifies |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tau in GRID:
        rows = []
        n_in = set()
        for k, ra in A.items():
            b_ = bias.get(k[:2])
            gated_in = b_ is not None and b_["available"] and b_["recent_ekf_speed_bias_mps"] >= tau
            rows.append(B[k] if gated_in else ra)
            if gated_in:
                n_in.add(k[:2])
        pt = per_seed(rows)
        dw = [pt[s]["w25"] - pa[s]["w25"] for s in seeds]
        dr = [pt[s]["wrong"] - pa[s]["wrong"] for s in seeds]
        gain = -np.mean(dw)
        ok = gain >= 0.5 * gain_b and all(x < 0 for x in dw) and all(x <= 1.0 for x in dr)
        res.append({"tau": tau, "gated_in": len(n_in), "drift": np.mean([pt[s]["drift"] for s in seeds]),
                    "wrong": np.mean([pt[s]["wrong"] for s in seeds]), "cross": np.mean([pt[s]["cross"] for s in seeds]),
                    "w25": np.mean([pt[s]["w25"] for s in seeds]), "dw": dw, "dr": dr, "qualifies": ok})
        print(f"| {tau:+.2f} | {len(n_in)}/{len({k[:2] for k in A})} | {res[-1]['drift']:.2f} | {res[-1]['wrong']:.1f} | "
              f"{res[-1]['cross']:.2f} | {res[-1]['w25']:.0f} | {[round(x, 1) for x in dw]} | {[round(x, 1) for x in dr]} | {ok} |")
        if ok:
            chosen = tau                               # grid ascending -> last qualifying = most conservative
    chosen = chosen if chosen is not None else -0.25
    print(f"\nselected minimum_allowed_bias_mps = {chosen:+.2f} (predeclared rule)")
    (ARTIFACTS / args.out).write_text(json.dumps({"rule": __doc__, "grid": res, "selected": chosen,
                                                  "shadow_bias": {f"{k[0]}@{k[1]}": v for k, v in bias.items()}},
                                                 indent=1, default=float))


if __name__ == "__main__":
    main()
