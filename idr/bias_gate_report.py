"""Breakdowns for the look-ahead bias gate (paired per outage-run against the control).

    python -m idr.bias_gate_report --drives S2_Vtb5 --runs A=ctrlPM B=l15Brec C=gateC --tau -0.25

Gate status per outage comes from the gated run's recorded pre-outage estimate (rows["recent_speed_bias"]); the
same status is applied to the ungated run B for the "by gate status" split.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from idr.normalise import ARTIFACTS


def rows_of(tag, sfx):
    doc = json.loads((ARTIFACTS / f"mm_eval_{tag}_{sfx}_60s.json").read_text())
    return {(r["drive"], r["t0"], r["seed"]): r for r in doc["rows"]["tft_ekf_map"]
            if r["calibration"]["calibration_quality"] != "cold_start"}


def acted(r):
    la = (r.get("stats") or {}).get("lookahead") or {}
    return la.get("n_applications", 0) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", default="S2_Vtb5")
    ap.add_argument("--runs", nargs=3, required=True, help="A=control B=ungated C=gated")
    ap.add_argument("--tau", type=float, default=-0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lab = [r.split("=", 1) for r in args.runs]
    A, B, C = (rows_of(t, args.drives) for _, t in lab)
    bias = {k[:2]: r.get("recent_speed_bias") for k, r in C.items()}
    gate_in = {k: (b is not None and b.get("available") and b["recent_ekf_speed_bias_mps"] >= args.tau)
               for k, b in bias.items()}
    seeds = sorted({k[2] for k in A})
    worst = set()
    for s in seeds:
        ks = [k for k in A if k[2] == s]
        dr = np.array([A[k]["err_end"] / A[k]["dist"] for k in ks])
        thr = np.percentile(dr, 75)
        worst |= {k for k, x in zip(ks, dr) if x >= thr}
    rep = {}
    print(f"\n== {args.drives}: gate tau = {args.tau:+.2f}; outages gated in "
          f"{sum(gate_in.values())}/{len(gate_in)} (unavailable estimate: "
          f"{sum(1 for b in bias.values() if not (b and b.get('available')))})")
    for name, T in ((lab[1][0], B), (lab[2][0], C)):
        flips = Counter()
        tail = {"gated_in": [], "gated_out": []}
        tail_w = {"gated_in": [], "gated_out": []}
        cov = sum(acted(r) for r in T.values())
        for k, r in T.items():
            a = A[k]
            grp = "gated_in" if gate_in.get(k[:2]) else "gated_out"
            act = "acted" if acted(r) else "not_acted"
            if a["wrong_road"] != r["wrong_road"]:
                d = "became_wrong" if r["wrong_road"] else "became_right"
                flips[(act, d)] += 1
                flips[(grp, d)] += 1
            tail[grp].append(r["along"] - a["along"])
            if k in worst:
                tail_w[grp].append(r["along"] - a["along"])
        out = {"activation_coverage": f"{cov}/{len(T)}",
               "wrong_road_flips": {f"{a_}/{d_}": n for (a_, d_), n in sorted(flips.items())},
               "mean_along_change_m": {g: (float(np.mean(v)) if v else None, len(v)) for g, v in tail.items()},
               "mean_along_change_in_control_worst_quarter_m": {g: (float(np.mean(v)) if v else None, len(v))
                                                                for g, v in tail_w.items()}}
        rep[name] = out
        print(f"\n{name}: prior acted in {out['activation_coverage']} outage-runs")
        for g in ("acted", "not_acted", "gated_in", "gated_out"):
            print(f"  wrong-road flips [{g:9s}] became wrong {flips[(g, 'became_wrong')]:3d} | became right "
                  f"{flips[(g, 'became_right')]:3d} | net {flips[(g, 'became_wrong')] - flips[(g, 'became_right')]:+d}")
        for g in ("gated_in", "gated_out"):
            m_, n_ = out["mean_along_change_m"][g]
            mw, nw = out["mean_along_change_in_control_worst_quarter_m"][g]
            print(f"  along-track change vs control [{g:9s}] all {m_ if m_ is None else round(m_, 1)} m (n={n_}) | "
                  f"control worst quarter {mw if mw is None else round(mw, 1)} m (n={nw})")
    # is the pre-outage estimate predictive of the in-outage cruising bias? (control run, constant-speed phase)
    xs, ys = [], []
    for k, r in A.items():
        b = bias.get(k[:2])
        sb = r.get("speed_bias", {}).get("constant_speed")
        if b and b.get("available") and sb and sb[1] > 50:
            xs.append(b["recent_ekf_speed_bias_mps"])
            ys.append(sb[0] / sb[1])
    if xs:
        r_ = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\npre-outage bias vs in-outage constant-speed bias (control, {len(xs)} outage-runs): Pearson r = {r_:+.2f}")
        rep["predictive_r"] = r_
    if args.out:
        (ARTIFACTS / args.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
