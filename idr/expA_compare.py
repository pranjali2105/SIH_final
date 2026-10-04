"""Experiment A report: control vs treatment tail diagnostics (same outages, same seeds).

`python -m idr.expA_compare --control tail_diag_ctrl3.json --treat tail_diag_expA3.json`

Tail = |final along-track| >= the CONTROL's seed-0 p75 threshold, or wrong road at the end (a fixed threshold, so
"fewer tail outages" is measured against the same bar). Classes follow idr.tail_diag.classify.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from idr.normalise import ARTIFACTS
from idr.tail_diag import CLASSES, classify


def load(name):
    return json.loads((ARTIFACTS / name).read_text())["rows"]


def summarise(rows, q):
    tail = [r for r in rows if r["abs_along_end_m"] >= q or r["wrong_end"]]
    for r in rows:
        r["class_fixed"] = classify(r) if (r["abs_along_end_m"] >= q or r["wrong_end"]) else 0
    ab = np.array([r["abs_along_end_m"] for r in rows])
    jd = np.array([abs(x) for r in rows for x in r.get("jct_dt_s", [])])
    per_seed = sorted({r["seed"] for r in rows})
    return {
        "n": len(rows),
        "median_drift_pct": float(np.median([r["drift_pct"] for r in rows])),
        "wrong_road_pct": float(np.mean([r["wrong_end"] for r in rows]) * 100),
        "median_abs_along_m": float(np.median(ab)),
        "worst25_along_m": float(np.mean(np.sort(ab)[-max(1, len(ab) // 4):])),
        "p75_along_m": float(np.percentile(ab, 75)),
        "median_cross_end_m": float(np.median([r["cross_end_m"] for r in rows if r.get("cross_end_m") is not None])),
        "n_tail": len(tail), "tail_pct": 100 * len(tail) / len(rows),
        "tail_classes": {CLASSES[k]: v for k, v in sorted(Counter(r["class_fixed"] for r in tail).items())},
        "wrong_successor_n": sum(r["class_fixed"] == 3 for r in rows),
        "impoverishment_n": sum(r["class_fixed"] == 7 for r in rows),
        "conf_actions": dict(sum((Counter({k: v for k, v in r.get("conf_stats", {}).items() if k.startswith("conf_")})
                                  for r in rows), Counter())),
        "junction_n": int(len(jd)), "junction_abs_dt_median_s": float(np.median(jd)) if len(jd) else None,
        "junction_abs_dt_p90_s": float(np.percentile(jd, 90)) if len(jd) else None,
        "junction_late_early_gt3s_pct": float(np.mean(jd > 3) * 100) if len(jd) else None,
        "ess_prior_med": float(np.median([r["ess_prior_med"] for r in rows if r.get("ess_prior_med") is not None])),
        "ess_post_med": float(np.median([r["ess_post_med"] for r in rows if r.get("ess_post_med") is not None])),
        "ess_resampled_med": float(np.median([r["ess_resampled_med"] for r in rows if r.get("ess_resampled_med") is not None])),
        "n_roads_med": float(np.median([r["n_roads_med"] for r in rows if r.get("n_roads_med") is not None])),
        "alm_events": int(sum(r.get("alm_events", 0) for r in rows)),
        "alm_applied": int(sum(r.get("alm_applied", 0) for r in rows)),
        "alm_skips": dict(sum((Counter(r.get("alm_skips", {})) for r in rows), Counter())),
        "alm_abs_r_med_m": float(np.median([abs(x) for r in rows for x in r.get("alm_r_med", []) if x is not None]))
        if any(r.get("alm_r_med") for r in rows) else None,
        "per_seed_median_drift": [float(np.median([r["drift_pct"] for r in rows if r["seed"] == s])) for s in per_seed],
        "per_seed_wrong_road": [float(np.mean([r["wrong_end"] for r in rows if r["seed"] == s]) * 100) for s in per_seed],
        "per_seed_worst25_along": [float(np.mean(np.sort([r["abs_along_end_m"] for r in rows if r["seed"] == s])
                                                 [-max(1, sum(r["seed"] == s for r in rows) // 4):])) for s in per_seed],
    }


def acceptance(mm_c, mm_t, sc, st):
    """The Experiment A acceptance criteria, treatment vs control (global = evaluate_mm 3 seeds, tail = tail_diag)."""
    g = lambda f: json.loads((ARTIFACTS / f).read_text())["summary"]["tft_ekf_map"]
    c, t = g(mm_c), g(mm_t)
    crit = [
        ("worst-quarter along-track < control", c["worst25_along_m"], t["worst25_along_m"], t["worst25_along_m"] < c["worst25_along_m"]),
        ("fewer tail outages", sc["n_tail"], st["n_tail"], st["n_tail"] < sc["n_tail"]),
        ("median drift <= control", c["median_drift_pct"], t["median_drift_pct"], t["median_drift_pct"] <= c["median_drift_pct"] + 1e-9),
        ("wrong-road <= control", c["wrong_road_pct"], t["wrong_road_pct"], t["wrong_road_pct"] <= c["wrong_road_pct"] + 1e-9),
        ("no cross-track increase (<= +0.5 m)", c["median_cross_m"], t["median_cross_m"], t["median_cross_m"] <= c["median_cross_m"] + 0.5),
        ("lower junction timing error (p90)", sc["junction_abs_dt_p90_s"], st["junction_abs_dt_p90_s"], st["junction_abs_dt_p90_s"] < sc["junction_abs_dt_p90_s"]),
        ("wrong successor not increased", sc["wrong_successor_n"], st["wrong_successor_n"], st["wrong_successor_n"] <= sc["wrong_successor_n"]),
        ("impoverishment not increased", sc["impoverishment_n"], st["impoverishment_n"], st["impoverishment_n"] <= sc["impoverishment_n"]),
    ]
    out = [{"criterion": n, "control": a, "treatment": b, "pass": bool(ok)} for n, a, b, ok in crit]
    return {"criteria": out, "all_pass": all(x["pass"] for x in out),
            "median_drift_range_treatment": t.get("median_drift_pct_range"),
            "wrong_road_range_treatment": t.get("wrong_road_pct_range")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="tail_diag_ctrl3.json")
    ap.add_argument("--treat", default="tail_diag_expA3.json")
    ap.add_argument("--out", default="expA_report.json")
    ap.add_argument("--mm_control", default="mm_eval_valA_S2_Vtb5_60s.json")
    ap.add_argument("--mm_treat", default=None, help="evaluate_mm JSON of the treatment (global acceptance metrics)")
    args = ap.parse_args()
    c, t = load(args.control), load(args.treat)
    q = float(np.percentile([r["abs_along_end_m"] for r in c if r["seed"] == 0], 75))
    sc, st = summarise(c, q), summarise(t, q)
    key = lambda r: (r["drive"], r["t0"], r["seed"])
    tc = {key(r): r for r in c}
    trans = Counter()
    helped, hurt = [], []
    for r in t:
        k = key(r)
        if k not in tc:
            continue
        a, b = tc[k]["class_fixed"], r["class_fixed"]
        trans[(a, b)] += 1
        d = r["abs_along_end_m"] - tc[k]["abs_along_end_m"]
        (helped if d < -10 else hurt if d > 10 else []).append(d)
    names = {0: "ok", **{k: v for k, v in CLASSES.items()}}
    trans_named = {f"{names[a]} -> {names[b]}": n for (a, b), n in sorted(trans.items()) if a != b}
    rep = {"tail_threshold_m (control seed-0 p75)": q, "control": sc, "treatment": st,
           "class_transitions (changed only)": trans_named,
           "paired_along_change": {"n_improved_gt10m": len(helped), "n_worse_gt10m": len(hurt),
                                   "median_improvement_m": float(np.median(helped)) if helped else None,
                                   "median_worsening_m": float(np.median(hurt)) if hurt else None}}
    if args.mm_treat:
        rep["acceptance"] = acceptance(args.mm_control, args.mm_treat, sc, st)
    (ARTIFACTS / args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
