"""Control vs look-ahead slowdown-prior report (S2 + Vtb5, headline variant tft_ekf_map, 3 seeds).

    python -m idr.lookahead_report --runs Control=ctrlOsmV2 A3-gate=a3gateOsm L1=lookL1 L2=lookL2

First run = control. All runs must share calibration mode and route-input source (idr.ablation_table guard) and the
same outages. Speed biases need evaluate_mm --speed_bias (map-aided EKF speed minus GNSS speed, speed_slices labels).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from idr.ablation_table import route_source, run_mode
from idr.normalise import ARTIFACTS

KEY = "tft_ekf_map"
BIAS = [("braking", "Braking bias"), ("roundabout_entry", "Roundabout-entry bias"), ("after_turn", "After-turn bias"),
        ("junction_approach", "Junction-approach bias"), ("constant_speed", "Constant-speed bias"),
        ("high_speed", "High-speed (>15 m/s) bias"), ("all", "All-sample bias")]


def per_seed(rows):
    out = {}
    for sd in sorted({r["seed"] for r in rows}):
        R = [r for r in rows if r["seed"] == sd]
        drift = np.array([r["err_end"] / r["dist"] * 100 for r in R])
        along = np.array([r["along"] for r in R])
        worst = drift >= np.percentile(drift, 75)
        b = {}
        if "speed_bias" in R[0]:
            for k, _ in BIAS:
                se = sum(r["speed_bias"][k][0] for r in R)
                n = sum(r["speed_bias"][k][1] for r in R)
                b[k] = se / n if n else np.nan
        out[sd] = {"drift": float(np.median(drift)), "wrong": float(np.mean([r["wrong_road"] for r in R]) * 100),
                   "cross": float(np.median([r["cross"] for r in R])), "along": float(np.median(along)),
                   "w25": float(np.median(along[worst])), "bias": b}
    return out


def activation(rows):
    la = [r["stats"].get("lookahead") for r in rows if r.get("stats") and r["stats"].get("lookahead")]
    if not la:
        return None
    n_app = sum(x["n_applications"] for x in la)
    sum_abs = sum(x["total_abs_correction_mps2_steps"] for x in la)
    red = np.array([x["total_speed_reduction_mps"] for x in la])
    ev, rej = Counter(), Counter()
    for x in la:
        ev.update(x["events_detected"])
        rej.update(x["not_applied_by_reason"])
    rec = np.array([x.get("total_recovered_mps", 0.0) for x in la])
    net = np.array([x.get("net_map_speed_change_mps", -x["total_speed_reduction_mps"]) for x in la])
    ends = Counter(e.get("recovery_end_reason") for x in la for e in x.get("events", []) if e.get("event_reduction_mps", 0) > 0)
    return {"recovered_per_outage_mps": {"mean": float(rec.mean()), "median": float(np.median(rec)), "max": float(rec.max())},
            "net_map_change_per_outage_mps": {"mean": float(net.mean()), "min": float(net.min()), "max": float(net.max())},
            "recovery_end_reasons": dict(ends),
            "runs": len(la), "applications": n_app, "mode": la[0].get("prior_mode", "acceleration"),
            "outage_runs_event_cap": int(sum(x.get("events_reaching_event_cap", 0) > 0 for x in la)),
            "outage_runs_outage_cap": int(sum(bool(x.get("outage_cap_reached")) for x in la)), "applications_per_outage_run": n_app / len(la),
            "outage_runs_with_any_application": int(sum(x["n_applications"] > 0 for x in la)),
            "mean_abs_correction_mps2": sum_abs / n_app if n_app else 0.0,
            "max_abs_correction_mps2": max(x["max_abs_correction_mps2"] for x in la),
            "speed_reduction_per_outage_mps": {"mean": float(red.mean()), "median": float(np.median(red)),
                                               "max": float(red.max())},
            "events_detected": dict(ev), "not_applied_by_reason": dict(rej.most_common())}


DRIVES = "S2_Vtb5"


def load(tag):
    doc = json.loads((ARTIFACTS / f"mm_eval_{tag}_{DRIVES}_60s.json").read_text())
    # headline protocol: cold_start outages (no usable pre-outage calibration) are excluded, as in evaluate_mm
    rows = [r for r in doc["rows"][KEY] if r.get("calibration", {}).get("calibration_quality") != "cold_start"]
    pf = [r["stats"]["pf_health"] for r in rows if r.get("stats") and r["stats"].get("pf_health")]
    return {"doc": doc, "rows": rows, "seed": per_seed(rows), "act": activation(rows), "pf": pf,
            "mode": run_mode(doc), "route": route_source(doc), "outages": sorted({(r["drive"], r["t0"]) for r in rows})}


def fmt(vals, nd=1):
    v = np.array(vals, float)
    return f"{np.nanmean(v):.{nd}f} [{np.nanmin(v):.{nd}f}–{np.nanmax(v):.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=evaluate_mm out_tag (first = control)")
    ap.add_argument("--out", default="lookahead_report.json")
    ap.add_argument("--drives", default="S2_Vtb5", help="drive suffix of the evaluate_mm files, e.g. Vw4")
    ap.add_argument("--policy", default="strict", choices=["strict", "final"],
                    help="final = predeclared enable-by-default criteria (2026-10-04): uniform +-0.05 m/s bias band, "
                         "worst-quarter per seed below control or at most +5 m, net map change never positive")
    args = ap.parse_args()
    global DRIVES
    DRIVES = args.drives
    runs = [(r.split("=", 1)[0], load(r.split("=", 1)[1])) for r in args.runs]
    if len({x["mode"] for _, x in runs}) != 1 or len({x["route"] for _, x in runs}) != 1:
        raise ValueError(f"mixed calibration/route conditions: {[(l, x['mode'], x['route']) for l, x in runs]}")
    if len({tuple(x["outages"]) for _, x in runs}) != 1:
        raise ValueError("runs do not share the same outages")
    ctrl = runs[0][1]
    seeds = sorted(ctrl["seed"])
    head = ["Metric"] + [l for l, _ in runs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(name, fn, nd=1):
        lines.append("| " + name + " | " + " | ".join(fn(x, nd) for _, x in runs) + " |")

    row("Median drift %", lambda x, nd: fmt([x["seed"][s]["drift"] for s in seeds], nd))
    row("Wrong-road %", lambda x, nd: fmt([x["seed"][s]["wrong"] for s in seeds], nd))
    row("Cross-track (m)", lambda x, nd: fmt([x["seed"][s]["cross"] for s in seeds], nd))
    row("Along-track (m)", lambda x, nd: fmt([x["seed"][s]["along"] for s in seeds], 0), 0)
    row("Worst-quarter along (m)", lambda x, nd: fmt([x["seed"][s]["w25"] for s in seeds], 0), 0)
    for k, name in BIAS:
        row(name + " (m/s)", lambda x, nd, k=k: fmt([x["seed"][s]["bias"].get(k, np.nan) for s in seeds], 2), 2)
    act = lambda f: (lambda x, nd: f(x["act"]) if x["act"] else "off")
    row("Prior applications (steps, 3 seeds)", act(lambda a: f"{a['applications']}"))
    row("Outage-runs with ≥1 application", act(lambda a: f"{a['outage_runs_with_any_application']}/{a['runs']}"))
    row("Mean / max |correction| (m/s²)", act(lambda a: f"{a['mean_abs_correction_mps2']:.3f} / {a['max_abs_correction_mps2']:.3f}"))
    row("Speed reduction per outage mean / median / max (m/s)",
        act(lambda a: "{mean:.2f} / {median:.2f} / {max:.2f}".format(**a["speed_reduction_per_outage_mps"])))
    row("Recovered per outage mean / median / max (m/s)",
        act(lambda a: "{mean:.2f} / {median:.2f} / {max:.2f}".format(**a["recovered_per_outage_mps"])))
    row("Net map speed change per outage mean / min / max (m/s)",
        act(lambda a: "{mean:.2f} / {min:.2f} / {max:.2f}".format(**a["net_map_change_per_outage_mps"])))
    row("Outage-runs reaching an event cap", act(lambda a: f"{a['outage_runs_event_cap']}/{a['runs']}"))
    row("Outage-runs reaching the outage cap", act(lambda a: f"{a['outage_runs_outage_cap']}/{a['runs']}"))
    pfr = lambda f: (lambda x, nd: f(x["pf"]) if x["pf"] else "n/a")
    row("PF ESS after weighting, median (fraction of N)", pfr(lambda pf: f"{np.median([q['ess_post_median'] for q in pf]):.2f}"))
    row("PF steps with ESS < 10% N (mean %)", pfr(lambda pf: f"{100 * np.mean([q['frac_ess_post_below_10pct'] for q in pf]):.1f}"))
    row("PF steps with a single road hypothesis (mean %)", pfr(lambda pf: f"{100 * np.mean([q['frac_single_road'] for q in pf]):.1f}"))
    print("\n".join(lines))

    # acceptance vs control, per seed (paired)
    verdicts = {}
    for lab, x in runs[1:]:
        c, t = ctrl["seed"], x["seed"]
        d_w25 = [t[s]["w25"] - c[s]["w25"] for s in seeds]
        d_cross = [t[s]["cross"] - c[s]["cross"] for s in seeds]
        d_wrong = [t[s]["wrong"] - c[s]["wrong"] for s in seeds]
        mean_b = lambda run, k: np.nanmean([run[s]["bias"].get(k, np.nan) for s in seeds])
        crit = {
            "worst-quarter along < control in EVERY seed": all(v < 0 for v in d_w25),
            "worst-quarter along mean < control": float(np.mean(d_w25)) < 0,
            "no material cross-track degradation (mean +<=0.5 m)": float(np.mean(d_cross)) <= 0.5,
            "no material wrong-road degradation (mean +<=1.0 pt)": float(np.mean(d_wrong)) <= 1.0,
            "constant-speed |bias| not worse (+<=0.05 m/s)":
                abs(mean_b(t, "constant_speed")) <= abs(mean_b(c, "constant_speed")) + 0.05,
            "high-speed |bias| not worse (+<=0.05 m/s)": abs(mean_b(t, "high_speed")) <= abs(mean_b(c, "high_speed")) + 0.05,
            "braking |bias| lower": abs(mean_b(t, "braking")) < abs(mean_b(c, "braking")),
            "roundabout-entry |bias| lower": abs(mean_b(t, "roundabout_entry")) < abs(mean_b(c, "roundabout_entry")),
            "after-turn |bias| lower": abs(mean_b(t, "after_turn")) < abs(mean_b(c, "after_turn")),
        }
        if args.policy == "final":
            net_max = max((x2.get("net_map_speed_change_mps", -x2["total_speed_reduction_mps"])
                           for r in x["rows"] for x2 in [r["stats"].get("lookahead") or {}] if x2), default=None)
            within = lambda k: abs(mean_b(t, k) - mean_b(c, k)) <= 0.05
            crit = {
                "worst-quarter along per seed below control or <= +5 m": all(v <= 5.0 for v in d_w25),
                "  (strictly below control in every seed)": all(v < 0 for v in d_w25),
                "braking |bias| improves": abs(mean_b(t, "braking")) < abs(mean_b(c, "braking")),
                "roundabout-entry |bias| improves": abs(mean_b(t, "roundabout_entry")) < abs(mean_b(c, "roundabout_entry")),
                "after-turn bias within +-0.05 m/s of control": within("after_turn"),
                "constant-speed bias within +-0.05 m/s of control": within("constant_speed"),
                "high-speed bias within +-0.05 m/s of control": within("high_speed"),
                "wrong-road not worse by > 1 pt": float(np.mean(d_wrong)) <= 1.0,
                "cross-track not worse by > 0.5 m": float(np.mean(d_cross)) <= 0.5,
                "net map-induced speed change never positive (every outage-run)": net_max is None or net_max <= 1e-9,
            }
            informational = "  (strictly below control in every seed)"
            passed = all(v for k, v in crit.items() if k != informational)
            # alternative reading of the band: "not worse than control by more than 0.05 m/s" (improvements allowed)
            not_worse = lambda k: abs(mean_b(t, k)) <= abs(mean_b(c, k)) + 0.05
            alt = {k: v for k, v in crit.items() if k != informational}
            for k_, ph in (("after-turn bias within +-0.05 m/s of control", "after_turn"),
                           ("constant-speed bias within +-0.05 m/s of control", "constant_speed"),
                           ("high-speed bias within +-0.05 m/s of control", "high_speed")):
                alt[k_] = not_worse(ph)
            passed_alt = all(alt.values())
            deltas = {ph: round(float(mean_b(t, ph) - mean_b(c, ph)), 3) for ph in ("after_turn", "constant_speed",
                                                                                   "high_speed", "braking",
                                                                                   "roundabout_entry")}
        else:
            passed = all(crit.values())
        verdicts[lab] = {"criteria": crit, "all_pass_policy": passed, "all_pass": all(crit.values()),
                         "paired_seed_deltas": {"worst25_along_m": d_w25, "cross_m": d_cross, "wrong_road_pts": d_wrong}}
        print(f"\n{lab} [{args.policy} policy, literal +-0.05 band]: {'ALL CRITERIA PASS' if passed else 'FAILS'}")
        if args.policy == "final":
            print(f"{lab} [final policy, 'not worse by > 0.05' reading]: {'ALL CRITERIA PASS' if passed_alt else 'FAILS'}")
            print(f"  signed bias change vs control (m/s): {deltas}")
            verdicts[lab].update({"all_pass_not_worse_reading": passed_alt, "bias_deltas_vs_control": deltas})
        for k, v in crit.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"  paired worst-quarter delta per seed (m): {[round(v, 1) for v in d_w25]}")
        print(f"  paired wrong-road delta per seed (pts): {[round(v, 1) for v in d_wrong]}; cross (m): {[round(v, 2) for v in d_cross]}")
    out = {"control": runs[0][0], "condition": {"calibration_mode": ctrl["mode"], "route_input_source": ctrl["route"]},
           "n_outages": len(ctrl["outages"]), "seeds": seeds,
           "runs": {l: {"per_seed": x["seed"], "activation": x["act"]} for l, x in runs}, "acceptance": verdicts}
    (ARTIFACTS / args.out).write_text(json.dumps(out, indent=1, default=str))
    for lab, x in runs:
        if x["act"]:
            print(f"\n{lab} activation: events detected {x['act']['events_detected']}")
            print(f"  not applied by reason: {x['act']['not_applied_by_reason']}")
            if any(k is not None for k in x["act"]["recovery_end_reasons"]):
                print(f"  recovery end reasons (per event): {x['act']['recovery_end_reasons']}")


if __name__ == "__main__":
    main()
