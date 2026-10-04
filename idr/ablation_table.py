"""Standard ablation table (S2 + Vtb5 validation, 60 s outages, 3 seeds). `python -m idr.ablation_table`.

    python -m idr.ablation_table --runs Control=reproCtrl:tail_diag_reproCtrl.json A3-gate=reproA3gate:tail_diag_reproA3gate.json

Each run = <label>=<evaluate_mm out_tag>:<tail_diag json>. Metrics:
  median drift %, wrong-road %, cross-track (m), along-track (m)      evaluate_mm, tft_ekf_map, mean of 3 seeds
  worst-quarter along-track (m) and pass/fail                         pass = below the CONTROL (first run) value
  along-track-first %                                                  tail_diag: share of worst-quarter failures
                                                                       (with both times) where along-track error
                                                                       crosses 30 m > 1 s before the first wrong edge
"""
from __future__ import annotations

import argparse
import json

from idr.normalise import ARTIFACTS

DRIVES = "S2_Vtb5"


def run_mode(doc) -> str:
    """Calibration mode of a results file. Files written before calibration modes existed are legacy oracle."""
    return (doc.get("calibration") or {}).get("calibration_mode") or doc.get("summary", {}).get("calibration_mode") \
        or "offline_oracle"


def route_source(doc) -> str:
    """TFT route-input source of a results file. Files written before route sources existed used the true track."""
    return (doc.get("route") or {}).get("route_input_source") or doc.get("summary", {}).get("route_input_source") \
        or "oracle_future_gnss"


def load(tag, td):
    doc = json.loads((ARTIFACTS / f"mm_eval_{tag}_{DRIVES}_60s.json").read_text())
    mm = doc["summary"]["tft_ekf_map"]
    tdoc = json.loads((ARTIFACTS / td).read_text()) if td else None
    t = tdoc["summary"] if tdoc else {}
    modes = {run_mode(doc)} | ({run_mode(tdoc)} if tdoc else set())
    if len(modes) != 1:
        raise ValueError(f"{tag}: evaluation and tail diagnostics use different calibration modes {modes}")
    routes = {route_source(doc)} | ({route_source(tdoc)} if tdoc else set())
    if len(routes) != 1:
        raise ValueError(f"{tag}: evaluation and tail diagnostics use different route sources {routes}")
    return {"drift": mm["median_drift_pct"], "drift_rng": mm.get("median_drift_pct_range"),
            "wrong": mm["wrong_road_pct"], "cross": mm["median_cross_m"], "along": mm["median_along_m"],
            "w25": mm["worst25_along_m"], "first": t.get("along_diverges_first_pct"), "mode": modes.pop(), "route": routes.pop()}


def table(runs, input_definition_change=False):
    """input_definition_change=True: allow runs that differ ONLY in the TFT route-input source (labelled as an
    input-definition change, not a treatment). Calibration modes may never be mixed."""
    rows = [(lab, load(tag, td)) for lab, tag, td in runs]
    modes = {r["mode"] for _, r in rows}
    if len(modes) != 1:
        raise ValueError(f"refusing to compare runs with different calibration modes: "
                         f"{[(lab, r['mode']) for lab, r in rows]}")
    routes = {r["route"] for _, r in rows}
    if len(routes) != 1 and not input_definition_change:
        raise ValueError(f"refusing to compare runs with different route-input sources "
                         f"{[(lab, r['route']) for lab, r in rows]} (pass --input_definition_change to show them "
                         f"side by side as an input-definition change)")
    ctrl = rows[0][1]["w25"]
    out = ["| Variant | Median drift % | Wrong-road % | Cross-track (m) | Along-track (m) | Worst-quarter along (m) "
           "| Worst-quarter criterion | Along-track first % |", "|---|---|---|---|---|---|---|---|"]
    if len(routes) != 1:
        out.insert(0, "INPUT-DEFINITION CHANGE (route-input source differs; not a treatment comparison)\n")
    for i, (lab, r) in enumerate(rows):
        lab = f"{lab} [route: {r['route']}]" if len(routes) != 1 else lab
        rng = f" [{r['drift_rng'][0]:.1f}–{r['drift_rng'][1]:.1f}]" if r["drift_rng"] else ""
        crit = "reference" if i == 0 else ("PASS" if r["w25"] < ctrl else "FAIL")
        first = f"{r['first']:.0f}" if r["first"] is not None else "–"
        out.append(f"| {lab} | {r['drift']:.1f}{rng} | {r['wrong']:.1f} | {r['cross']:.1f} | {r['along']:.0f} | "
                   f"{r['w25']:.0f} | {crit} | {first} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=mm_out_tag:tail_diag.json (first = Control)")
    ap.add_argument("--input_definition_change", action="store_true")
    args = ap.parse_args()
    runs = []
    for r in args.runs:
        lab, rest = r.split("=", 1)
        tag, td = (rest.split(":", 1) + [None])[:2]
        runs.append((lab, tag, td))
    print(table(runs, args.input_definition_change))


if __name__ == "__main__":
    main()
