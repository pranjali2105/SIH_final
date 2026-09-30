"""Per-step roundabout state labels + traversal/manoeuvre list from GNSS truth and OSM rings.

States: 0 straight, 1 ordinary_turn, 2 entry (3 s before joining a ring), 3 circulating, 4 exit (3 s after leaving).
`python -m idr.roundabout_labels` prints traversal counts per split and manoeuvre.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from idr.data import DT
from idr.osm_map import OSM_DIR, drive_roadmap, ring_arc

STATES = ["straight", "ordinary_turn", "entry", "circulating", "exit"]
ENTRY_EXIT_STEPS = 30
RING_TOL_M = 8.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def label_drive(d, rm) -> tuple[np.ndarray, list[dict]]:
    """d: dataset.Drive (pos, psi, v, u_true), rm: RoadMap with rings. Returns (states (N,), traversals)."""
    N = len(d.v)
    ring_id = np.full(N, -1)
    dist, idx = rm.tree.query(d.pos, k=4, distance_upper_bound=RING_TOL_M)
    for k in range(4):
        ok = np.isfinite(dist[:, k])
        e = np.where(ok, rm.edge[np.minimum(idx[:, k], len(rm.edge) - 1)], -1)
        rid = np.where(ok & (e >= 0), rm.e_ring[np.maximum(e, 0)], -1)
        brg = rm.bearing[np.minimum(idx[:, k], len(rm.edge) - 1)]
        aligned = np.abs(wrap(brg - d.psi)) < np.radians(60)
        take = (ring_id < 0) & (rid >= 0) & aligned & (d.v > 1.0)
        ring_id[take] = rid[take]
    # close 1 s gaps inside a traversal
    for i in range(1, N - 10):
        if ring_id[i] < 0 and ring_id[i - 1] >= 0:
            j = i + np.argmax(ring_id[i:i + 10] >= 0) if (ring_id[i:i + 10] >= 0).any() else -1
            if j > 0 and ring_id[j] == ring_id[i - 1]:
                ring_id[i:j] = ring_id[i - 1]
    states = np.where(np.abs(d.u_true[:, 0]) > 0.1, 1, 0).astype(np.int8)
    trav = []
    i = 0
    while i < N:
        if ring_id[i] < 0:
            i += 1
            continue
        j = i
        while j < N and ring_id[j] == ring_id[i]:
            j += 1
        if (j - i) * DT >= 1.0:
            ring = rm.rings[ring_id[i]]
            states[max(0, i - ENTRY_EXIT_STEPS):i] = 2
            states[i:j] = 3
            states[j:min(N, j + ENTRY_EXIT_STEPS)] = 4
            a0, a1 = ring_arc(ring, d.pos[i]), ring_arc(ring, d.pos[j - 1])
            travelled = float(np.sum(d.v[i:j]) * DT)
            circ = ring["circumference"]
            arc_deg = travelled / circ * 360.0
            span = (a1 - a0) % circ
            passed = [x for x in ring["exits"] if 0.5 < (x["arc"] - a0) % circ <= span + 3.0]
            exit_no = len(passed) if passed else 1
            hb = d.psi[max(0, i - ENTRY_EXIT_STEPS)]
            ha = d.psi[min(N - 1, j + ENTRY_EXIT_STEPS - 1)]
            turn = float(np.degrees(wrap(ha - hb)))
            if arc_deg > 300:
                man = "full_circle"
            elif abs(turn) > 150:
                man = "u_turn"
            elif abs(turn) < 35:
                man = "straight"
            else:
                man = "right" if turn > 0 else "left"
            trav.append({"start": int(i), "end": int(j), "ring": int(ring_id[i]), "radius": ring["r"],
                         "ring_rms": ring["rms"], "arc_deg": arc_deg, "exit_no": int(exit_no),
                         "n_exits": len(ring["exits"]), "turn_deg": turn, "manoeuvre": man})
        i = j
    return states, trav


def all_labels(drives, names) -> dict:
    out = {}
    for n in names:
        if not (OSM_DIR / f"{n}.pkl").exists():
            continue
        try:
            rm = drive_roadmap(n)
        except KeyError:            # cache built before ring support: rebuild with `python -m idr.osm_map <drive>`
            continue
        states, trav = label_drive(drives[n], rm)
        out[n] = {"states": states, "traversals": trav}
    return out


if __name__ == "__main__":
    from idr.dataset import load_drives, split_names
    drives, _ = load_drives()
    sp = split_names(list(drives))
    summary = {}
    for split, names in sp.items():
        lab = all_labels(drives, names)
        trav = [t for v in lab.values() for t in v["traversals"]]
        valid = [t for n, v in lab.items() for t in v["traversals"] if drives[n].valid[t["start"]:t["end"]].all()]
        man = {}
        for t in valid:
            man[t["manoeuvre"]] = man.get(t["manoeuvre"], 0) + 1
        exits = {}
        for t in valid:
            exits[t["exit_no"]] = exits.get(t["exit_no"], 0) + 1
        missing = [n for n in names if n not in lab]
        print(f"{split:9s} drives_with_map={len(lab):2d}/{len(names):2d}  traversals={len(trav):3d} "
              f"(with valid phone IMU: {len(valid):3d})  manoeuvres={man}  exit_no={dict(sorted(exits.items()))}"
              + (f"  [no map yet: {missing}]" if missing else ""))
        summary[split] = {"traversals": len(trav), "valid": len(valid), "manoeuvres": man}
    print(json.dumps(summary))
