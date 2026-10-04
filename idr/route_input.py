"""Route input for the TFT during evaluation.

    osm_planned_route     (DEFAULT, primary) shortest drivable route on the directed OSM graph from the last GNSS fix
                          strictly before outage start (projected with the same heading-aware rule the map matcher
                          uses) to the user-entered destination (projected to the nearest road). Bearings every 20 m
                          -> the TFT's existing route tensor (idr.route_features). Never compared with the future track.
    oracle_future_gnss    legacy: next 2 km of bearings of the TRUE driven GNSS track. Future information.
                          Diagnostic only: requires allow_oracle_diagnostic=True.
    none                  route features masked (destination features kept, route_available = 0).
    map_match_hypothesis  reserved, not implemented (raises).

Routing failure (no projection / no path) is never repaired: route features are masked and route_valid = 0.
Route metadata compares the planned route with the ground-truth path AFTER the features are built; it is
evaluation-only and never feeds the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from idr import route_features as rf

ROUTE_SOURCES = ("osm_planned_route", "oracle_future_gnss", "none", "map_match_hypothesis")
DEFAULT_SOURCE = "osm_planned_route"
PLANNER = "networkx.shortest_path(weight='length') on directed OSM edges; heading-aware origin projection"
ORACLE_ROUTE_WARNING = ("*** route_input_source=oracle_future_gnss: the TFT receives the next 2 km of the TRUE driven "
                        "track. Diagnostic only; not deployment-realistic. ***")


@dataclass(frozen=True)
class RouteConfig:
    source: str = DEFAULT_SOURCE
    allow_oracle_diagnostic: bool = False
    origin_max_dist_m: float = 50.0          # origin projection must be within this of a road
    dest_max_dist_m: float = 500.0           # snap the destination to the nearest REACHABLE road within this
    diverge_m: float = 20.0                  # metadata: route leaves the true path by more than this

    def __post_init__(self):
        if self.source not in ROUTE_SOURCES:
            raise ValueError(f"route_input_source must be one of {ROUTE_SOURCES}, got {self.source!r}")
        if self.source == "map_match_hypothesis":
            raise NotImplementedError("map_match_hypothesis route input is reserved and not implemented")
        if self.source == "oracle_future_gnss" and not self.allow_oracle_diagnostic:
            raise ValueError("oracle_future_gnss route input is diagnostic only: pass --allow_oracle_route_diagnostic")


def add_cli(ap) -> None:
    ap.add_argument("--route_input_source", default=DEFAULT_SOURCE, choices=ROUTE_SOURCES)
    ap.add_argument("--allow_oracle_route_diagnostic", action="store_true",
                    help="required to use route_input_source=oracle_future_gnss (future information)")


def config_from_args(args) -> RouteConfig:
    cfg = RouteConfig(args.route_input_source, args.allow_oracle_route_diagnostic)
    if cfg.source == "oracle_future_gnss":
        print("\n" + ORACLE_ROUTE_WARNING + "\n", flush=True)
    return cfg


def _project_origin(rm, p0, psi0, max_d):
    d, idx = rm.nearest(p0, k=16)
    best, cost_best = None, np.inf
    for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if not np.isfinite(di) or di > max_d:
            continue
        dh = abs((rm.bearing[ii] - psi0 + np.pi) % (2 * np.pi) - np.pi)
        cost = di + 30.0 * dh                                   # same rule as osm_map.planned_route
        if cost < cost_best:
            best, cost_best = int(ii), cost
    return best


def plan_route(rm, p0, psi0, dest, cfg: RouteConfig | None = None) -> dict:
    """OSM route polyline from (p0, psi0) to dest. Uses ONLY these three inputs and the map."""
    cfg = cfg or RouteConfig()
    out = {"route_valid": False, "route_success": False, "reason": None, "edges": [], "polyline": None,
           "route_length_m": 0.0}
    if rm is None or not (np.all(np.isfinite(p0)) and np.isfinite(psi0) and np.all(np.isfinite(dest))):
        out["reason"] = "invalid_inputs"
        return out
    i0 = _project_origin(rm, np.asarray(p0, float), float(psi0), cfg.origin_max_dist_m)
    if i0 is None:
        out["reason"] = "origin_not_on_map"
        return out
    e0, s0 = int(rm.edge[i0]), float(rm.s[i0])
    # destination: nearest REACHABLE road sample within dest_max_dist_m (standard snap-to-routable-road;
    # one shortest-path tree from the origin, no ground truth involved)
    dest = np.asarray(dest, float)
    jj = rm.tree.query_ball_point(dest, cfg.dest_max_dist_m)
    cand = [(float(np.hypot(*(rm.pts[j] - dest))), int(j)) for j in jj]
    if not cand:
        out["reason"] = "destination_not_on_map"
        return out
    src = int(rm.e_v[e0])
    try:
        dist_tree, path_tree = nx.single_source_dijkstra(rm.graph, src, weight="length")
    except nx.NodeNotFound:
        out["reason"] = "no_path"
        return out
    pts = lambda e: rm.pts[rm.e_start[e]:rm.e_start[e] + rm.e_n[e]]
    ss = lambda e: rm.s[rm.e_start[e]:rm.e_start[e] + rm.e_n[e]]
    poly, edges, snap = None, None, None
    for dist_d, j in sorted(cand):
        eg, sg = int(rm.edge[j]), float(rm.s[j])
        if eg == e0 and sg >= s0:
            poly, edges = pts(e0)[(ss(e0) >= s0) & (ss(e0) <= sg)], [e0]
        elif int(rm.e_u[eg]) in path_tree:
            nodes = path_tree[int(rm.e_u[eg])]
            mid = [rm.graph[a][b]["eid"] for a, b in zip(nodes[:-1], nodes[1:])]
            edges = [e0] + mid + [eg]
            parts = [pts(e0)[ss(e0) >= s0]] + [pts(e) for e in mid] + [pts(eg)[ss(eg) <= sg]]
            poly = np.vstack([q for q in parts if len(q)])
        else:
            continue
        snap = dist_d
        break
    if poly is None:
        out["reason"] = "no_path"
        return out
    out["destination_snap_m"] = snap
    if poly is None or len(poly) < 2:
        out["reason"] = "degenerate_route"
        return out
    keep = np.r_[True, np.hypot(*np.diff(poly, axis=0).T) > 1e-6]
    poly = poly[keep]
    out.update({"route_valid": True, "route_success": True, "reason": "ok", "edges": edges, "polyline": poly,
                "route_length_m": float(np.sum(np.hypot(*np.diff(poly, axis=0).T)))})
    return out


def route_bearings(plan: dict):
    """(bearings every 20 m from the route start, number of valid points up to the destination)."""
    poly = plan["polyline"]
    e, n, _ = rf.resample_polyline(poly[:, 0], poly[:, 1])
    b = rf.polyline_bearings(e, n)
    return b, len(b)


def route_metadata(plan: dict, cfg: RouteConfig, source: str, p0, dest, gt_path, outage_dist_m) -> dict:
    """Evaluation-only comparison of the planned route with the ground-truth path from outage start."""
    meta = {"route_input_source": source, "route_origin": [float(x) for x in p0],
            "route_destination": [float(x) for x in dest], "route_planner": PLANNER if source == "osm_planned_route"
            else ("ground-truth GNSS track" if source == "oracle_future_gnss" else None),
            "route_success": bool(plan.get("route_success", False)), "route_valid": int(plan.get("route_valid", False)),
            "route_length_m": float(plan.get("route_length_m", 0.0)), "route_failure_reason": plan.get("reason"),
            "destination_snap_m": plan.get("destination_snap_m"),
            "route_first_divergence_m": None, "route_matches_ground_truth_route": None}
    if source != "osm_planned_route" or not plan.get("route_valid"):
        if source == "oracle_future_gnss":
            meta["route_matches_ground_truth_route"] = True
        return meta
    gt = np.asarray(gt_path, float)
    gt = gt[np.all(np.isfinite(gt), axis=1)]
    if len(gt) < 2:
        return meta
    poly = plan["polyline"]
    e, n, s = rf.resample_polyline(poly[:, 0], poly[:, 1], step=5.0)
    gt_len = float(np.sum(np.hypot(*np.diff(gt, axis=0).T)))
    comp = s <= min(gt_len, s[-1] if len(s) else 0.0)
    dist, _ = cKDTree(gt).query(np.column_stack([e, n]))
    off = np.flatnonzero(comp & (dist > cfg.diverge_m))
    meta["route_first_divergence_m"] = float(s[off[0]]) if len(off) else None
    meta["route_matches_ground_truth_route"] = bool(len(off) == 0 or s[off[0]] >= outage_dist_m)
    return meta


def annotate_specs(drives, specs, cfg: RouteConfig, roadmaps=None, horizon=600):
    """Attach the route input to fixed eval specs. Returns (specs, metas). Specs not in route mode are untouched.
    osm_planned_route: spec gets route_b / route_n_valid (or route_mode='dest' when routing fails).
    none: route_mode='dest'. oracle_future_gnss: unchanged (dataset reads the true track)."""
    from idr.osm_map import drive_roadmap
    roadmaps = roadmaps if roadmaps is not None else {}
    out, metas = [], []
    for sp in specs:
        d = drives[sp["drive"]]
        t0, td = int(sp["t0"]), int(sp["td"])
        p0, psi0, dest = d.pos[t0 - 1], float(d.psi[t0 - 1]), d.pos[td]
        gt = d.pos[t0:]
        odist = float(np.nansum(d.v[t0:t0 + horizon]) * 0.1)
        sp2 = dict(sp)
        if sp.get("route_mode") != "route":
            metas.append({"route_input_source": f"not_route_mode:{sp.get('route_mode')}", "route_valid": 0})
            out.append(sp2)
            continue
        if cfg.source == "oracle_future_gnss":
            plan = {"route_valid": True, "route_success": True, "reason": "oracle"}
        elif cfg.source == "none":
            plan = {"route_valid": False, "route_success": False, "reason": "route_input_disabled"}
            sp2["route_mode"] = "dest"
        else:
            name = sp["drive"]
            if name not in roadmaps:
                roadmaps[name] = drive_roadmap(name)
            plan = plan_route(roadmaps[name], p0, psi0, dest, cfg)
            if plan["route_valid"]:
                b, nv = route_bearings(plan)
                sp2["route_b"], sp2["route_n_valid"] = b.tolist()[:rf.N_PTS + 5], int(nv)
            else:
                sp2["route_mode"] = "dest"                      # masked, never repaired with the true track
        sp2["route_input_source"] = cfg.source
        metas.append(route_metadata(plan, cfg, cfg.source, p0, dest, gt, odist))
        out.append(sp2)
    return out, metas
