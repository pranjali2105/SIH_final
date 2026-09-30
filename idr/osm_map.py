"""OSM road network per drive, in that drive's ENU frame. `python -m idr.osm_map [drive ...]`.

Directed edges are densified every 5 m into flat arrays (position, arc length, bearing, curvature) with a KD-tree for
candidate lookup. Two-way roads get one edge per direction. Planned route = shortest path from the road at the
outage start to the road at the trip's real end point (what the app's router would return).
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from idr.data import EARTH_R, enu_from_latlon, load_all

OSM_DIR = Path(__file__).resolve().parent / "cache" / "osm"
STEP = 5.0
CORRIDOR_M = 1500.0
DRIVABLE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
            "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street"}
IMPLIED_ONEWAY = {"motorway", "motorway_link"}
LANE_W = {"motorway": 7.5, "trunk": 7.0, "primary": 6.0, "secondary": 5.5, "tertiary": 5.0}


@dataclass
class RoadMap:
    pts: np.ndarray          # (M,2) ENU
    edge: np.ndarray         # (M,) edge id of each point
    s: np.ndarray            # (M,) arc length along its edge
    bearing: np.ndarray      # (M,) clockwise from North
    curv: np.ndarray         # (M,) rad/m, positive = turning clockwise (right)
    e_u: np.ndarray          # (E,) from-node
    e_v: np.ndarray          # (E,) to-node
    e_start: np.ndarray      # (E,) index of first point
    e_n: np.ndarray          # (E,) number of points
    e_len: np.ndarray        # (E,) length m
    e_width: np.ndarray      # (E,) half road width m
    e_rev: np.ndarray        # (E,) id of the opposite-direction twin, -1 if oneway
    e_ring: np.ndarray       # (E,) roundabout ring id of the edge, -1 if not on a roundabout
    out_edges: dict = field(default_factory=dict)   # node -> list of edge ids leaving it
    rings: list = field(default_factory=list)       # per ring: center, radius, direction, arc positions, exits
    node_deg: dict = field(default_factory=dict)
    tree: cKDTree | None = None
    graph: nx.DiGraph | None = None

    def build_index(self):
        self.tree = cKDTree(self.pts)
        g = nx.DiGraph()
        for e in range(len(self.e_u)):
            g.add_edge(int(self.e_u[e]), int(self.e_v[e]), eid=e, length=float(self.e_len[e]))
        self.graph = g
        return self

    def nearest(self, xy, k=1):
        return self.tree.query(xy, k=k)

    def edge_point(self, e: int, s: float):
        """Interpolated (x, y, bearing, curvature) at arc length s on edge e (clamped)."""
        i0, n = int(self.e_start[e]), int(self.e_n[e])
        j = int(np.clip(s // STEP, 0, n - 2)) if n > 1 else 0
        a = i0 + j
        b = min(a + 1, i0 + n - 1)
        f = float(np.clip((s - self.s[a]) / max(self.s[b] - self.s[a], 1e-6), 0.0, 1.0))
        p = self.pts[a] * (1 - f) + self.pts[b] * f
        return p, self.bearing[a], self.curv[a]


class _Collector:
    def __init__(self, bbox):
        self.bbox = bbox          # lat0, lat1, lon0, lon1
        self.ways = {}

    def run(self, path):
        import osmium
        lat0, lat1, lon0, lon1 = self.bbox
        fp = osmium.FileProcessor(str(path)).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
        for obj in fp:
            if not obj.is_way():
                continue
            hw = obj.tags.get("highway")
            if hw not in DRIVABLE or obj.id in self.ways:
                continue
            try:
                coords = [(n.ref, n.lat, n.lon) for n in obj.nodes]
            except Exception:
                continue
            la = np.array([c[1] for c in coords])
            lo = np.array([c[2] for c in coords])
            if not ((la >= lat0) & (la <= lat1) & (lo >= lon0) & (lo <= lon1)).any():
                continue
            ow = obj.tags.get("oneway", "")
            oneway = 1 if ow in ("yes", "true", "1") else -1 if ow == "-1" else 0
            ring = obj.tags.get("junction") in ("roundabout", "circular")
            if oneway == 0 and (hw in IMPLIED_ONEWAY or ring):
                oneway = 1
            self.ways[obj.id] = (hw, oneway, coords, ring)


def load_ways(bbox):
    col = _Collector(bbox)
    files = sorted(OSM_DIR.glob("*.osm.pbf"))
    for f in files:
        col.run(f)
    return col.ways, [f.name for f in files]


def _densify(xy: np.ndarray):
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    grid = np.linspace(0.0, L, max(2, int(np.ceil(L / STEP)) + 1))
    x, y = np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])
    b = np.arctan2(np.diff(x), np.diff(y))
    b = np.concatenate([b, b[-1:]])
    db = (np.diff(b) + np.pi) % (2 * np.pi) - np.pi
    ds = np.maximum(np.diff(grid), 1e-3)
    k = np.concatenate([db / ds, [0.0]])
    return np.column_stack([x, y]), grid, b, k


def build_roadmap(track_latlon: np.ndarray, lat_ref: float, lon_ref: float, ways) -> RoadMap:
    """Crop ways to a corridor around the track and build directed, densified edges in ENU (origin lat/lon_ref)."""
    te, tn = enu_from_latlon(track_latlon[:, 0], track_latlon[:, 1], lat_ref, lon_ref)
    tk = cKDTree(np.column_stack([te, tn])[::20])
    node_use: dict[int, int] = {}
    kept = []
    for wid, (hw, oneway, coords, ring) in ways.items():
        la = np.array([c[1] for c in coords])
        lo = np.array([c[2] for c in coords])
        e, n = enu_from_latlon(la, lo, lat_ref, lon_ref)
        d, _ = tk.query(np.column_stack([e, n]), distance_upper_bound=CORRIDOR_M)
        if not np.isfinite(d).any():
            continue
        refs = [c[0] for c in coords]
        kept.append((hw, oneway, refs, np.column_stack([e, n]), ring))
        for r in refs:
            node_use[r] = node_use.get(r, 0) + 1
    return assemble(kept, node_use)


def assemble(kept, node_use) -> RoadMap:
    """kept: list of (highway, oneway, node_refs, xy ENU (n,2)[, is_roundabout]); node_use: ref -> #ways using it."""
    pts, eid, ss, brg, crv = [], [], [], [], []
    e_u, e_v, e_start, e_n, e_len, e_w, e_rev, e_isring = [], [], [], [], [], [], [], []
    count = 0

    def add_edge(u, v, xy, hw, ring=False):
        nonlocal count
        P, S, B, K = _densify(xy)
        e_u.append(u); e_v.append(v); e_start.append(count); e_n.append(len(P)); e_len.append(S[-1])
        e_w.append(LANE_W.get(hw.replace("_link", ""), 4.0) / 2 + 1.0)
        e_rev.append(-1)
        e_isring.append(bool(ring))
        pts.append(P); ss.append(S); brg.append(B); crv.append(K)
        eid.append(np.full(len(P), len(e_u) - 1))
        count += len(P)
        return len(e_u) - 1

    for item in kept:
        hw, oneway, refs, xy = item[:4]
        ring = item[4] if len(item) > 4 else False
        # split the way at shared nodes (junctions) and its ends
        cut = [0] + [i for i in range(1, len(refs) - 1) if node_use[refs[i]] > 1] + [len(refs) - 1]
        for a, b in zip(cut[:-1], cut[1:]):
            seg = xy[a:b + 1]
            if len(seg) < 2 or np.hypot(*(seg[-1] - seg[0])) < 0.5 and len(seg) == 2:
                continue
            u, v = refs[a], refs[b]
            f = r = -1
            if oneway >= 0:
                f = add_edge(u, v, seg, hw, ring) if oneway == 1 or oneway == 0 else -1
            if oneway == 0:
                r = add_edge(v, u, seg[::-1].copy(), hw, ring)
                e_rev[f], e_rev[r] = r, f
            if oneway == -1:
                add_edge(v, u, seg[::-1].copy(), hw, ring)
    rm = RoadMap(np.concatenate(pts), np.concatenate(eid), np.concatenate(ss), np.concatenate(brg),
                 np.concatenate(crv), np.array(e_u), np.array(e_v), np.array(e_start), np.array(e_n),
                 np.array(e_len), np.array(e_w), np.array(e_rev), np.full(len(e_u), -1))
    for e, u in enumerate(e_u):
        rm.out_edges.setdefault(int(u), []).append(e)
    for u, v in zip(e_u, e_v):
        rm.node_deg[int(u)] = rm.node_deg.get(int(u), 0) + 1
        rm.node_deg[int(v)] = rm.node_deg.get(int(v), 0) + 1
    extract_rings(rm, np.array(e_isring, dtype=bool))
    return rm.build_index()


def extract_rings(rm: RoadMap, is_ring: np.ndarray):
    """Group roundabout edges into rings; fit center/radius; arc position of each ring node; entries and exits."""
    from idr.circle_fit import kasa, refine
    g = nx.Graph()
    for e in np.flatnonzero(is_ring):
        g.add_edge(int(rm.e_u[e]), int(rm.e_v[e]))
    rings = []
    for comp in nx.connected_components(g):
        edges = [int(e) for e in np.flatnonzero(is_ring) if int(rm.e_u[e]) in comp]
        idx = np.concatenate([np.arange(rm.e_start[e], rm.e_start[e] + rm.e_n[e]) for e in edges])
        p = rm.pts[idx]
        if len(p) < 6:
            continue
        (cx, cy, r), _, rms = refine(p, kasa(p))
        if not (4.0 < r < 150.0):
            continue
        direction = 1.0 if np.nanmean(rm.curv[idx]) > 0 else -1.0     # +1 = clockwise (UK)
        ref = np.array([cx, cy])

        def arc_of(xy):
            ang = np.arctan2(xy[0] - cx, xy[1] - cy)                    # bearing of radius vector
            return float((ang * direction) % (2 * np.pi) * r)

        node_arc = {}
        for e in edges:
            node_arc[int(rm.e_u[e])] = arc_of(rm.pts[rm.e_start[e]])
            node_arc[int(rm.e_v[e])] = arc_of(rm.pts[rm.e_start[e] + rm.e_n[e] - 1])
        rid = len(rings)
        rm.e_ring[edges] = rid
        exits, entries = [], []
        for e in range(len(rm.e_u)):
            if is_ring[e]:
                continue
            if int(rm.e_u[e]) in comp:
                exits.append({"edge": e, "arc": node_arc.get(int(rm.e_u[e]), 0.0),
                              "bearing": float(rm.bearing[rm.e_start[e]])})
            if int(rm.e_v[e]) in comp:
                entries.append({"edge": e, "arc": node_arc.get(int(rm.e_v[e]), 0.0)})
        exits.sort(key=lambda x: x["arc"])
        entries.sort(key=lambda x: x["arc"])
        rings.append({"id": rid, "cx": float(cx), "cy": float(cy), "r": float(r), "rms": float(rms),
                      "dir": direction, "circumference": float(2 * np.pi * r), "nodes": sorted(comp),
                      "node_arc": node_arc, "edges": edges, "exits": exits, "entries": entries})
    rm.rings = rings


def ring_arc(ring: dict, xy) -> float:
    """Arc position (m) of a point along a ring, in its circulation direction."""
    ang = np.arctan2(xy[0] - ring["cx"], xy[1] - ring["cy"])
    return float((ang * ring["dir"]) % (2 * np.pi) * ring["r"])


def planned_route(rm: RoadMap, start_xy, start_heading, end_xy) -> list[int]:
    """Shortest path (edge ids) from the road matching the start pose to the road nearest the destination."""
    d, idx = rm.nearest(start_xy, k=16)
    best, bestcost = None, np.inf
    for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if not np.isfinite(di):
            continue
        dh = abs((rm.bearing[ii] - start_heading + np.pi) % (2 * np.pi) - np.pi)
        cost = di + 30.0 * dh
        if cost < bestcost:
            best, bestcost = int(rm.edge[ii]), cost
    _, j = rm.nearest(end_xy)
    goal = int(rm.edge[j])
    if best is None:
        return []
    try:
        nodes = nx.shortest_path(rm.graph, int(rm.e_v[best]), int(rm.e_u[goal]), weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [best]
    return [best] + [rm.graph[a][b]["eid"] for a, b in zip(nodes[:-1], nodes[1:])] + [goal]


def drive_roadmap(name: str, refresh: bool = False) -> RoadMap:
    cache = OSM_DIR / f"{name}.pkl"
    if cache.exists() and not refresh:
        return _from_state(pickle.loads(cache.read_bytes()))
    df = load_all()[name]
    ll = df[["lat", "lon"]].to_numpy()
    pad = CORRIDOR_M / EARTH_R * 180 / np.pi
    bbox = (ll[:, 0].min() - pad, ll[:, 0].max() + pad, ll[:, 1].min() - 2 * pad, ll[:, 1].max() + 2 * pad)
    ways, files = load_ways(bbox)
    rm = build_roadmap(ll, float(df["lat"].iloc[0]), float(df["lon"].iloc[0]), ways)
    cache.write_bytes(pickle.dumps(_to_state(rm)))
    return rm


_ARRAYS = ["pts", "edge", "s", "bearing", "curv", "e_u", "e_v", "e_start", "e_n", "e_len", "e_width", "e_rev",
           "e_ring"]


def _to_state(rm: RoadMap) -> dict:
    """Plain-data cache (arrays + dicts) so it loads regardless of which module pickled it."""
    return {**{k: getattr(rm, k) for k in _ARRAYS}, "out_edges": rm.out_edges, "node_deg": rm.node_deg,
            "rings": rm.rings}


def _from_state(st: dict) -> RoadMap:
    rm = RoadMap(*[st[k] for k in _ARRAYS])
    rm.out_edges, rm.node_deg, rm.rings = st["out_edges"], st["node_deg"], st.get("rings", [])
    return rm.build_index()


def coverage(name: str, rm: RoadMap, tol=15.0) -> float:
    df = load_all()[name]
    m = (df["v_true"] > 2).to_numpy()
    e, n = enu_from_latlon(df["lat"].to_numpy(), df["lon"].to_numpy(), df["lat"].iloc[0], df["lon"].iloc[0])
    d, _ = rm.tree.query(np.column_stack([e, n])[m])
    return float((d < tol).mean())


if __name__ == "__main__":
    names = sys.argv[1:] or ["Y1", "M"]
    for n in names:
        try:
            rm = drive_roadmap(n, refresh=True)
        except ValueError as ex:          # no drivable roads in the downloaded extracts for this drive
            print(f"{n}: no map ({ex})", flush=True)
            continue
        cov = coverage(n, rm)
        df = load_all()[n]
        e, nn = enu_from_latlon(df["lat"].to_numpy(), df["lon"].to_numpy(), df["lat"].iloc[0], df["lon"].iloc[0])
        route = planned_route(rm, np.array([e[0], nn[0]]), float(df["psi_true"].iloc[0]), np.array([e[-1], nn[-1]]))
        print(f"{n}: edges={len(rm.e_u)} points={len(rm.pts)} coverage(<15 m)={cov * 100:.1f}% "
              f"planned route edges={len(route)} length={sum(rm.e_len[route]) / 1000:.1f} km", flush=True)
        rr = [r for r in rm.rings]
        if rr:
            print(f"   rings={len(rr)}  radius median {np.median([r['r'] for r in rr]):.1f} m  "
                  f"fit rms median {np.median([r['rms'] for r in rr]):.2f} m (max {max(r['rms'] for r in rr):.2f})  "
                  f"clockwise {np.mean([r['dir'] > 0 for r in rr]) * 100:.0f}%  exits/ring median "
                  f"{np.median([len(r['exits']) for r in rr]):.0f}")
