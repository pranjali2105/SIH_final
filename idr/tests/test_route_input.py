"""TFT route-input source: leakage regression tests. `python -m pytest idr/tests/test_route_input.py -q`"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pytest

from idr import ablation_table
from idr.dataset import HORIZON, OutageDataset, load_drives
from idr.evaluate import make_specs
from idr.map_matching import _synthetic_map
from idr.route_input import (DEFAULT_SOURCE, RouteConfig, add_cli, annotate_specs, config_from_args, plan_route,
                             route_bearings, route_metadata)

OSM = RouteConfig()


@pytest.fixture(scope="module")
def s2():
    drives, _ = load_drives(["S2"])
    specs = make_specs(drives, ["S2"], HORIZON, stride=600)
    return drives, specs


def items(drives, specs):
    ds = OutageDataset(drives, list(drives), len(specs), mode="eval")
    ds.fixed = specs
    return [ds[i] for i in range(len(specs))]


def poison(d, t0):
    """All GNSS-derived ground truth at/after outage start -> NaN, except the user-entered destination (last row)."""
    v, psi, pos, s = d.v.copy(), d.psi.copy(), d.pos.copy(), d.s.copy()
    dest = pos[-1].copy()
    v[t0:], psi[t0:], pos[t0:], s[t0:] = np.nan, np.nan, np.nan, np.nan
    pos[-1] = dest
    return dataclasses.replace(d, v=v, psi=psi, pos=pos, s=s, grid_b=np.full_like(d.grid_b, np.nan))


# (a) future GNSS poison: the OSM planned-route input is unchanged
def test_a_future_gnss_poison_leaves_osm_route_unchanged(s2):
    drives, specs = s2
    sp = specs[5:9]
    a, ma = annotate_specs(drives, sp, OSM)
    for spec in sp:
        drives_p = {"S2": poison(drives["S2"], spec["t0"])}
        b, mb = annotate_specs(drives_p, [spec], OSM)
        i = sp.index(spec)
        assert a[i].get("route_b") == b[0].get("route_b") and a[i].get("route_n_valid") == b[0].get("route_n_valid")
        assert a[i]["route_mode"] == b[0]["route_mode"]
        ia, ib = items(drives, [a[i]])[0], items(drives_p, b)[0]
        for k in ("route", "static", "seq", "u_raw"):
            np.testing.assert_array_equal(ia[k].numpy(), ib[k].numpy(), err_msg=k)
        assert ma[i]["route_origin"] == mb[0]["route_origin"] and ma[i]["route_destination"] == mb[0]["route_destination"]
    assert sum(m["route_valid"] for m in ma) >= 3


# (b) primary evaluation refuses the oracle route unless explicitly requested as a diagnostic
def test_b_oracle_route_refused_unless_explicit():
    assert RouteConfig().source == DEFAULT_SOURCE == "osm_planned_route"
    with pytest.raises(ValueError):
        RouteConfig("oracle_future_gnss")
    assert RouteConfig("oracle_future_gnss", allow_oracle_diagnostic=True).source == "oracle_future_gnss"
    with pytest.raises(NotImplementedError):
        RouteConfig("map_match_hypothesis")
    ap = argparse.ArgumentParser()
    add_cli(ap)
    assert config_from_args(ap.parse_args([])).source == "osm_planned_route"
    with pytest.raises(ValueError):
        config_from_args(ap.parse_args(["--route_input_source", "oracle_future_gnss"]))
    assert config_from_args(ap.parse_args(["--route_input_source", "oracle_future_gnss",
                                           "--allow_oracle_route_diagnostic"])).source == "oracle_future_gnss"


# (c) the route is generated from the pre-outage fix and the destination only
def test_c_route_from_pre_outage_origin_and_destination(s2):
    rm = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 0), ([2, 3], [[0, 1000], [800, 1000]], "secondary", 0)])
    plan = plan_route(rm, np.array([2.0, 100.0]), 0.0, np.array([700.0, 1003.0]))
    assert plan["route_valid"] and plan["reason"] == "ok"
    poly = plan["polyline"]
    assert np.hypot(*(poly[0] - [0, 100])) < 6 and np.hypot(*(poly[-1] - [700, 1000])) < 6
    assert abs(plan["route_length_m"] - (900 + 700)) < 15
    other = plan_route(rm, np.array([2.0, 100.0]), 0.0, np.array([0.0, 600.0]))
    assert other["route_length_m"] < plan["route_length_m"] - 500          # destination drives the route
    drives, specs = s2
    sp = specs[5]
    _, meta = annotate_specs(drives, [sp], OSM)
    d = drives["S2"]
    assert meta[0]["route_origin"] == [float(x) for x in d.pos[sp["t0"] - 1]]
    assert meta[0]["route_destination"] == [float(x) for x in d.pos[sp["td"]]]


# (d) no silent fallback: a routing failure masks the route features and sets route_valid = 0
def test_d_routing_failure_masks_route(s2):
    split = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 1),           # one-way north
                            ([3, 4], [[5000, 0], [5000, 1000]], "secondary", 0)])     # unconnected road
    plan = plan_route(split, np.array([0.0, 100.0]), 0.0, np.array([5000.0, 500.0]))
    assert not plan["route_valid"] and plan["reason"] == "no_path"
    assert plan_route(split, np.array([900.0, 100.0]), 0.0, np.array([0.0, 900.0]))["reason"] == "origin_not_on_map"
    drives, specs = s2
    sp = specs[5]
    out, meta = annotate_specs(drives, [sp], OSM, roadmaps={"S2": split})              # S2 is nowhere near it
    assert meta[0]["route_valid"] == 0 and meta[0]["route_success"] is False
    assert "route_b" not in out[0] and out[0]["route_mode"] == "dest"
    it = items(drives, out)[0]
    assert float(it["route"].abs().sum()) == 0.0 and float(it["static"][-1]) == 0.0    # masked, route_available = 0


# (e) a valid route that differs from what was actually driven is used as is, never replaced by the true path
def test_e_valid_but_different_route_is_kept():
    rm = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 0),
                         ([1, 3], [[0, 0], [300, 0]], "secondary", 0), ([3, 4], [[300, 0], [300, 1000]], "secondary", 0),
                         ([4, 2], [[300, 1000], [0, 1000]], "secondary", 0)])
    p0, dest = np.array([0.0, 10.0]), np.array([0.0, 990.0])
    plan = plan_route(rm, p0, 0.0, dest)
    b, _ = route_bearings(plan)
    assert np.allclose(np.sin(b[:30]), 0.0, atol=1e-6)                    # straight north, the shortest route
    detour = np.vstack([np.column_stack([np.linspace(0, 300, 31), np.zeros(31)]),
                        np.column_stack([np.full(101, 300.0), np.linspace(0, 1000, 101)]),
                        np.column_stack([np.linspace(300, 0, 31), np.full(31, 1000.0)])])
    meta = route_metadata(plan, OSM, "osm_planned_route", p0, dest, detour[1:], 800.0)
    assert meta["route_matches_ground_truth_route"] is False and meta["route_first_divergence_m"] is not None
    assert meta["route_first_divergence_m"] < 100


# (f) ablation_table refuses to treat oracle_future_gnss and osm_planned_route as equivalent conditions
def test_f_comparison_guard():
    assert ablation_table.route_source({"route": {"route_input_source": "osm_planned_route"}}) == "osm_planned_route"
    assert ablation_table.route_source({"summary": {}}) == "oracle_future_gnss"          # legacy files
    real = ablation_table.load
    try:
        ablation_table.load = lambda tag, td: {"drift": 9, "drift_rng": None, "wrong": 1, "cross": 1, "along": 1,
                                               "w25": 1, "first": None, "mode": "causal_pre_outage", "route": tag}
        runs = [("osm", "osm_planned_route", None), ("oracle", "oracle_future_gnss", None)]
        with pytest.raises(ValueError):
            ablation_table.table(runs)
        txt = ablation_table.table(runs, input_definition_change=True)
        assert "INPUT-DEFINITION CHANGE" in txt and "[route: oracle_future_gnss]" in txt
    finally:
        ablation_table.load = real


# planned_route (map-matcher route prior) uses the same planner and never returns a degenerate start-only route
def test_planned_route_multi_edge_or_empty(s2):
    from idr.osm_map import drive_roadmap, planned_route
    rm = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 0), ([2, 3], [[0, 1000], [800, 1000]], "secondary", 0),
                         ([3, 4], [[800, 1000], [800, 2000]], "secondary", 0)])
    r = planned_route(rm, np.array([2.0, 100.0]), 0.0, np.array([800.0, 1900.0]))
    assert len(r) >= 3 and len(set(r)) == len(r)
    split = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 1), ([3, 4], [[5000, 0], [5000, 1000]], "secondary", 0)])
    assert planned_route(split, np.array([0.0, 100.0]), 0.0, np.array([5000.0, 500.0])) == []
    drives, specs = s2                                         # the case that used to degenerate on every S2 outage
    d, sp = drives["S2"], specs[5]
    real = planned_route(drive_roadmap("S2"), d.pos[sp["t0"] - 1], float(d.psi[sp["t0"] - 1]), d.pos[-1])
    assert len(real) > 10
