"""Unit tests for the OSM look-ahead slowdown prior. `python -m pytest idr/tests/test_map_lookahead.py -q`"""
from __future__ import annotations

import numpy as np
import pytest

from idr.confidence import ConfidenceReport
from idr.map_lookahead import LookaheadConfig, LookaheadPrior, decel_prior, extract_lookahead, gate, target_speed
from idr.map_matching import _synthetic_map, run_outage

CFG = LookaheadConfig(enable_map_lookahead_speed_prior=True, prior_mode="acceleration")


def report(p1=0.95, margin=0.9, cross=2.0, dpsi=0.02, ess=0.6):
    return ConfidenceReport(ekf_confidence=1.0, map_confidence=1.0, route_confidence=margin, ess_ratio=ess,
                            cloud_sigma_m=5.0, pos_sigma_m=5.0, heading_sigma_deg=2.0, speed_scale_sigma=0.02,
                            dist_since_fix_m=0.0, nhc_nis=1.0, tft_resid=None, cross_track_residual_m=cross,
                            heading_residual_rad=dpsi, along_track_residual_m=0.0, best_route_probability=p1,
                            route_margin=margin)


def edge_into(rm, u_xy, v_xy):
    """Directed edge whose first point is u_xy and last point v_xy."""
    for e in range(len(rm.e_len)):
        a, b = rm.pts[rm.e_start[e]], rm.pts[rm.e_start[e] + rm.e_n[e] - 1]
        if np.hypot(*(a - u_xy)) < 1 and np.hypot(*(b - v_xy)) < 1:
            return e
    raise AssertionError("edge not found")


def prior_correction(la, v, rep=None):
    """Feed a look-ahead result into LookaheadPrior (as refresh() would) and return one step's correction."""
    p = LookaheadPrior(CFG)
    p.report = rep or report()
    p.passed, p.map_conf, p.reason = gate(p.report, CFG)
    p.events = [e for e in la.events if e.type in CFG.allowed_event_types] if p.passed else []
    return p.correction(v), p


@pytest.fixture(scope="module")
def maps():
    straight = _synthetic_map([([1, 2], [[0, 0], [0, 1000]], "secondary", 0)])
    elbow = _synthetic_map([([1, 2], [[0, -300], [0, 0]], "secondary", 0), ([2, 3], [[0, 0], [300, 0]], "secondary", 0)])
    tee = _synthetic_map([([1, 2], [[0, -300], [0, 0]], "secondary", 0), ([2, 3], [[0, 0], [0, 300]], "secondary", 0),
                          ([2, 4], [[0, 0], [300, 0]], "secondary", 0)])
    dead = _synthetic_map([([1, 2], [[0, -300], [0, 0]], "secondary", 1)])
    R = 20.0
    node_b = {1: np.pi, 2: 1.5 * np.pi, 3: 0.0, 4: 0.5 * np.pi}
    ring_pt = lambda b: np.array([R * np.sin(b), R * np.cos(b)])
    polys, order = [], [1, 2, 3, 4, 1]
    for a_, b_ in zip(order[:-1], order[1:]):
        ba, bb = node_b[a_], node_b[b_]
        bb = bb if bb > ba else bb + 2 * np.pi
        polys.append(([a_] + [1000 + 20 * a_ + k for k in range(14)] + [b_],
                      [ring_pt(x) for x in np.linspace(ba, bb, 16)], "primary", 1, True))
    for nid, b in node_b.items():
        polys.append(([nid, 100 + nid], [ring_pt(b), ring_pt(b) * 16], "primary", 0))
    rb = _synthetic_map(polys)
    return {"straight": straight, "elbow": elbow, "tee": tee, "dead": dead, "rb": (rb, ring_pt(np.pi))}


def test_1_straight_road_no_event(maps):
    rm = maps["straight"]
    e = edge_into(rm, np.array([0, 0]), np.array([0, 1000]))
    la = extract_lookahead(rm, e, 100.0, CFG)
    assert la.events == [] and la.max_abs_curvature < 1e-6
    a, _ = prior_correction(la, 15.0)
    assert a == 0.0


def test_2_sharp_turn_bounded_nonpositive(maps):
    rm = maps["elbow"]
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG)
    turns = [x for x in la.events if x.type == "turn"]
    assert turns and abs(abs(turns[0].turn_angle_rad) - np.pi / 2) < 0.1 and abs(turns[0].distance_m - 25) < 1
    a, p = prior_correction(la, 12.0)
    assert -CFG.max_decel_prior_mps2 <= a < 0.0
    assert la.distance_to_next_material_turn_m == pytest.approx(25.0, abs=1.0)


def test_3_roundabout_bounded_nonpositive(maps):
    rm, entry_xy = maps["rb"]
    e = edge_into(rm, entry_xy * 16, entry_xy)
    la = extract_lookahead(rm, e, rm.e_len[e] - 30.0, CFG)
    rbs = [x for x in la.events if x.type == "roundabout"]
    assert rbs and abs(rbs[0].distance_m - 30) < 1 and rbs[0].ring_r_m == pytest.approx(20.0, abs=1.0)
    assert la.distance_to_roundabout_entry_m == pytest.approx(30.0, abs=1.0)
    assert target_speed(rbs[0], 14.0, CFG) == pytest.approx(np.sqrt(2.5 * 20.0), rel=0.05)
    a, _ = prior_correction(la, 14.0)
    assert -CFG.max_decel_prior_mps2 <= a < 0.0


def test_4_event_beyond_horizon(maps):
    rm = maps["elbow"]
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 100.0, CFG)
    assert la.events == []
    a, _ = prior_correction(la, 15.0)
    assert a == 0.0
    # also: an event inside the horizon but closer than event_min_distance -> ramp 0
    assert decel_prior(15.0, 5.0, 3.0, CFG)[1] == 0.0


def test_5_low_map_confidence(maps):
    rm = maps["elbow"]
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG)
    for rep, why in [(report(p1=0.5), "low_route_confidence"), (report(margin=0.05), "low_route_margin"),
                     (report(cross=30.0), "cross_track_residual"), (report(dpsi=0.6), "heading_residual"),
                     (report(ess=0.02), "particle_ess")]:
        a, p = prior_correction(la, 12.0, rep)
        assert a == 0.0 and p.reason == why
    assert gate(None, CFG)[0] is False


def test_6_ambiguous_or_disconnected_successor(maps):
    rm = maps["tee"]                                  # straight-on AND right turn available, no planned route
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG)
    assert la.reason == "ambiguous_successor" and not [x for x in la.events if x.type in ("turn", "curve")]
    a, _ = prior_correction(la, 12.0)
    assert a == 0.0
    # with the right branch on the planned route the turn becomes a valid directed event
    right = edge_into(rm, np.array([0, 0]), np.array([300, 0]))
    la_r = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG, route=frozenset({right}))
    assert [x for x in la_r.events if x.type == "turn"]
    rm = maps["dead"]
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG)
    assert la.valid is False and la.reason == "disconnected" and la.events == []
    a, _ = prior_correction(la, 12.0)
    assert a == 0.0


def test_7_speed_below_target(maps):
    rm = maps["elbow"]
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 25.0, CFG)
    v_t = target_speed([x for x in la.events if x.type == "turn"][0], 30.0, CFG)
    a, p = prior_correction(la, v_t + 0.2)            # within minimum_speed_excess_mps
    assert a == 0.0 and p.rejects["speed_below_target"] == 1


def test_8_disabled_is_bit_for_bit(maps):
    rm = maps["elbow"]
    T = 500
    w = np.zeros(T); w[300:340] = -np.pi / 2 / 4.0           # 90 deg right turn
    a = np.zeros(T); a[250:300] = -1.0
    base = run_outage(rm, np.array([0.0, -280.0]), 0.0, 12.0, w, a, seed=3)
    off = run_outage(rm, np.array([0.0, -280.0]), 0.0, 12.0, w, a, seed=3,
                     lookahead={"enable_map_lookahead_speed_prior": False})
    np.testing.assert_array_equal(base[0], off[0])
    np.testing.assert_array_equal(base[1], off[1])
    assert base[2] == off[2]
    on = run_outage(rm, np.array([0.0, -280.0]), 0.0, 12.0, w, a, seed=3, return_trace=True,
                    lookahead={"enable_map_lookahead_speed_prior": True, "debug_trace": True,
                               "prior_mode": "acceleration"})
    tr = on[3]["lookahead_trace"]
    assert len(tr) == T and all(x["a_correction_mps2"] <= 0 for x in tr)
    assert all(x["a_correction_mps2"] >= -CFG.max_decel_prior_mps2 for x in tr)
    s = on[2]["lookahead"]
    assert s["total_speed_reduction_mps"] <= CFG.max_total_speed_reduction_per_outage_mps + 1e-9


def test_2b_curve_event_on_open_road():
    R = 50.0
    arc = [[R - R * np.cos(t), R * np.sin(t)] for t in np.linspace(0, np.pi / 2, 30)]
    rm = _synthetic_map([([1, 2], [[0, -300], [0, 0]] , "secondary", 0),
                         ([2] + [500 + i for i in range(28)] + [3], arc, "secondary", 0),
                         ([3, 4], [[R, R], [R + 300, R]], "secondary", 0)])
    e = edge_into(rm, np.array([0, -300]), np.array([0, 0]))
    la = extract_lookahead(rm, e, rm.e_len[e] - 20.0, CFG)
    curves = [x for x in la.events if x.type == "curve"]
    assert curves and not [x for x in la.events if x.type == "turn"]
    assert 1 / curves[0].kappa == pytest.approx(R, rel=0.25)          # R = 50 m recovered
    assert target_speed(curves[0], 20.0, CFG) == pytest.approx(np.sqrt(2.5 * R), rel=0.15)
    a, _ = prior_correction(la, 20.0)
    assert -CFG.max_decel_prior_mps2 <= a < 0.0



# ------------------------------------------------------------------ speed pseudo-measurement mode
def zigzag(n_turns=6, leg=150.0, v=14.0, R=8.0):
    """Synthetic road of straight legs joined by 90 deg left/right turns (each corner a node with one successor),
    plus a matching ground-truth drive (gyro consistent with the road) at constant speed v."""
    pts, d, h = [np.array([0.0, 0.0])], 1.0, 0.0
    corners = []
    for i in range(n_turns + 1):
        nxt = pts[-1] + leg * np.array([np.sin(h), np.cos(h)])
        pts.append(nxt)
        corners.append(nxt)
        h += (np.pi / 2) * (1 if i % 2 == 0 else -1)
    polys = [([10 + i, 11 + i], [pts[i].tolist(), pts[i + 1].tolist()], "secondary", 0) for i in range(len(pts) - 1)]
    rm = _synthetic_map(polys)
    # truth path with rounded corners of radius R
    path = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        u = (b - a) / np.linalg.norm(b - a)
        s0 = R if i > 0 else 0.0
        s1 = np.linalg.norm(b - a) - (R if i < len(pts) - 2 else 0.0)
        path += [a + u * x for x in np.arange(s0, s1, 0.5)]
    P = np.array(path)
    sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    tt = np.arange(0, sl[-1] / v, 0.1)
    Pt = np.column_stack([np.interp(tt * v, sl, P[:, 0]), np.interp(tt * v, sl, P[:, 1])])
    hd = np.unwrap(np.arctan2(np.gradient(Pt[:, 0]), np.gradient(Pt[:, 1])))
    return rm, Pt, -np.gradient(hd, 0.1), v


PSEUDO = {"enable_map_lookahead_speed_prior": True, "prior_mode": "speed_pseudo_measurement", "debug_trace": True,
          "lookahead_horizon_m": 100.0, "event_max_distance_m": 120.0, "heading_residual_max_deg": 20.0}


def run_zig(extra=None, seed=0):
    rm, Pt, yaw, v = zigzag()
    T = len(yaw)
    out = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=seed, return_trace=True,
                     lookahead={**PSEUDO, **(extra or {})})
    return out, rm, Pt


@pytest.fixture(scope="module")
def zig():
    return run_zig()


def test_p1_never_positive_and_active_only_in_approach_or_inside(zig):
    (tr, sn, st, trc), _, _ = zig
    t = trc["lookahead_trace"]
    assert st["lookahead"]["n_applications"] > 0                     # the scenario really exercises the prior
    assert all(x["delta_v_mps"] <= 0.0 for x in t)
    assert all(x["state"] in ("approach", "inside_event") for x in t if x["delta_v_mps"] < 0)


def test_p2_per_step_cap(zig):
    (_, _, _, trc), _, _ = zig
    assert min(x["delta_v_mps"] for x in trc["lookahead_trace"]) >= -0.10 - 1e-9


def test_p3_per_event_cap():
    (_, _, st, trc), _, _ = run_zig({"max_speed_reduction_per_event_mps": 0.3, "max_total_speed_reduction_per_outage_mps": 50.0})
    per = {}
    for x in trc["lookahead_trace"]:
        if x["delta_v_mps"] < 0:
            per[x["event_key"]] = per.get(x["event_key"], 0.0) - x["delta_v_mps"]
    assert per and max(per.values()) <= 0.3 + 1e-9
    assert st["lookahead"]["events_reaching_event_cap"] >= 1


def test_p4_and_p8_outage_cap_with_chained_events():
    (_, _, st, trc), _, _ = run_zig({"max_speed_reduction_per_event_mps": 1.0, "max_total_speed_reduction_per_outage_mps": 1.5})
    t = trc["lookahead_trace"]
    events = {x["event_key"] for x in t if x["delta_v_mps"] < 0}
    total = -sum(x["delta_v_mps"] for x in t)
    assert len(events) >= 2                                             # chained events
    assert total <= 1.5 + 1e-9 and st["lookahead"]["total_speed_reduction_mps"] <= 1.5 + 1e-9
    assert st["lookahead"]["outage_cap_reached"] is True
    first_cap = next(i for i, x in enumerate(t) if x["outage_reduction_mps"] >= 1.5 - 1e-9)
    assert all(x["delta_v_mps"] == 0.0 for x in t[first_cap + 1:])     # nothing after the cap


def test_p5_after_event_applies_nothing(zig):
    (_, _, _, trc), _, _ = zig
    after = [x for x in trc["lookahead_trace"] if x["state"] == "after_event"]
    assert after and all(x["delta_v_mps"] == 0.0 for x in after)


def test_p6_low_route_confidence_no_correction():
    (_, _, st, trc), _, _ = run_zig({"map_confidence_threshold": 1.01})
    assert st["lookahead"]["n_applications"] == 0
    assert all(x["delta_v_mps"] == 0.0 for x in trc["lookahead_trace"])


def test_p7_disabled_pseudo_mode_bit_identical():
    rm, Pt, yaw, v = zigzag()
    T = len(yaw)
    base = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=2)
    off = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=2,
                     lookahead={**PSEUDO, "enable_map_lookahead_speed_prior": False})
    np.testing.assert_array_equal(base[0], off[0])
    np.testing.assert_array_equal(base[1], off[1])


def test_p9_pseudo_update_survives_delayed_replay():
    from idr.nav_filter import IE, IN, IVF, NavEKF
    ekf = NavEKF(0.0, 0.0, 0.0, 12.0)
    ref = NavEKF(0.0, 0.0, 0.0, 12.0)
    for k in range(40):
        for f in (ekf, ref):
            f.propagate(0.0, 0.0)
        if k == 35:
            ekf.speed_pseudo_update(-0.5, 2.0)
    v_before = ekf.x[IVF]
    H = np.zeros((2, ekf.x.size)); H[0, IE] = H[1, IN] = 1.0
    ok = ekf.delayed_update(ekf.k - 20, ekf.state_at(ekf.k - 20)[0][[IE, IN]], H, np.eye(2) * 25.0)
    assert ok
    assert ekf.x[IVF] < ref.x[IVF] - 0.4                       # the -0.5 m/s survived the replay
    assert abs(ekf.x[IVF] - v_before) < 0.05


# ------------------------------------------------------------------ bounded post-event recovery
REC = {"enable_post_event_recovery": True}


@pytest.fixture(scope="module")
def zig_rec():
    return run_zig(REC)


def _cum(trace):
    red = np.cumsum([-min(x["delta_v_mps"], 0.0) for x in trace])
    rec = np.cumsum([max(x["delta_v_mps"], 0.0) for x in trace])
    return red, rec


def test_r_a_recovers_at_most_what_the_event_removed(zig_rec):
    (_, _, st, trc), _, _ = zig_rec
    ev = st["lookahead"]["events"]
    assert any(e["event_recovered_mps"] > 0 for e in ev)                  # recovery really happens
    for e in ev:
        assert e["event_recovered_mps"] <= e["event_reduction_mps"] + 1e-9
    per = {}
    for x in trc["lookahead_trace"]:
        if x["delta_v_mps"] > 0:
            per[x["recovery_event_key"]] = per.get(x["recovery_event_key"], 0.0) + x["delta_v_mps"]
    red = {e["event_key"]: e["event_reduction_mps"] for e in ev}
    assert all(v <= red[k] + 1e-9 for k, v in per.items())


def test_r_b_never_above_pre_event_speed(zig_rec):
    (_, _, st, trc), _, _ = zig_rec
    pre = {e["event_key"]: e["pre_event_speed_mps"] for e in st["lookahead"]["events"]}
    for x in trc["lookahead_trace"]:
        if x["delta_v_mps"] > 0:
            assert x["v_est_mps"] + x["delta_v_mps"] <= pre[x["recovery_event_key"]] + 1e-4   # summary rounds to 4 dp


def test_r_c_net_map_change_never_positive(zig_rec):
    (_, _, st, trc), _, _ = zig_rec
    red, rec = _cum(trc["lookahead_trace"])
    assert np.all(rec - red <= 1e-9)
    assert st["lookahead"]["net_map_speed_change_mps"] <= 1e-9
    assert st["lookahead"]["total_speed_reduction_mps"] <= 1.5 + 1e-9          # caps not bypassed


def test_r_d_recovery_stops_when_restored():
    """Unit level (no map pulling the speed back): a 0.3 m/s deficit is handed back in bounded steps, the record
    ends as 'restored', and no further positive update follows."""
    from idr.nav_filter import IVF, NavEKF
    cfg = LookaheadConfig(enable_map_lookahead_speed_prior=True, enable_post_event_recovery=True)
    pr = LookaheadPrior(cfg)
    pr.report, pr.passed = report(), True
    ekf = NavEKF(0.0, 0.0, 0.0, 15.0)
    for _ in range(20):
        ekf.propagate(0.0, 0.0)
    key = ("turn", 1)
    pr.pre_speed[key] = 15.0
    ekf.speed_pseudo_update(-0.3, 2.0)
    pr._book(key, 0.3)
    rec = pr._log_event(key, "turn")
    rec["event_reduction_mps"] = 0.3
    rec["_rec_steps"] = 0
    pr.pending, pr.state = [rec], "recovery"
    deltas = []
    for _ in range(200):
        ekf.propagate(0.0, 0.0)
        pr.step_i += 1
        dv, why = pr._recover(ekf, float(ekf.x[IVF]), 1.0, 0.0) if pr.pending else (0.0, "none")
        deltas.append(dv)
    assert rec["recovery_end_reason"] == "restored"
    assert abs(rec["event_recovered_mps"] - 0.3) < 1e-9 and max(deltas) <= cfg.max_recovery_step_mps + 1e-12
    last = max(i for i, d in enumerate(deltas) if d > 0)
    assert all(d == 0.0 for d in deltas[last + 1:]) and not pr.pending
    assert ekf.x[IVF] <= 15.0 + 1e-9


def test_r_e_pauses_on_imu_deceleration():
    rm, Pt, yaw, v = zigzag()
    T = len(yaw)
    a = np.where((np.arange(T) // 20) % 2 == 0, -0.6, 0.6)                    # 2 s braking / 2 s accelerating
    out = run_outage(rm, Pt[0], 0.0, v, yaw, a, seed=0, return_trace=True, lookahead={**PSEUDO, **REC})
    t = out[3]["lookahead_trace"]
    rec_steps = [x for x in t if x["state"] == "recovery"]
    assert rec_steps
    assert all(x["delta_v_mps"] == 0.0 for x in rec_steps if x["a_recent_mps2"] < -0.3)
    assert any(x["reason_prior_not_applied"] == "recovery_paused_imu_decel" for x in rec_steps)


def test_r_f_low_route_confidence_pauses_recovery():
    (_, _, st, trc), _, _ = run_zig({**REC, "recovery_route_confidence_threshold": 1.01})
    assert all(x["delta_v_mps"] <= 0.0 for x in trc["lookahead_trace"])
    assert any(x["reason_prior_not_applied"] == "recovery_paused_low_route_confidence" for x in trc["lookahead_trace"])
    assert st["lookahead"]["total_recovered_mps"] == 0.0


def test_r_g_recovery_survives_delayed_replay():
    from idr.nav_filter import IE, IN, IVF, NavEKF
    ekf, ref = NavEKF(0.0, 0.0, 0.0, 12.0), NavEKF(0.0, 0.0, 0.0, 12.0)
    for k in range(40):
        for f in (ekf, ref):
            f.propagate(0.0, 0.0)
        if k == 25:
            ekf.speed_pseudo_update(-0.6, 2.0)
        if k == 33:
            ekf.speed_recovery_update(0.4, 2.0)
    v_before = ekf.x[IVF]
    H = np.zeros((2, ekf.x.size)); H[0, IE] = H[1, IN] = 1.0
    assert ekf.delayed_update(ekf.k - 20, ekf.state_at(ekf.k - 20)[0][[IE, IN]], H, np.eye(2) * 25.0)
    assert abs((ekf.x[IVF] - ref.x[IVF]) - (-0.2)) < 0.05 and abs(ekf.x[IVF] - v_before) < 0.05


def test_r_h_recovery_disabled_is_bit_identical_to_l15b():
    rm, Pt, yaw, v = zigzag()
    T = len(yaw)
    a = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=1, lookahead=dict(PSEUDO))
    b = run_outage(rm, Pt[0], 0.0, v, yaw, np.zeros(T), seed=1, lookahead={**PSEUDO, "enable_post_event_recovery": False})
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
