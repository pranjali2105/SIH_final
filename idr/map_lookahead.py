"""Confidence-gated OSM look-ahead slowdown prior (experimental, OFF by default).

Known failure mode: the TFT speed estimate reacts too slowly when the vehicle slows for turns, roundabouts and
sharp curves (+1.8 to +2.2 m/s bias), so along-track distance runs ahead and the matcher reaches junctions early.
This module adds a bounded, NON-POSITIVE forward-acceleration correction to the dead-reckoning input when the
current map hypothesis is confident and a speed-limiting event lies just ahead along the directed road graph.
It never commands, never accelerates, never overwrites speed and never touches map-matching strength.

Causality: uses only the current EKF state, the current particle-filter hypothesis, the user's planned route (if
any, fixed at outage start) and OSM geometry ahead. No future IMU, GNSS, labels or ground-truth route.

Pieces:
  extract_lookahead(rm, edge, s, cfg, route)   directed walk ahead -> features + candidate events
  gate(report, cfg)                            reuse idr.confidence.ConfidenceReport -> (passed, map_conf, reason)
  target_speed(event, v_est, cfg)              conservative geometric upper bound
  decel_prior(v_est, v_target, d, cfg)         a_prior = min(0, (vt^2 - v^2) / (2 max(d, d_min))) + ramp weight
  LookaheadPrior                               state between 1 Hz map steps; per-step correction + logging
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from idr.osm_map import STEP

EVENT_TYPES = ("none", "junction", "turn", "roundabout", "curve")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class LookaheadConfig:
    enable_map_lookahead_speed_prior: bool = False
    lookahead_horizon_m: float = 40.0
    event_min_distance_m: float = 5.0
    event_max_distance_m: float = 60.0
    ramp_m: float = 5.0                      # smooth fade to 0 over this distance at both window limits
    minimum_speed_excess_mps: float = 0.5
    map_confidence_threshold: float = 0.70   # best road-hypothesis probability (ConfidenceReport.best_route_probability)
    route_margin_min: float = 0.20           # best - second-best
    cross_track_max_m: float = 15.0
    heading_residual_max_deg: float = 25.0
    ess_ratio_min: float = 0.10
    a_lat_max_mps2: float = 2.5
    max_decel_prior_mps2: float = 0.30
    prior_gain: float = 0.30
    maximum_speed_correction_mps: float | None = None   # legacy per-event cap (acceleration mode); overrides below
    # prior_mode: "speed_pseudo_measurement" (bounded Kalman update on EKF speed, approach/inside-event only) or
    # "acceleration" (L1/L2: non-positive forward-acceleration input, integrated by the EKF)
    prior_mode: str = "speed_pseudo_measurement"
    speed_prior_sigma_mps: float = 2.0
    max_speed_update_per_step_mps: float = 0.10
    max_speed_reduction_per_event_mps: float = 1.0
    max_total_speed_reduction_per_outage_mps: float = 1.5
    inside_event_m: float = 20.0             # inside_event lasts until the vehicle is this far past the event point
    # bounded post-event recovery (speed_pseudo_measurement mode): hand back ONLY what the same event removed
    enable_post_event_recovery: bool = False
    recovery_duration_s: float = 5.0
    max_recovery_step_mps: float = 0.05
    recovery_imu_decel_pause_mps2: float = -0.3
    recovery_route_confidence_threshold: float = 0.70
    recovery_max_window_factor: float = 3.0  # give up after factor x duration (pauses included); deficit logged
    # causal pre-outage speed-bias gate (idr.speed_bias_gate): no negative prior when the dead-reckoned speed was
    # already running low just before the outage
    enable_speed_bias_gate: bool = False
    minimum_allowed_bias_mps: float = -0.25
    branch_guard_distance_m: float = 30.0       # next branch closer than this -> require bias >= 0
    recent_bias_window_s: float = 30.0
    bias_unknown_policy: str = "suppress"       # "suppress" | "allow" when no shadow estimate is available
    allowed_event_types: tuple = ("roundabout", "turn", "curve")
    enable_for_generic_junctions: bool = False
    # geometry
    turn_min_deg: float = 45.0               # material turn
    straight_tol_deg: float = 30.0           # a successor within this of straight-on = "can go straight"
    turn_arc_m: float = 15.0                 # assumed arc length of a junction turn: kappa = |angle| / turn_arc_m
    curve_window_m: float = 20.0             # curvature = heading change over this arc (filters OSM vertex kinks)
    curve_kappa_min: float = 0.01            # rad/m (R < 100 m) to count as a curve event
    roundabout_cap_mps: float = 9.0          # configured_event_cap per type
    turn_cap_mps: float = 9.0
    curve_cap_mps: float = 30.0
    junction_cap_mps: float = 12.0
    eps_kappa: float = 1e-4
    debug_trace: bool = False


@dataclass
class Event:
    type: str
    distance_m: float
    turn_angle_rad: float = 0.0
    kappa: float = 0.0
    ring_r_m: float | None = None
    key: tuple = ()


@dataclass
class Lookahead:
    valid: bool
    reason: str
    events: list = field(default_factory=list)
    distance_to_next_junction_m: float | None = None
    distance_to_roundabout_entry_m: float | None = None
    distance_to_next_material_turn_m: float | None = None
    next_turn_angle_rad: float | None = None
    max_abs_curvature: float = 0.0
    mean_abs_curvature: float = 0.0
    walked_m: float = 0.0


def _successors(rm, cur):
    return [int(e) for e in rm.out_edges.get(int(rm.e_v[cur]), []) if e != rm.e_rev[cur]]


def _end_bearing(rm, e, back_m=10.0):
    i1 = rm.e_start[e] + rm.e_n[e] - 1
    i0 = max(rm.e_start[e], i1 - int(back_m // STEP))
    return float(np.angle(np.mean(np.exp(1j * rm.bearing[i0:i1 + 1]))))


def _start_bearing(rm, e, fwd_m=10.0):
    i0 = rm.e_start[e]
    i1 = min(rm.e_start[e] + rm.e_n[e] - 1, i0 + int(fwd_m // STEP))
    return float(np.angle(np.mean(np.exp(1j * rm.bearing[i0:i1 + 1]))))


def extract_lookahead(rm, edge, s, cfg: LookaheadConfig, route=frozenset()) -> Lookahead:
    """Walk the DIRECTED graph ahead of (edge, s) for cfg.lookahead_horizon_m. Successor choice: the only successor;
    else the single planned-route successor; else, if every successor turns materially (no straight-on option),
    a forced-turn event at that node and the walk stops; else ambiguous -> stop (no events beyond)."""
    horizon = cfg.lookahead_horizon_m
    out = Lookahead(valid=True, reason="ok")
    brg, dist, sidx = [], [], []
    cur, j0 = int(edge), int(np.clip(s // STEP, 0, rm.e_n[edge] - 1))
    d = -(s - j0 * STEP)                          # distance of sample j0 relative to the vehicle (<= 0)
    turn_thr = np.radians(cfg.turn_min_deg)
    for _ in range(50):
        i0 = rm.e_start[cur]
        for jj in range(j0, rm.e_n[cur]):
            dj = d + (jj - j0) * STEP
            if dj >= 0:
                brg.append(rm.bearing[i0 + jj]); dist.append(dj); sidx.append(int(i0 + jj))
        d_node = d + (rm.e_len[cur] - j0 * STEP)  # distance to this edge's end node
        if d_node > horizon:
            break
        succ = _successors(rm, cur)
        if not succ:
            out.valid, out.reason = False, "disconnected"
            break
        b_in = _end_bearing(rm, cur)
        if rm.rings and rm.e_ring[cur] < 0:
            ring_next = [e for e in succ if rm.e_ring[e] >= 0]
            if ring_next:                            # entering a roundabout at this node
                ring = rm.rings[int(rm.e_ring[ring_next[0]])]
                out.events.append(Event("roundabout", d_node, ring_r_m=float(ring["r"]),
                                        kappa=1.0 / max(float(ring["r"]), 1.0), key=("rb", int(rm.e_v[cur]))))
                out.distance_to_roundabout_entry_m = d_node
                break
        angles = {e: _wrap(_start_bearing(rm, e) - b_in) for e in succ}
        if len(succ) == 1:
            nxt = succ[0]
        else:
            if out.distance_to_next_junction_m is None:
                out.distance_to_next_junction_m = d_node
            in_route = [e for e in succ if e in route]
            straight = [e for e in succ if abs(angles[e]) < np.radians(cfg.straight_tol_deg)]
            if len(in_route) == 1:
                nxt = in_route[0]
            elif not straight:
                ang = min(angles.values(), key=abs)  # forced turn, conservative (smallest) angle
                out.events.append(Event("turn", d_node, turn_angle_rad=float(ang),
                                        kappa=abs(ang) / cfg.turn_arc_m, key=("turn", int(rm.e_v[cur]))))
                break
            else:
                out.events.append(Event("junction", d_node, key=("jct", int(rm.e_v[cur]))))
                out.reason = "ambiguous_successor"
                break
        ang = angles[nxt]
        if abs(ang) >= turn_thr:
            out.events.append(Event("turn", d_node, turn_angle_rad=float(ang), kappa=abs(ang) / cfg.turn_arc_m,
                                    key=("turn", int(rm.e_v[cur]))))
        cur, j0, d = nxt, 0, d_node
    # curvature along the walked path (heading change over curve_window_m)
    if len(brg) >= 2:
        b = np.unwrap(np.array(brg))
        dd = np.array(dist)
        w = max(1, int(round(cfg.curve_window_m / STEP)))
        if len(b) > w:
            k = np.abs(b[w:] - b[:-w]) / (dd[w:] - dd[:-w] + 1e-6)
            kd = dd[:-w]
            inside = kd <= horizon
            node_d = [e.distance_m for e in out.events if e.type in ("turn", "roundabout")]
            clean = inside.copy()
            for nd in node_d:                     # the junction turn already represents that heading change
                clean &= ~((kd <= nd) & (kd + cfg.curve_window_m >= nd))
            if inside.any():
                out.max_abs_curvature = float(k[inside].max())
                out.mean_abs_curvature = float(k[inside].mean())
                hit = np.flatnonzero(clean & (k > cfg.curve_kappa_min))
                if len(hit):
                    i = int(hit[0])
                    seg = inside & (kd >= kd[i]) & (kd <= kd[i] + cfg.curve_window_m)
                    out.events.append(Event("curve", float(kd[i] + cfg.curve_window_m / 2), kappa=float(k[seg].max()),
                                            turn_angle_rad=float(b[min(i + w, len(b) - 1)] - b[i]),
                                            key=("curve", sidx[i] // 4)))
        out.walked_m = float(dd[-1])
    turns = [e for e in out.events if e.type in ("turn", "curve") and abs(e.turn_angle_rad) >= turn_thr]
    if turns:
        t0 = min(turns, key=lambda e: e.distance_m)
        out.distance_to_next_material_turn_m, out.next_turn_angle_rad = t0.distance_m, t0.turn_angle_rad
    out.events = [e for e in out.events if e.distance_m <= horizon]
    return out


def gate(report, cfg: LookaheadConfig):
    """Reuse the existing ConfidenceReport (idr.confidence). Returns (passed, map_confidence, reason)."""
    if report is None:
        return False, 0.0, "no_confidence_report"
    checks = [
        (report.best_route_probability >= cfg.map_confidence_threshold, "low_route_confidence"),
        (report.route_margin >= cfg.route_margin_min, "low_route_margin"),
        (report.cross_track_residual_m <= cfg.cross_track_max_m, "cross_track_residual"),
        (abs(np.degrees(report.heading_residual_rad)) <= cfg.heading_residual_max_deg, "heading_residual"),
        (report.ess_ratio >= cfg.ess_ratio_min, "particle_ess"),
    ]
    for ok, why in checks:
        if not ok:
            return False, 0.0, why
    return True, float(report.best_route_probability), "ok"


def target_speed(ev: Event, v_est, cfg: LookaheadConfig):
    caps = {"roundabout": cfg.roundabout_cap_mps, "turn": cfg.turn_cap_mps, "curve": cfg.curve_cap_mps,
            "junction": cfg.junction_cap_mps}
    if ev.type == "junction":                     # generic junction: no curvature evidence -> cap only
        v_geo = np.inf
    else:
        v_geo = float(np.sqrt(cfg.a_lat_max_mps2 / max(abs(ev.kappa), cfg.eps_kappa)))
    return float(min(v_est, v_geo, caps.get(ev.type, np.inf)))


def _smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def decel_prior(v_est, v_target, d, cfg: LookaheadConfig):
    """Non-positive kinematic deceleration to reach v_target at distance d, and the window ramp weight in [0, 1]."""
    a = min(0.0, (v_target ** 2 - v_est ** 2) / (2.0 * max(d, cfg.event_min_distance_m)))
    lo, hi, r = cfg.event_min_distance_m, cfg.event_max_distance_m, max(cfg.ramp_m, 1e-6)
    ramp = _smooth((d - lo) / r) * _smooth((hi - d) / r) if lo <= d <= hi else 0.0
    return float(a), float(ramp)


PRIOR_MODES = ("speed_pseudo_measurement", "acceleration")
STATES = ("inactive", "approach", "inside_event", "after_event", "recovery")


class LookaheadPrior:
    """Holds the latest look-ahead (refreshed at 1 Hz with the map step) and produces the per-step correction.

    Caps (both modes): per event (max_speed_reduction_per_event_mps, or legacy maximum_speed_correction_mps) and per
    outage (max_total_speed_reduction_per_outage_mps); once either is reached no further negative correction."""

    def __init__(self, cfg: LookaheadConfig, bias_info: dict | None = None):
        if cfg.prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}")
        self.cfg = cfg
        self.bias_info = bias_info or {"available": False}
        self.next_branch_m = None
        self.gate_counts = Counter()
        self.outage_total = 0.0
        self.capped_events = set()
        self.outage_capped = False
        self.active: Event | None = None
        self.state = "inactive"
        self.state_steps = Counter()
        self.step_i = 0
        self.pre_speed = {}                    # event key -> EKF speed when the event became active (pre-event)
        self.event_log = {}                    # event key -> per-event reduction / recovery record
        self.rec = None                        # event record recovered in the current step (logging)
        self.pending = []                      # events with an unrecovered deficit, oldest first
        self.outage_recovered = 0.0
        self.cur_edge, self.prev_edge, self.route_changed = None, None, False
        self.events: list[Event] = []
        self.passed, self.map_conf, self.reason = False, 0.0, "not_yet_evaluated"
        self.report = None
        self.used = {}                         # event key -> cumulative speed reduction (m/s)
        self.trace = []
        self.n_applied, self.sum_abs, self.max_abs = 0, 0.0, 0.0
        self.events_seen = Counter()
        self._seen_keys = set()
        self.rejects = Counter()

    def refresh(self, mm):
        """Called right after MapMatcher.measure(): current dominant road hypothesis -> look-ahead."""
        cfg = self.cfg
        self.report = mm.conf
        self.passed, self.map_conf, self.reason = gate(self.report, cfg)
        self.events = []
        keys = np.array([mm._road_key(e) for e in mm.edge])
        uniq, inv = np.unique(keys, return_inverse=True)
        b = int(np.argmax(np.bincount(inv, weights=mm.w)))
        sel = inv == b
        e_major = int(np.bincount(mm.edge[sel], weights=mm.w[sel]).argmax())
        self.prev_edge, self.cur_edge = self.cur_edge, e_major
        if (self.pending and self.prev_edge is not None and e_major != self.prev_edge
                and int(mm.rm.e_u[e_major]) != int(mm.rm.e_v[self.prev_edge])):
            self.route_changed = True          # the hypothesis jumped (not a successor of the previous road)
        if not self.passed:
            return
        on = sel & (mm.edge == e_major)
        s_cur = float(np.average(mm.s[on], weights=mm.w[on]))
        la = extract_lookahead(mm.rm, e_major, s_cur, cfg, frozenset(mm.route))
        allowed = set(cfg.allowed_event_types) | ({"junction"} if cfg.enable_for_generic_junctions else set())
        self.events = [e for e in la.events if e.type in allowed]
        self.next_branch_m = la.distance_to_next_junction_m
        if self.active is not None:                    # keep tracking the active event with the fresher distance
            same = [e for e in self.events if e.key == self.active.key]
            if same:
                self.active = same[0]
        for e in la.events:
            if e.key not in self._seen_keys:
                self._seen_keys.add(e.key)
                self.events_seen[e.type] += 1
        if not self.events:
            self.reason = la.reason if la.reason != "ok" else "no_event"

    def advance(self, ddist):
        for e in self.events:
            e.distance_m -= ddist
        if self.active is not None and all(self.active is not e for e in self.events):
            self.active.distance_m -= ddist
        if self.next_branch_m is not None:
            self.next_branch_m -= ddist

    # ---- caps
    def _event_cap(self):
        c = self.cfg
        return c.maximum_speed_correction_mps if c.maximum_speed_correction_mps is not None \
            else c.max_speed_reduction_per_event_mps

    def remaining(self, key):
        return max(0.0, min(self._event_cap() - self.used.get(key, 0.0),
                            self.cfg.max_total_speed_reduction_per_outage_mps - self.outage_total))

    def _book(self, key, reduction):
        self.used[key] = self.used.get(key, 0.0) + reduction
        self.outage_total += reduction
        if self.used[key] >= self._event_cap() - 1e-9:
            self.capped_events.add(key)
        if self.outage_total >= self.cfg.max_total_speed_reduction_per_outage_mps - 1e-9:
            self.outage_capped = True

    # ---- event state machine (speed_pseudo_measurement mode)
    def bias_gate(self):
        """(passed, branch_guard_active, reason). Only ever blocks a NEGATIVE prior; uses the pre-outage estimate."""
        cfg = self.cfg
        if not cfg.enable_speed_bias_gate:
            return True, False, None
        b = self.bias_info.get("recent_ekf_speed_bias_mps") if self.bias_info.get("available") else None
        if b is None:
            return cfg.bias_unknown_policy == "allow", False, "bias_unknown"
        branch = self.next_branch_m is not None and 0.0 <= self.next_branch_m <= cfg.branch_guard_distance_m
        if b < cfg.minimum_allowed_bias_mps:
            return False, branch, "bias_gate"
        if branch and b < 0.0:
            return False, True, "branch_guard"
        return True, branch, None

    def _log_event(self, key, typ):
        if key not in self.event_log:
            self.event_log[key] = {"event_key": str(key), "type": typ, "event_reduction_mps": 0.0,
                                   "event_recovered_mps": 0.0, "event_remaining_deficit_mps": 0.0,
                                   "pre_event_speed_mps": self.pre_speed.get(key), "recovery_start_time": None,
                                   "recovery_end_time": None, "recovery_end_reason": None,
                                   "recent_ekf_speed_bias_mps": self.bias_info.get("recent_ekf_speed_bias_mps"),
                                   "recent_gnss_speed_mean_mps": self.bias_info.get("recent_gnss_speed_mean_mps"),
                                   "recent_ekf_speed_mean_mps": self.bias_info.get("recent_ekf_speed_mean_mps"),
                                   "bias_gate_passed": None, "branch_guard_active": False, "prior_applied": False,
                                   "_suppress": Counter()}
        return self.event_log[key]

    def _end_recovery(self, rec, reason):
        rec["recovery_end_time"] = round(self.step_i * 0.1, 1)
        rec["recovery_end_reason"] = reason
        self.pending = [r for r in self.pending if r is not rec]
        if not self.pending:
            self.route_changed = False
            if self.state == "recovery":
                self.state = "inactive"

    def _update_state(self, v_est):
        cfg = self.cfg
        if self.active is not None and self.active.distance_m < -cfg.inside_event_m:
            key = self.active.key
            self.active = None
            self.state = "after_event"
            if cfg.enable_post_event_recovery:
                if self.used.get(key, 0.0) > 0.0:
                    rec = self.event_log[key]
                    rec["recovery_start_time"] = round(self.step_i * 0.1, 1)
                    rec["_rec_steps"] = 0
                    self.pending.append(rec)
                self.state = "recovery" if self.pending else "inactive"
        if self.active is None or self.active.distance_m > 0:          # never switch away while inside an event
            cands = [e for e in self.events if 0 < e.distance_m <= cfg.event_max_distance_m] if self.passed else []
            if cands:
                prev = self.active
                self.active = min(cands, key=lambda e: (target_speed(e, v_est, cfg), e.distance_m))
                if self.active.key not in self.pre_speed:
                    self.pre_speed[self.active.key] = v_est
                if cfg.enable_speed_bias_gate:
                    self._log_event(self.active.key, self.active.type)
                if prev is None:                                        # a new event PAUSES pending recovery
                    for rec in self.pending:
                        rec["preemptions"] = rec.get("preemptions", 0) + 1
        if self.active is not None:
            self.state = "inside_event" if self.active.distance_m <= 0 else "approach"
        elif self.state == "recovery" or (self.pending and cfg.enable_post_event_recovery):
            self.state = "recovery" if self.pending else "inactive"
        elif self.state in ("approach", "inside_event"):
            self.state = "after_event" if not cfg.enable_post_event_recovery else "inactive"
        self.state_steps[self.state] += 1

    def _recover(self, ekf, v_est, lam, a_recent):
        """One recovery step: bounded positive pseudo-measurement for the OLDEST pending event whose own pre-event
        speed is still above the current speed. Each event recovers at most its own reduction."""
        from idr.nav_filter import IVF
        cfg = self.cfg
        self.rec = None
        if self.route_changed:
            for rec in list(self.pending):
                self._end_recovery(rec, "route_hypothesis_changed")
            return 0.0, "recovery_cancelled_route_change"
        for rec in list(self.pending):
            rec["_rec_steps"] += 1
            if rec["event_reduction_mps"] - rec["event_recovered_mps"] <= 1e-9:
                self._end_recovery(rec, "restored")
            elif rec["_rec_steps"] > cfg.recovery_max_window_factor * cfg.recovery_duration_s / 0.1:
                self._end_recovery(rec, "timeout")
        if not self.pending:
            return 0.0, "recovery_complete"
        r = self.report
        if r is None or r.best_route_probability < cfg.recovery_route_confidence_threshold:
            return 0.0, "recovery_paused_low_route_confidence"
        if a_recent is not None and a_recent < cfg.recovery_imu_decel_pause_mps2:
            return 0.0, "recovery_paused_imu_decel"
        rec = next((q for q in self.pending if q["pre_event_speed_mps"] is not None
                    and v_est < q["pre_event_speed_mps"]), None)
        if rec is None:
            return 0.0, "recovery_paused_at_pre_event_speed"
        v_pre = rec["pre_event_speed_mps"]
        rem = rec["event_reduction_mps"] - rec["event_recovered_mps"]
        # schedule: the event's reduction handed back over recovery_duration_s (a gain-weighted pull would be
        # asymptotic and never finish); applied through the same replay-safe Kalman speed update
        rate = rec["event_reduction_mps"] * 0.1 / cfg.recovery_duration_s
        net_room = self.outage_total - self.outage_recovered      # net map change must stay <= 0
        dv = min(cfg.max_recovery_step_mps, rate, rem, v_pre - v_est, net_room)
        if dv <= 0.0:
            return 0.0, "recovery_zero_update"
        ekf.speed_recovery_update(dv / lam, cfg.speed_prior_sigma_mps / lam)
        rec["event_recovered_mps"] += dv
        self.outage_recovered += dv
        self.rec = rec
        assert self.outage_recovered <= self.outage_total + 1e-9
        if rec["event_reduction_mps"] - rec["event_recovered_mps"] <= 1e-9:
            self._end_recovery(rec, "restored")
        return dv, "recovered"

    def speed_update(self, ekf, v_raw=None, a_recent=None):
        """speed_pseudo_measurement mode, called right after ekf.propagate(): bounded non-positive speed update in
        approach / inside_event; optional bounded positive recovery of the same event's reduction afterwards."""
        from idr.nav_filter import IDS, IVF
        cfg = self.cfg
        self.step_i += 1
        lam = float(np.clip(1.0 + ekf.x[IDS], 0.6, 1.4))
        v_est = lam * max(float(ekf.x[IVF]), 0.0)
        self._update_state(v_est)
        dv, vt, why = 0.0, None, self.reason
        self.rec = None
        if self.state == "recovery":
            dv, why = self._recover(ekf, v_est, lam, a_recent)
        rec_key = self.rec["event_key"] if self.rec is not None else None
        if self.state == "recovery" or why in ("recovered", "recovery_complete", "recovery_cancelled_route_change"):
            pass
        elif self.state not in ("approach", "inside_event"):
            why = "after_event" if self.state == "after_event" else (self.reason if not self.passed else "no_event")
        elif not self.passed:
            why = self.reason
        else:
            vt = target_speed(self.active, v_est, cfg)
            rem = self.remaining(self.active.key)
            gp, branch, gwhy = self.bias_gate()
            if cfg.enable_speed_bias_gate:
                ev = self._log_event(self.active.key, self.active.type)
                ev["bias_gate_passed"] = gp if ev["bias_gate_passed"] is None else (ev["bias_gate_passed"] and gp)
                ev["branch_guard_active"] = ev["branch_guard_active"] or branch
            if v_est <= vt + cfg.minimum_speed_excess_mps:
                why = "speed_below_target"
            elif not gp:
                why = gwhy
                self.gate_counts[gwhy] += 1
                ev["_suppress"][gwhy] += 1
            elif rem <= 0.0:
                why = "outage_cap_reached" if self.outage_capped else "event_cap_reached"
            else:
                pvv = float(ekf.P[IVF, IVF]) * lam ** 2                 # variance in position-speed units
                k = pvv / (pvv + cfg.speed_prior_sigma_mps ** 2)
                dv = float(np.clip(k * (vt - v_est), -min(cfg.max_speed_update_per_step_mps, rem), 0.0))
                if dv < 0.0:
                    ekf.speed_pseudo_update(dv / lam, cfg.speed_prior_sigma_mps / lam)
                    self._book(self.active.key, -dv)
                    ev_ = self._log_event(self.active.key, self.active.type)
                    ev_["event_reduction_mps"] += -dv
                    ev_["prior_applied"] = True
                    self.n_applied += 1
                    self.sum_abs += -dv / 0.1                          # same units as acceleration mode (m/s^2)
                    self.max_abs = max(self.max_abs, -dv / 0.1)
                    why = "applied"
                else:
                    why = "zero_update"
        assert dv <= 0.0 or why == "recovered"
        if why not in ("applied", "recovered"):
            self.rejects[why] += 1
        if cfg.debug_trace:
            r = self.report
            self.trace.append({
                "map_lookahead_enabled": True, "prior_mode": cfg.prior_mode, "state": self.state,
                "map_prior_gate_passed": self.passed,
                "route_confidence": getattr(r, "best_route_probability", None),
                "route_margin": getattr(r, "route_margin", None),
                "event_type": self.active.type if self.active else "none",
                "event_key": self.active.key if self.active else None,
                "event_distance_m": self.active.distance_m if self.active else None, "v_raw_mps": v_raw,
                "v_est_mps": v_est, "v_target_mps": vt, "delta_v_mps": dv,
                "event_reduction_mps": self.used.get(self.active.key, 0.0) if self.active else 0.0,
                "outage_reduction_mps": self.outage_total, "outage_recovered_mps": self.outage_recovered,
                "recovery_event_key": rec_key, "a_recent_mps2": a_recent, "a_correction_mps2": 0.0,
                "reason_prior_not_applied": None if dv != 0 else why})

    def correction(self, v_est, v_raw=None):
        cfg = self.cfg
        a_corr, best, info = 0.0, None, {"reason": self.reason}
        gp, _, gwhy = self.bias_gate()
        if self.passed and self.events and not gp:
            info = {"reason": gwhy}
            self.gate_counts[gwhy] += 1
        elif self.passed and self.events:
            cands = []
            for e in self.events:
                vt = target_speed(e, v_est, cfg)
                if v_est <= vt + cfg.minimum_speed_excess_mps:
                    cands.append((0.0, e, vt, 0.0, "speed_below_target")); continue
                a_p, ramp = decel_prior(v_est, vt, e.distance_m, cfg)
                if ramp <= 0:
                    cands.append((0.0, e, vt, a_p, "outside_distance_window")); continue
                left = self.remaining(e.key)
                if left <= 0.0:
                    cands.append((0.0, e, vt, a_p, "outage_cap_reached" if self.outage_capped
                                  else "max_speed_correction_reached")); continue
                a_c = cfg.prior_gain * self.map_conf * float(np.clip(a_p, -cfg.max_decel_prior_mps2, 0.0)) * ramp
                a_c = max(a_c, -left / 0.1)          # never exceed the event / outage caps within one 0.1 s step
                cands.append((a_c, e, vt, a_p, "applied" if a_c < 0 else "zero_prior"))
            a_corr, best, vt, a_p, why = min(cands, key=lambda c: c[0])
            info = {"reason": why, "v_target": vt, "a_prior": a_p}
            if a_corr < 0:
                self._book(best.key, -a_corr * 0.1)
                self.n_applied += 1
                self.sum_abs += -a_corr
                self.max_abs = max(self.max_abs, -a_corr)
        if info["reason"] != "applied":
            self.rejects[info["reason"]] += 1
        assert a_corr <= 0.0
        if cfg.debug_trace:
            r = self.report
            self.trace.append({
                "map_lookahead_enabled": True, "map_prior_gate_passed": self.passed,
                "route_confidence": getattr(r, "best_route_probability", None),
                "route_margin": getattr(r, "route_margin", None),
                "event_type": best.type if best else "none", "event_distance_m": best.distance_m if best else None,
                "event_turn_angle_rad": best.turn_angle_rad if best else None,
                "event_max_curvature": best.kappa if best else None, "v_raw_mps": v_raw, "v_est_mps": v_est,
                "v_target_mps": info.get("v_target"), "a_prior_mps2": info.get("a_prior"),
                "a_correction_mps2": a_corr, "reason_prior_not_applied": None if a_corr < 0 else info["reason"]})
        return a_corr

    def log_corrected(self, v_corr):
        if self.cfg.debug_trace and self.trace:
            self.trace[-1]["v_corrected_mps"] = v_corr

    def summary(self):
        return {"prior_mode": self.cfg.prior_mode, "n_applications": self.n_applied,
                "total_abs_correction_mps2_steps": self.sum_abs,
                "total_speed_reduction_mps": self.outage_total,
                "events_reaching_event_cap": len(self.capped_events), "outage_cap_reached": self.outage_capped,
                "state_steps": dict(self.state_steps),
                "total_recovered_mps": self.outage_recovered,
                "bias_gate": {"enabled": self.cfg.enable_speed_bias_gate, **{k: v for k, v in self.bias_info.items()},
                              "minimum_allowed_bias_mps": self.cfg.minimum_allowed_bias_mps,
                              "branch_guard_distance_m": self.cfg.branch_guard_distance_m,
                              "suppressed_steps_by_reason": dict(self.gate_counts)},
                "net_map_speed_change_mps": self.outage_recovered - self.outage_total,
                "events": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in {
                    **e, "event_remaining_deficit_mps": e["event_reduction_mps"] - e["event_recovered_mps"],
                    "prior_suppressed_reason": (e["_suppress"].most_common(1)[0][0] if e.get("_suppress") else None)
                    }.items() if not k.startswith("_")} for e in self.event_log.values()],
                "mean_abs_correction_mps2": self.sum_abs / self.n_applied if self.n_applied else 0.0,
                "max_abs_correction_mps2": self.max_abs, "events_detected": dict(self.events_seen),
                "not_applied_by_reason": dict(self.rejects), "config": asdict(self.cfg)}
