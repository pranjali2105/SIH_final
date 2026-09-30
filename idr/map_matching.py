"""Map-constrained particle filter with delayed decision, feeding soft/strong corrections into the NavEKF.

Particle = (edge, s along edge, lateral offset d, speed-scale error). Particles ride the directed road graph with the
EKF's distance increments and branch at junctions (planned-route bonus, U-turn penalty). Every second they are
weighted with the residual r = [d_cross, wrap(psi - psi_road), d_along, kappa - kappa_road]:
    log w = -1/2 r^T Sigma^-1 r + log P(transition)
The road decision is taken for t - LAG from particle ancestry (fixed-lag), and fed back to the EKF as a delayed
cross-track (+ heading when consistent on a straight road, + along-track at bends) pseudo-measurement.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from idr.nav_filter import IBG, IE, IN, IPSI, IVF, MAP_AFFECT, NS, NavEKF, wrap
from idr.confidence import BETA, ConfGates, evaluate as conf_evaluate, junction_confidence
from idr.osm_map import STEP, RoadMap, ring_arc

LAG_S = 3          # decision delay, seconds (= samples at 1 Hz)
TURN_SIG = np.radians(55)       # flat: keep several branches alive through junctions/roundabouts


@dataclass
class MMConfig:
    n_particles: int = 300
    route_bonus: float = 3.0
    uturn_penalty: float = 0.02
    sig_psi_road: float = np.radians(4.0)
    sig_kappa: float = 0.01
    along_rate: float = 0.05          # along-track uncertainty growth, fraction of distance since last fix
    strong_conf: float = 0.8
    soft_conf: float = 0.5
    strong_consistency: int = 3
    straight_kappa: float = 0.003
    junction_clear_m: float = 25.0
    use_heading_update: bool = True
    use_along_update: bool = True
    heading_gate: float = np.radians(20)   # map may nudge heading, never swing it
    reseed_frac: float = 0.05              # recovery particles per second (escape wrong-road locks)
    scale_sig0: float = 0.10           # prior spread of per-particle speed-scale error (TFT speed error ~10%)
    # roundabout mode
    roundabout_mode: bool = True
    exit_route_bonus: float = 3.0      # prior on the planned route's exit
    full_circle_prior: float = 0.1
    ring_rms_max: float = 3.0          # rings that are not circular (gyratories) get topology only, no circle residuals
    exit_tol_m: float = 6.0            # a particle may leave at its exit once within this arc distance of it
    sig_ring_radial: float = 3.0
    heading_arc: bool = True           # ring progress + likelihood from gyro heading (not distance / EKF position)
    ring_alpha_entry_m: float = 10.0   # distance-driven progress for the first metres on the ring (entry transition)
    # Defaults = validated Step-0 behaviour (9.5% Y1+M / 9.1% held-out Vw4, 3 seeds): full-state map updates.
    # Config B (heading_update_affect=MAP_AFFECT) looked better on Y1+M but was worse on held-out Vw4 (10.9%).
    pos_update_affect: tuple | None = None
    heading_update_affect: tuple | None = None
    landmark_scoring: bool = False     # gyro turn events vs particle junction turns; no measurable effect (Step 4)
    turn_min_deg: float = 45.0
    turn_time_sig_s: float = 2.0
    turn_miss_penalty: float = 8.0
    open_loop_odometry: bool = False   # True: particles advance by uncorrected IMU distance (no EKF<->particle loop); measured slightly worse on Y1+M
    # Experiment A: gated junction/turn along-track update. A completed gyro turn fixes when the car was AT the
    # junction; particles that turned there (same sign, similar angle, similar time) get their arc position pulled
    # towards "junction + distance driven since the gyro turn", s += K_s * r_s (gradual), a bounded speed-scale nudge,
    # and the EKF gets one along-track pseudo-measurement towards the corrected cluster. Off by default.
    along_landmark: bool = False
    along_gain: float = 0.3            # K_s
    along_scale_gain: float = 0.3      # per-event scale nudge = gain * r_s / max(distance before the turn, 200 m)
    along_scale_step: float = 0.03     # |per-event scale change| <= this
    along_scale_max: float = 0.3       # |particle scale| <= this after the nudge
    along_time_win_s: float = 8.0      # particle junction turn within +-this of the gyro turn centre
    along_eval_delay_s: float = 6.0    # evaluate the event this long after the gyro turn midpoint
    # "particle" (v1): landmark = matched particles' own junction turns (measured neutral: 81% of events skipped, the
    # dominant particles rarely carry a turn record). "map" (v2): walk back from the dominant cluster along the road
    # topology, following the gyro heading history, and locate where the ROAD bearing changes by the gyro turn.
    along_landmark_mode: str = "map"
    along_win_frac: float = 0.4        # v2 search window: back-distance in [(1-f) D - 20, (1+f) D + 30] m
    along_pred_tol_deg: float = 45.0   # v2 backward walk: predecessor end bearing vs gyro heading at that time
    # Confidence layer (idr/confidence.py). confidence_log: compute + log reports, no behaviour change.
    # confidence_gate_along (A3-gate): along-track updates (junction landmark + _decide along) only on
    # hard_gated_update and C_junction. confidence_policy (A3-full): + tempered likelihood, soft / no EKF map updates,
    # expanded support at very low confidence.
    confidence_log: bool = False
    confidence_gate_along: bool = False
    confidence_policy: bool = False
    cjunction_min: float = 0.70
    cfactor_min: float = 0.30
    soft_sig_mult: float = 2.0
    soft_clip_cross_m: float = 10.0
    soft_clip_along_m: float = 30.0
    expand_mult: int = 3
    along_ang_tol_deg: float = 35.0    # |particle turn angle - gyro heading change|
    along_min_share: float = 0.6       # matched particles must carry this weight share (else ambiguous -> skip)
    along_max_resid_m: float = 60.0    # |weighted median r_s| above this = implausible -> skip
    along_psi_gate_deg: float = 15.0   # EKF heading vs matched road bearing
    along_sig0_m: float = 5.0          # EKF along-track sigma = sig0 + frac * |r_s|
    along_sig_frac: float = 0.1


class MapMatcher:
    def __init__(self, rm: RoadMap, ekf: NavEKF, cfg: MMConfig | None = None, route_edges=(), rng=None):
        self.rm, self.ekf = rm, ekf
        self.cfg = cfg or MMConfig()
        self.rng = rng or np.random.default_rng(0)
        self.route = set(int(e) for e in route_edges)
        for e in list(self.route):
            if rm.e_rev[e] >= 0:
                self.route.add(int(rm.e_rev[e]))
        self.steps = 0
        self.dist_since_fix = 0.0
        self.kappa_buf = deque(maxlen=10)
        self.hist = deque(maxlen=LAG_S + 1)     # per-second snapshots of (k, edge, s) per particle
        self.consistent = 0
        self.last_edge = -1
        self.stats = {"soft": 0, "strong": 0, "along": 0, "rejected": 0, "none": 0, "ring_radial": 0}
        self.trace = []
        self.psi_hist = deque(maxlen=250)                  # (step, EKF heading) at 10 Hz for turn-event detection
        self.last_event_k = -10 ** 9
        self.last_along_event_k = -10 ** 9
        self.dist_hist = deque(maxlen=250)                 # (step, EKF distance) at 10 Hz for the along-track landmark
        self.along_events = []
        self.stats.update({"alm_events": 0, "alm_applied": 0, "alm_skip_share": 0, "alm_skip_resid": 0,
                           "alm_skip_psi": 0, "alm_skip_cross": 0, "alm_ekf": 0})
        self._ess_prior = self._ess_post = float(self.cfg.n_particles)
        self.conf = None
        self.conf_gates = ConfGates()
        self.tft_resid_now = None
        self._R_map, self._R_best = {}, None
        self._conf_on = self.cfg.confidence_log or self.cfg.confidence_gate_along or self.cfg.confidence_policy
        if self.cfg.along_landmark:                        # keep enough EKF history to replay from the turn midpoint
            ekf.hist_len = max(ekf.hist_len, int(10 * (self.cfg.along_eval_delay_s + 6)))
            ekf.hist = deque(ekf.hist, maxlen=ekf.hist_len)
        self._init_particles()

    # ---- initialisation ----
    def _init_particles(self):
        rm, x, n = self.rm, self.ekf.x, self.cfg.n_particles
        r = max(30.0, 3 * self.ekf.pos_sigma())
        idx = rm.tree.query_ball_point(x[[IE, IN]], r)
        if not idx:
            _, j = rm.nearest(x[[IE, IN]], k=8)
            idx = list(np.atleast_1d(j))
        idx = np.array(idx)
        d = np.hypot(*(rm.pts[idx] - x[[IE, IN]]).T)
        dh = wrap(rm.bearing[idx] - x[IPSI])
        logw = -0.5 * (d / 10.0) ** 2 - 0.5 * (dh / np.radians(15)) ** 2
        p = np.exp(logw - logw.max())
        p /= p.sum()
        pick = self.rng.choice(len(idx), size=n, p=p)
        self.edge = rm.edge[idx[pick]].astype(np.int64)
        self.s = rm.s[idx[pick]].astype(float)
        self.d = self.rng.normal(0, 1.0, n)
        self.scale = self.rng.normal(0, self.cfg.scale_sig0, n)
        self.w = np.full(n, 1.0 / n)
        self.anc = np.arange(n)
        # roundabout hypothesis per particle: ring, exit edge, arc target from entry, arc progress on the ring
        self.ring_of = np.full(n, -1)
        self.exit_edge = np.full(n, -1)
        self.exit_target = np.full(n, np.nan)
        self.ring_prog = np.zeros(n)
        self.hscale = self.rng.normal(0, 0.03, n)          # per-particle gyro-scale error for heading-driven arcs
        self.turn_k = np.full(n, -10 ** 9)                 # step of the particle's last sharp junction turn
        self.turn_ang = np.zeros(n)                        # its signed angle (bearing change, right = +)
        self.turn_checked = np.ones(n, dtype=bool)
        self.odo = np.zeros(n)                             # particle's own travelled distance
        self.odo_turn = np.zeros(n)                        # ... at its last sharp junction turn
        self.turn_used = np.ones(n, dtype=bool)            # that turn already consumed by an along-track landmark
        self.turn_xy = np.zeros((n, 2))                    # junction node of that turn
        for i in np.flatnonzero(self.rm.e_ring[self.edge] >= 0):
            ring = self.rm.rings[self.rm.e_ring[self.edge[i]]]
            self._assign_exit(i, ring, ring_arc(ring, self.rm.edge_point(int(self.edge[i]), self.s[i])[0]))

    # ---- motion ----
    def _assign_exit(self, i, ring, entry_arc):
        """Draw an exit hypothesis for particle i entering `ring` at arc position entry_arc."""
        circ = ring["circumference"]
        cands, prior = [], []
        for x in ring["exits"]:
            tgt = (x["arc"] - entry_arc) % circ
            if tgt < 3.0:                               # the exit at the entry point = full circle / U-turn
                tgt += circ
                p = self.cfg.full_circle_prior
            else:
                p = 1.0
            if x["edge"] in self.route:
                p *= self.cfg.exit_route_bonus
            cands.append((x["edge"], tgt))
            prior.append(p)
        if not cands:
            return
        prior = np.array(prior) / np.sum(prior)
        e, tgt = cands[self.rng.choice(len(cands), p=prior)]
        self.ring_of[i], self.exit_edge[i], self.exit_target[i], self.ring_prog[i] = ring["id"], e, tgt, 0.0

    def _branch(self, i, node):
        rm = self.rm
        outs = rm.out_edges.get(int(node), [])
        if not outs:
            return None
        cur = self.edge[i]
        if self.cfg.roundabout_mode and rm.rings:
            rid = rm.e_ring[cur]
            if rid >= 0:
                # on the ring: stay until the particle's own exit is reached (no premature exits)
                ring_next = [e for e in outs if rm.e_ring[e] == rid]
                if self.exit_edge[i] in outs and self.ring_prog[i] >= self.exit_target[i] - self.cfg.exit_tol_m:
                    self.ring_of[i] = -1
                    return int(self.exit_edge[i])
                if self.ring_prog[i] > self.exit_target[i] + 15.0:
                    # passed its hypothesised exit: re-draw among the exits still ahead
                    ring = rm.rings[rid]
                    self._assign_exit(i, ring, ring_arc(ring, rm.pts[rm.e_start[cur] + rm.e_n[cur] - 1]))
                if ring_next:
                    return int(ring_next[0])
            else:
                nxt = self._generic_branch(i, outs, cur)
                if nxt is not None and rm.e_ring[nxt] >= 0:
                    ring = rm.rings[rm.e_ring[nxt]]
                    self._assign_exit(i, ring, ring_arc(ring, rm.pts[rm.e_start[nxt]]))
                return nxt
        return self._generic_branch(i, outs, cur)

    def _generic_branch(self, i, outs, cur):
        nxt = self._generic_branch_pick(i, outs, cur)
        if nxt is not None and (self.cfg.landmark_scoring or self.cfg.along_landmark) and self.rm.e_ring[nxt] < 0:
            rm = self.rm
            b_in = rm.bearing[rm.e_start[cur] + rm.e_n[cur] - 1]
            b_out = rm.bearing[min(rm.e_start[nxt] + 2, rm.e_start[nxt] + rm.e_n[nxt] - 1)]
            ang = wrap(b_out - b_in)
            if abs(ang) > np.radians(self.cfg.turn_min_deg):
                self.turn_k[i], self.turn_ang[i], self.turn_checked[i] = self.steps, ang, False
                self.odo_turn[i] = self.odo[i] - (self.s[i] - rm.e_len[cur])     # odometer AT the node
                self.turn_xy[i] = rm.pts[rm.e_start[nxt]]
                self.turn_used[i] = False
        return nxt

    def _generic_branch_pick(self, i, outs, cur):
        rm = self.rm
        rev = rm.e_rev[cur]
        b_now = self.ekf.x[IPSI]
        probs = []
        for e in outs:
            b0 = rm.bearing[rm.e_start[e]]
            p = np.exp(-0.5 * (wrap(b0 - b_now) / TURN_SIG) ** 2) + 1e-3
            if e == rev:
                p *= self.cfg.uturn_penalty
            if e in self.route:
                p *= self.cfg.route_bonus
            probs.append(p)
        probs = np.array(probs) / np.sum(probs)
        return outs[self.rng.choice(len(outs), p=probs)]

    def advance(self, ddist: float, dpsi: float = 0.0):
        """Move particles by the EKF distance increment (called every 0.1 s after ekf.propagate).

        On circular rings (roundabout mode) progress comes from the gyro heading change, arc = r * |dpsi|, which does
        not inherit the speed error that dominates along-track drift."""
        rm = self.rm
        step = ddist * (1 + self.scale)
        on_ring = rm.e_ring[self.edge] >= 0
        if self.cfg.roundabout_mode and self.cfg.heading_arc and on_ring.any() and ddist > 0:
            # s_ring += a * r * dpsi_signed + (1 - a) * ds_DR ; a = 0 during the entry transition (first
            # ring_alpha_entry_m on the ring), 1 during circulation. dpsi_signed = dir * dpsi clipped at 0, so the
            # opposite-direction swerve when entering does not count as progress.
            idx = np.flatnonzero(on_ring)
            for j in idx:
                ring = rm.rings[rm.e_ring[self.edge[j]]]
                if ring["rms"] > self.cfg.ring_rms_max or self.ring_prog[j] < self.cfg.ring_alpha_entry_m:
                    continue
                step[j] = ring["r"] * max(ring["dir"] * dpsi, 0.0) * (1 + self.hscale[j])
        self.s += step
        self.odo += step
        self.ring_prog[on_ring] += step[on_ring]
        self.psi_hist.append((self.steps, self.ekf.x[IPSI]))
        self.dist_hist.append((self.steps, self.ekf.dist, self.ekf.k))
        self.dist_since_fix += ddist
        over = np.flatnonzero(self.s > rm.e_len[self.edge])
        for i in over:
            guard = 0
            while self.s[i] > rm.e_len[self.edge[i]] and guard < 5:
                nxt = self._branch(i, rm.e_v[self.edge[i]])
                if nxt is None:
                    self.s[i] = rm.e_len[self.edge[i]]
                    break
                self.s[i] -= rm.e_len[self.edge[i]]
                self.edge[i] = nxt
                guard += 1
        self.steps += 1

    # ---- weighting at 1 Hz ----
    def _road_geom(self):
        rm = self.rm
        i0 = rm.e_start[self.edge]
        j = np.clip((self.s // STEP).astype(np.int64), 0, rm.e_n[self.edge] - 1)
        a = i0 + j
        return rm.pts[a], rm.bearing[a], rm.curv[a], a

    def measure(self, kappa_obs: float):
        cfg, rm, ekf = self.cfg, self.rm, self.ekf
        self.kappa_buf.append(kappa_obs)
        if self._conf_on:
            self.conf = conf_evaluate(self, self.conf_gates)
            self.stats[f"conf_{self.conf.action}"] = self.stats.get(f"conf_{self.conf.action}", 0) + 1
        if cfg.confidence_policy and self.conf.action == "expand_support":
            self._reseed(mult=cfg.expand_mult, radius=max(50.0, 3 * ekf.pos_sigma()))
        else:
            self._reseed()
        pts, brg, curv, _ = self._road_geom()
        t = np.column_stack([np.sin(brg), np.cos(brg)])
        nrm = np.column_stack([np.cos(brg), -np.sin(brg)])
        delta = ekf.pos - pts
        cross = (delta * nrm).sum(1) - self.d
        along = (delta * t).sum(1)
        sp = ekf.pos_sigma()
        half_w = rm.e_width[self.edge]
        s_cross2 = sp ** 2 + half_w ** 2
        s_along2 = sp ** 2 + (cfg.along_rate * self.dist_since_fix) ** 2 + 25.0
        dpsi = wrap(ekf.x[IPSI] - brg)
        s_psi2 = ekf.P[IPSI, IPSI] + cfg.sig_psi_road ** 2
        k_obs = float(np.mean(self.kappa_buf))
        dk = k_obs - curv
        comp = {"d": cross ** 2 / s_cross2, "s": along ** 2 / s_along2, "psi": dpsi ** 2 / s_psi2,
                "kappa": dk ** 2 / cfg.sig_kappa ** 2}
        logl = -0.5 * (comp["d"] + comp["s"] + comp["psi"] + comp["kappa"])
        if cfg.confidence_policy:
            logl = BETA[self.conf.action] * logl              # tempered map likelihood (beta = 1 when confident)
        logl[np.abs(dpsi) > np.radians(90)] -= 50.0          # impossible direction (never tempered)
        if cfg.roundabout_mode and rm.rings:
            logl = self._ring_loglik(logl, pts, k_obs, sp)
        lm_ll = None
        if cfg.landmark_scoring:
            lm_ll = self._landmark_loglik()
            logl = logl + lm_ll
        self._ess_prior = float(1.0 / np.sum(self.w ** 2))
        logw = np.log(self.w + 1e-300) + logl
        logw -= logw.max()
        self.w = np.exp(logw)
        self.w /= self.w.sum()
        self._ess_post = float(1.0 / np.sum(self.w ** 2))
        if self._conf_on:
            self._R_map = {k: float((self.w * v).sum()) for k, v in comp.items()}
            self._R_map["T"] = float((self.w * -lm_ll).sum()) if lm_ll is not None else 0.0
            keys = np.array([self._road_key(e) for e in self.edge])
            uniq, inv = np.unique(keys, return_inverse=True)
            b = int(np.argmax(np.bincount(inv, weights=self.w)))
            m = inv == b
            self._R_best = float((self.w[m] * (comp["d"][m] + comp["psi"][m])).sum() / self.w[m].sum())
        if cfg.along_landmark:
            self._along_landmark()
        self.hist.append((ekf.k, self.edge.copy(), self.s.copy(), self.d.copy()))
        if 1.0 / np.sum(self.w ** 2) < cfg.n_particles / 2:
            self._resample()
        return self._decide()

    def _ring_loglik(self, logl, pts, k_obs, sp):
        """On circular rings replace point-to-polyline terms with circle / tangent / arc / curvature residuals."""
        rm, ekf, cfg = self.rm, self.ekf, self.cfg
        rid = rm.e_ring[self.edge]
        for r_id in np.unique(rid[rid >= 0]):
            ring = rm.rings[r_id]
            if ring["rms"] > cfg.ring_rms_max:
                continue
            m = rid == r_id
            c = np.array([ring["cx"], ring["cy"]])
            p = ekf.pos
            r_circle = np.hypot(*(p - c)) - ring["r"]
            r_curv = k_obs - ring["dir"] / ring["r"]
            if cfg.heading_arc:
                # On a circle the heading IS the arc position: compare the EKF heading with the tangent at each
                # particle's own point. The radial term is kept loose because the EKF position carries the
                # along-track (speed) error.
                q = pts[m]
                psi_tan_p = np.arctan2(q[:, 0] - c[0], q[:, 1] - c[1]) + ring["dir"] * np.pi / 2
                r_tan = wrap(ekf.x[IPSI] - psi_tan_p)
                s_rad2 = sp ** 2 + 25.0 + (cfg.along_rate * self.dist_since_fix) ** 2
                ll = -0.5 * (r_circle ** 2 / s_rad2 + r_tan ** 2 / (ekf.P[IPSI, IPSI] + np.radians(8) ** 2)
                             + r_curv ** 2 / 0.02 ** 2)
            else:
                psi_tan = np.arctan2(p[0] - c[0], p[1] - c[1]) + ring["dir"] * np.pi / 2
                r_tan = wrap(ekf.x[IPSI] - psi_tan)
                circ = ring["circumference"]
                a_ekf = ring_arc(ring, p)
                a_par = np.array([ring_arc(ring, q) for q in pts[m]])
                r_arc = (a_par - a_ekf + circ / 2) % circ - circ / 2
                ll = -0.5 * (r_circle ** 2 / (sp ** 2 + 4.0) + r_tan ** 2 / (ekf.P[IPSI, IPSI] + np.radians(8) ** 2)
                             + r_arc ** 2 / (sp ** 2 + (0.1 * self.ring_prog[m]) ** 2 + 25.0) + r_curv ** 2 / 0.02 ** 2)
            logl[m] = ll
        return logl

    def _heading_change(self, k0, k1):
        """EKF heading change (right = +) between steps k0 and k1 from the 10 Hz history (None if unavailable)."""
        hs = [(k, p) for k, p in self.psi_hist if k0 <= k <= k1]
        if len(hs) < 2:
            return None
        ps = np.unwrap([p for _, p in hs])
        return float(ps[-1] - ps[0])

    def _landmark_loglik(self):
        """Turn-event consistency. A completed gyro turn (> turn_min within ~5 s) must be matched by a same-sign
        junction turn of the particle at about the same time; a particle's sharp junction turn must be matched by a
        gyro turn. Each event / particle turn is scored once."""
        cfg, n, now = self.cfg, len(self.w), self.steps
        out = np.zeros(n)
        thr = np.radians(cfg.turn_min_deg)
        # 1) gyro event completed in the last few seconds?
        best, best_c = 0.0, None
        for c in range(now - 80, now - 25, 5):
            dpsi = self._heading_change(c - 25, c + 25)
            if dpsi is not None and abs(dpsi) > abs(best):
                best, best_c = dpsi, c
        if best_c is not None and abs(best) > thr and best_c - self.last_event_k > 60:
            self.last_event_k = best_c
            sig_k = cfg.turn_time_sig_s * 10
            recent = self.turn_k > best_c - 80
            same = recent & (np.sign(self.turn_ang) == np.sign(best))
            dt = (self.turn_k - best_c) / sig_k
            out = np.where(same, -0.5 * dt ** 2, -cfg.turn_miss_penalty)
            self.turn_checked[recent] = True
        # 2) particle turned sharply but the gyro saw no turn around that time
        due = (~self.turn_checked) & (self.turn_k <= now - 30)
        for i in np.flatnonzero(due):
            dpsi = self._heading_change(self.turn_k[i] - 30, self.turn_k[i] + 30)
            if dpsi is not None and abs(dpsi) < 0.5 * thr:
                out[i] -= cfg.turn_miss_penalty
            self.turn_checked[i] = True
        return out

    def _gyro_turn_event(self):
        """Largest completed gyro heading change (5 s window) in the last ~12 s: (dpsi, midpoint step) or None.
        The midpoint is where half of the heading change has happened (the junction crossing)."""
        now, best, best_c = self.steps, 0.0, None
        for c in range(now - 120, now - 25, 5):
            dpsi = self._heading_change(c - 25, c + 25)
            if dpsi is not None and abs(dpsi) > abs(best):
                best, best_c = dpsi, c
        if best_c is None or abs(best) <= np.radians(self.cfg.turn_min_deg):
            return None
        hs = [(k, p) for k, p in self.psi_hist if best_c - 25 <= k <= best_c + 25]
        ks = np.array([k for k, _ in hs])
        ps = np.unwrap([p for _, p in hs])
        frac = (ps - ps[0]) / (ps[-1] - ps[0])
        return float(ps[-1] - ps[0]), int(ks[np.argmax(frac >= 0.5)])

    def _dist_at(self, k):
        for kk, dd, ek in self.dist_hist:
            if kk >= k:
                return dd, ek
        return None

    def _along_landmark(self):
        """Experiment A: gated junction/turn along-track correction (see MMConfig.along_landmark)."""
        cfg, rm, ekf = self.cfg, self.rm, self.ekf
        ev = self._gyro_turn_event()
        if ev is None:
            return
        dpsi, c = ev
        if c - self.last_along_event_k <= 60:              # same physical turn, already handled
            return
        if self.steps - c < cfg.along_eval_delay_s * 10:   # wait until late particles have branched too
            return
        self.last_along_event_k = c
        self.stats["alm_events"] += 1
        if cfg.along_landmark_mode == "map":
            return self._along_landmark_map(dpsi, c)
        at = self._dist_at(c)
        if at is None:
            return
        d_c, ek_c = at
        D_car = ekf.dist - d_c                             # EKF distance driven since the car was at the junction
        win = cfg.along_time_win_s * 10
        match = (~self.turn_used) & (np.abs(self.turn_k - c) <= win) & (np.sign(self.turn_ang) == np.sign(dpsi)) \
            & (np.abs(self.turn_ang - dpsi) < np.radians(cfg.along_ang_tol_deg)) & (np.abs(self.turn_ang) < np.radians(150)) \
            & (rm.e_ring[self.edge] < 0)
        share = float(self.w[match].sum())
        rec = {"k": int(c), "k_now": int(self.steps), "dpsi_deg": float(np.degrees(dpsi)), "share": share,
               "n_match": int(match.sum()), "applied": False}
        self.along_events.append(rec)
        # every matched or not, the turn candidates in the window are consumed (a landmark is used once)
        self.turn_used[np.abs(self.turn_k - c) <= win] = True
        if share < cfg.along_min_share:
            self.stats["alm_skip_share"] += 1
            rec["skip"] = "share"
            return
        idx = np.flatnonzero(match)
        r = (1 + self.scale[idx]) * D_car - (self.odo[idx] - self.odo_turn[idx])
        wm = self.w[idx] / self.w[idx].sum()
        o = np.argsort(r)
        r_med = float(r[o][np.searchsorted(np.cumsum(wm[o]), 0.5)])
        trav = self.odo[idx] - self.odo_turn[idx]
        rec.update({"r_med": r_med, "D_car": float(D_car), "part_travel_med": float(np.median(trav)),
                    "turn_dk_med": float(np.median(self.turn_k[idx] - c)), "ekf_pos": ekf.pos.tolist()})
        if abs(r_med) > cfg.along_max_resid_m:
            self.stats["alm_skip_resid"] += 1
            rec["skip"] = "resid"
            return
        # geometry gates on the matched cluster (before correcting it)
        pts, brg, _, _ = self._road_geom()
        mu0 = (wm[:, None] * pts[idx]).sum(0)
        b0 = float(np.angle(np.sum(wm * np.exp(1j * brg[idx]))))
        if abs(wrap(ekf.x[IPSI] - b0)) > np.radians(cfg.along_psi_gate_deg):
            self.stats["alm_skip_psi"] += 1
            rec["skip"] = "psi"
            return
        nrm0 = np.array([np.cos(b0), -np.sin(b0)])
        half_w = float(np.average(rm.e_width[self.edge[idx]], weights=wm))
        if abs((ekf.pos - mu0) @ nrm0) > max(15.0, 3 * half_w) + 2 * ekf.pos_sigma():
            self.stats["alm_skip_cross"] += 1
            rec["skip"] = "cross"
            return
        # gradual particle correction, clipped to the current edge (never branches here)
        ds = cfg.along_gain * r
        self.s[idx] = np.clip(self.s[idx] + ds, 0.0, rm.e_len[self.edge[idx]] - 1e-3)
        self.odo[idx] += ds
        # bounded slow speed-scale nudge (distance before the turn carries the accumulated scale error)
        base = np.maximum(self.odo_turn[idx], 200.0)
        dsc = np.clip(cfg.along_scale_gain * r / base, -cfg.along_scale_step, cfg.along_scale_step)
        self.scale[idx] = np.clip(self.scale[idx] + dsc, -cfg.along_scale_max, cfg.along_scale_max)
        # EKF: at the gyro-turn midpoint the car was at the junction node -> delayed position fix at that step,
        # replayed to now (corrects the pre-turn distance error and, through the replay, the current position)
        node = (wm[:, None] * self.turn_xy[idx]).sum(0)
        spread = float(np.sqrt((wm * ((self.turn_xy[idx] - node) ** 2).sum(1)).sum()))
        sig = cfg.along_sig0_m + cfg.along_sig_frac * abs(r_med)
        pre = ekf.pos.copy()
        rec.update({"applied": True, "node": node.tolist(), "node_spread_m": spread, "sig_m": sig})
        self.stats["alm_applied"] += 1
        if spread < 10.0:
            H = np.zeros((2, NS))
            H[0, IE] = H[1, IN] = 1.0
            x_c = ekf.state_at(ek_c)[0]
            ok = ekf.delayed_update(ek_c, node, H, sig ** 2 * np.eye(2), gate=None, affect=cfg.pos_update_affect)
            if ok:
                self.stats["alm_ekf"] += 1
                self.dist_since_fix = 0.0
                rec.update({"ekf_node_resid_m": float(np.hypot(*(node - x_c[[IE, IN]]))),
                            "ekf_shift_m": (ekf.pos - pre).tolist()})

    def _psi_at(self, k):
        best = None
        for kk, p in self.psi_hist:
            if kk >= k:
                return p
            best = p
        return best

    def _in_edges(self, node):
        rm = self.rm
        if not hasattr(rm, "_in_edges"):
            d = {}
            for e in range(len(rm.e_v)):
                d.setdefault(int(rm.e_v[e]), []).append(e)
            rm._in_edges = d
        return rm._in_edges.get(int(node), [])

    def _walk_back(self, e, s, max_back, k_now, dist_now, scale):
        """Road samples behind (e, s), back-distance ascending: (b, point, bearing, node_flag). At nodes the predecessor
        is the in-edge whose end bearing best matches the gyro heading at the time the car was there."""
        rm, cfg = self.rm, self.cfg
        out_b, out_p, out_h, out_n = [], [], [], []
        b, cur, j = 0.0, int(e), int(np.clip(s // STEP, 0, rm.e_n[e] - 1))
        s_here = s
        for _ in range(40):
            i0 = rm.e_start[cur]
            for jj in range(j, -1, -1):
                bb = b + max(s_here - jj * STEP, 0.0)
                out_b.append(bb); out_p.append(rm.pts[i0 + jj]); out_h.append(rm.bearing[i0 + jj]); out_n.append(jj == 0)
            b += s_here
            if b > max_back:
                break
            cands = [x for x in self._in_edges(rm.e_u[cur]) if x != rm.e_rev[cur] and rm.e_ring[x] < 0]
            if not cands:
                break
            at = self._step_at_dist(dist_now - b / (1 + scale))
            psi_b = self._psi_at(at) if at is not None else None
            if psi_b is None:
                break
            dev = [abs(wrap(rm.bearing[rm.e_start[x] + rm.e_n[x] - 1] - psi_b)) for x in cands]
            if min(dev) > np.radians(cfg.along_pred_tol_deg):
                break
            cur = int(cands[int(np.argmin(dev))])
            j, s_here = int(rm.e_n[cur] - 1), float(rm.e_len[cur])
        return np.array(out_b), np.array(out_p), np.array(out_h), np.array(out_n)

    def _step_at_dist(self, dd):
        for kk, d_, _ in self.dist_hist:
            if d_ >= dd:
                return kk
        return None

    def _along_landmark_map(self, dpsi, c):
        """Experiment A v2: map-located turn landmark (see MMConfig.along_landmark_mode)."""
        cfg, rm, ekf = self.cfg, self.rm, self.ekf
        rec = {"k": int(c), "k_now": int(self.steps), "dpsi_deg": float(np.degrees(dpsi)), "applied": False, "mode": "map"}
        self.along_events.append(rec)
        at = self._dist_at(c)
        if at is None:
            rec["skip"] = "hist"
            return
        d_c, ek_c = at
        D_car = ekf.dist - d_c
        # stable match: dominant current road
        keys = np.array([self._road_key(e) for e in self.edge])
        uniq, inv = np.unique(keys, return_inverse=True)
        share_r = np.bincount(inv, weights=self.w)
        best = int(np.argmax(share_r))
        rec["share"] = float(share_r[best])
        if share_r[best] < cfg.along_min_share or rm.e_ring[self.edge[inv == best][0]] >= 0:
            self.stats["alm_skip_share"] += 1
            rec["skip"] = "share"
            return
        sel = inv == best
        e_major = int(np.bincount(self.edge[sel], weights=self.w[sel]).argmax())
        on = sel & (self.edge == e_major)
        wm = self.w[on] / self.w[on].sum()
        o = np.argsort(self.s[on])
        s_med = float(self.s[on][o][np.searchsorted(np.cumsum(wm[o]), 0.5)])
        scale = float(np.average(self.scale[on], weights=wm))
        D = (1 + scale) * D_car
        p_now, b_now, _ = rm.edge_point(e_major, s_med)
        # gates on the current match: heading and corridor
        if abs(wrap(ekf.x[IPSI] - b_now)) > np.radians(cfg.along_psi_gate_deg):
            self.stats["alm_skip_psi"] += 1
            rec["skip"] = "psi"
            return
        nrm = np.array([np.cos(b_now), -np.sin(b_now)])
        if abs((ekf.pos - p_now) @ nrm) > max(15.0, 3 * float(rm.e_width[e_major])) + 2 * ekf.pos_sigma():
            self.stats["alm_skip_cross"] += 1
            rec["skip"] = "cross"
            return
        f = cfg.along_win_frac
        lo, hi = max(0.0, (1 - f) * D - 20.0), (1 + f) * D + 30.0
        bs, ps, hs, ns = self._walk_back(e_major, s_med, hi + 40.0, self.steps, ekf.dist, scale)
        if len(bs) < 3 or bs[-1] < lo + 10:
            self.stats.setdefault("alm_skip_path", 0)
            self.stats["alm_skip_path"] += 1
            rec["skip"] = "path"
            return
        # fraction of the gyro turn the road has completed at each back-distance: 1 after the turn, 0 before it
        psi_in = self._psi_at(c - 25)
        frac = wrap(hs - psi_in) / dpsi
        after = frac > 0.5
        # crossings (walking back: after -> before) inside the window
        cross = [i for i in range(1, len(bs)) if after[i - 1] and not after[i] and lo <= bs[i] <= hi]
        if len(cross) != 1:
            self.stats.setdefault("alm_skip_ambig", 0)
            self.stats["alm_skip_ambig"] += 1
            rec["skip"] = "ambiguous" if cross else "no_turn"
            rec["n_cand"] = len(cross)
            return
        i = cross[0]
        # valid turn: road bearing change over +-30 m around the landmark matches the gyro (sign and size)
        pre = (bs >= bs[i] + 5) & (bs <= bs[i] + 35)
        post = (bs <= bs[i] - 5) & (bs >= bs[i] - 35)
        if not pre.any() or not post.any():
            rec["skip"] = "short"
            self.stats.setdefault("alm_skip_ambig", 0)
            self.stats["alm_skip_ambig"] += 1
            return
        road_turn = float(wrap(np.angle(np.mean(np.exp(1j * hs[post]))) - np.angle(np.mean(np.exp(1j * hs[pre])))))
        if np.sign(road_turn) != np.sign(dpsi) or abs(road_turn - dpsi) > np.radians(cfg.along_ang_tol_deg):
            self.stats.setdefault("alm_skip_angle", 0)
            self.stats["alm_skip_angle"] += 1
            rec.update({"skip": "angle", "road_turn_deg": float(np.degrees(road_turn))})
            return
        b_star = 0.5 * (bs[i - 1] + bs[i])
        lm = 0.5 * (ps[i - 1] + ps[i])
        r = float(D - b_star)                               # > 0: cluster is behind where the landmark says it is
        rec.update({"b_star": float(b_star), "D": float(D), "r_med": r, "road_turn_deg": float(np.degrees(road_turn)),
                    "at_node": bool(ns[max(i - 2, 0):i + 2].any()), "landmark": lm.tolist()})
        if abs(r) > cfg.along_max_resid_m:
            self.stats["alm_skip_resid"] += 1
            rec["skip"] = "resid"
            return
        if self._conf_on:
            srt = np.sort(share_r)[::-1]
            k_map = self._step_at_dist(ekf.dist - b_star / (1 + scale))
            dt_s = (k_map - c) * 0.1 if k_map is not None else 99.0
            cj = junction_confidence(float((ekf.pos - p_now) @ nrm), float(wrap(ekf.x[IPSI] - b_now)),
                                     float(wrap(dpsi - road_turn)), dt_s, float(srt[0]),
                                     float(srt[1]) if len(srt) > 1 else 0.0)
            rec.update(cj)
            rec["action"] = self.conf.action
            rec["cr"] = self.conf.compact()
            if (cfg.confidence_gate_along or cfg.confidence_policy) and (
                    self.conf.action != "hard_gated_update" or cj["C_junction"] < cfg.cjunction_min
                    or min(cj["C_edge"], cj["C_turn"], cj["C_timing"], cj["C_cloud"]) < cfg.cfactor_min):
                self.stats["alm_skip_conf"] = self.stats.get("alm_skip_conf", 0) + 1
                rec["skip"] = "confidence"
                return
        # gradual particle correction for the dominant road's particles (clipped to their edge)
        idx = np.flatnonzero(sel)
        self.s[idx] = np.clip(self.s[idx] + cfg.along_gain * r, 0.0, rm.e_len[self.edge[idx]] - 1e-3)
        self.odo[idx] += cfg.along_gain * r
        dsc = float(np.clip(cfg.along_scale_gain * r / max(d_c, 200.0), -cfg.along_scale_step, cfg.along_scale_step))
        self.scale[idx] = np.clip(self.scale[idx] + dsc, -cfg.along_scale_max, cfg.along_scale_max)
        # EKF: delayed 2-D fix at the gyro-turn midpoint on the landmark, replayed to now
        sig = cfg.along_sig0_m + cfg.along_sig_frac * abs(r)
        H = np.zeros((2, NS))
        H[0, IE] = H[1, IN] = 1.0
        x_c = ekf.state_at(ek_c)[0]
        pre_pos = ekf.pos.copy()
        ok = ekf.delayed_update(ek_c, lm, H, sig ** 2 * np.eye(2), gate=None, affect=cfg.pos_update_affect)
        self.stats["alm_applied"] += 1
        rec.update({"applied": True, "sig_m": sig, "scale_nudge": dsc})
        if ok:
            self.stats["alm_ekf"] += 1
            self.dist_since_fix = 0.0
            rec.update({"ekf_lm_resid_m": float(np.hypot(*(lm - x_c[[IE, IN]]))),
                        "ekf_shift_m": (ekf.pos - pre_pos).tolist()})

    def exit_probabilities(self):
        """Weight share of each hypothesised exit among particles holding a roundabout hypothesis."""
        m = self.exit_edge >= 0
        if not m.any() or self.w[m].sum() <= 0:
            return {}
        out = {}
        for e in np.unique(self.exit_edge[m]):
            out[int(e)] = float(self.w[m & (self.exit_edge == e)].sum() / self.w[m].sum())
        return out

    def _reseed(self, mult=1, radius=None):
        """Replace the lowest-weight particles with fresh candidates on roads around the EKF pose."""
        n = int(mult * self.cfg.reseed_frac * self.cfg.n_particles)
        if n == 0:
            return
        rm, x = self.rm, self.ekf.x
        idx = rm.tree.query_ball_point(x[[IE, IN]], radius or max(30.0, 2 * self.ekf.pos_sigma()))
        if not idx:
            return
        idx = np.array(idx)
        dh = wrap(rm.bearing[idx] - x[IPSI])
        good = idx[np.abs(dh) < np.radians(30)]
        if len(good) == 0:
            return
        pick = good[self.rng.integers(len(good), size=n)]
        worst = np.argsort(self.w)[:n]
        self.edge[worst] = rm.edge[pick]
        self.s[worst] = rm.s[pick]
        self.d[worst] = self.rng.normal(0, 1.0, n)
        self.scale[worst] = self.rng.normal(0, self.cfg.scale_sig0, n)
        self.ring_of[worst], self.exit_edge[worst], self.exit_target[worst], self.ring_prog[worst] = -1, -1, np.nan, 0
        self.turn_k[worst], self.turn_checked[worst] = -10 ** 9, True
        self.turn_used[worst], self.odo_turn[worst] = True, self.odo[worst]
        self.turn_xy[worst] = 0.0
        for i in worst[self.rm.e_ring[self.edge[worst]] >= 0]:
            ring = self.rm.rings[self.rm.e_ring[self.edge[i]]]
            self._assign_exit(i, ring, ring_arc(ring, self.rm.edge_point(int(self.edge[i]), self.s[i])[0]))
        self.w[worst] = np.median(self.w)
        self.w /= self.w.sum()
        # their lagged history is unknown: copy the current state so fixed-lag decisions stay well defined
        self.hist = deque([(k, np.where(np.isin(np.arange(len(e)), worst), self.edge, e),
                            np.where(np.isin(np.arange(len(sv)), worst), self.s, sv),
                            np.where(np.isin(np.arange(len(dv)), worst), self.d, dv)) for k, e, sv, dv in self.hist],
                          maxlen=LAG_S + 1)

    def _resample(self):
        n = self.cfg.n_particles
        pos = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), n - 1)
        self.edge, self.s, self.d, self.scale = self.edge[idx], self.s[idx], self.d[idx], self.scale[idx]
        self.ring_of, self.exit_edge, self.hscale = self.ring_of[idx], self.exit_edge[idx], self.hscale[idx]
        self.turn_k, self.turn_ang, self.turn_checked = self.turn_k[idx], self.turn_ang[idx], self.turn_checked[idx]
        self.odo, self.odo_turn, self.turn_used = self.odo[idx], self.odo_turn[idx], self.turn_used[idx]
        self.turn_xy = self.turn_xy[idx]
        self.exit_target, self.ring_prog = self.exit_target[idx], self.ring_prog[idx]
        self.hist = deque([(k, e[idx], s[idx], d[idx]) for k, e, s, d in self.hist], maxlen=LAG_S + 1)
        self.d = self.d + self.rng.normal(0, 0.3, n)
        self.scale = self.scale + self.rng.normal(0, 0.005, n)
        self.w = np.full(n, 1.0 / n)

    # ---- delayed decision + EKF feedback ----
    def _road_key(self, e):
        r = self.rm.e_rev[e]
        return min(e, r) if r >= 0 else e

    def _decide(self):
        cfg, rm, ekf = self.cfg, self.rm, self.ekf
        if len(self.hist) <= LAG_S:
            return None
        k_lag, e_lag, s_lag, d_lag = self.hist[0]
        keys = np.array([self._road_key(e) for e in e_lag])
        uniq, inv = np.unique(keys, return_inverse=True)
        share = np.bincount(inv, weights=self.w)
        best = int(np.argmax(share))
        conf = float(share[best])
        sel = inv == best
        e_major = np.bincount(e_lag[sel], weights=self.w[sel]).argmax()
        s_med = float(np.average(s_lag[sel & (e_lag == e_major)], weights=self.w[sel & (e_lag == e_major)]))
        p_road, b_road, k_road = rm.edge_point(int(e_major), s_med)
        self.consistent = self.consistent + 1 if (uniq[best] == self.last_edge and conf >= cfg.strong_conf) else \
            (1 if conf >= cfg.strong_conf else 0)
        self.last_edge = uniq[best]
        rid = int(rm.e_ring[e_major]) if rm.rings else -1
        sh = share[share > 0]
        self.trace.append({"k": int(k_lag), "k_now": int(ekf.k), "ring": rid, "conf": conf, "edge": int(e_major),
                           "s": s_med, "n_roads": int(len(sh)), "entropy": float(-(sh * np.log(sh)).sum()),
                           "ess": float(1.0 / np.sum(self.w ** 2)), "ess_prior": self._ess_prior,
                           "ess_post": self._ess_post,
                           "speed_scale": float(1.0 + ekf.x[7]), "exit_probs": self.exit_probabilities(),
                           "nhc_sigma": None})
        action = self.conf.action if self.conf is not None else "hard_gated_update"
        if self.conf is not None:
            self.trace[-1]["cr"] = self.conf.compact()
        if cfg.confidence_policy and action in ("preserve_hypotheses", "expand_support"):
            # low confidence: the map only reweights particles (rejects impossible roads); no EKF snap
            self.stats["conf_skip"] = self.stats.get("conf_skip", 0) + 1
            return {"conf": conf, "edge": int(e_major), "p_road": p_road, "k": k_lag, "applied": "conf_skip"}
        if conf < cfg.soft_conf:
            self.stats["none"] += 1
            return {"conf": conf, "edge": int(e_major), "p_road": p_road, "k": k_lag, "applied": "none"}
        if cfg.roundabout_mode and rid >= 0 and rm.rings[rid]["rms"] <= cfg.ring_rms_max:
            # circulating: soft radial pseudo-measurement ||p - c|| = r only; no heading snap, no exit commitment
            ring = rm.rings[rid]
            c = np.array([ring["cx"], ring["cy"]])
            x_lag = ekf.state_at(k_lag)[0]
            u = x_lag[[IE, IN]] - c
            nu = np.hypot(*u)
            if nu > 1e-3:
                u = u / nu
                H = np.zeros((1, NS))
                H[0, IE], H[0, IN] = u
                ok = ekf.delayed_update(k_lag, np.array([ring["r"] + u @ c]), H,
                                        np.array([[cfg.sig_ring_radial ** 2]]), gate=None, affect=MAP_AFFECT)
                self.stats["ring_radial"] += int(ok)
            self.trace[-1]["dist_center"] = float(nu)
            return {"conf": conf, "edge": int(e_major), "p_road": p_road, "k": k_lag, "applied": "ring_radial"}

        # straightness / junction clearance around the matched point
        ks = [rm.edge_point(int(e_major), s_med + o)[2] for o in (-30, -15, 0, 15, 30)]
        straight = max(abs(k) for k in ks) < cfg.straight_kappa
        near_junction = s_med < cfg.junction_clear_m or rm.e_len[e_major] - s_med < cfg.junction_clear_m
        strong = conf >= cfg.strong_conf and self.consistent >= cfg.strong_consistency and not near_junction
        nrm = np.array([np.cos(b_road), -np.sin(b_road)])
        tng = np.array([np.sin(b_road), np.cos(b_road)])
        half_w = float(rm.e_width[e_major])
        # Particle cloud on the dominant road at t-lag -> 2-D position pseudo-measurement. Cross-track is always
        # informative; along-track only once the weights have actually concentrated the cloud (e.g. after a turn),
        # otherwise the cloud just mirrors the EKF's own distance and must not be credited as evidence.
        pos = self._positions(e_lag[sel], s_lag[sel], d_lag[sel])
        wn = self.w[sel] / self.w[sel].sum()
        mu = (wn[:, None] * pos).sum(0)
        dev = pos - mu
        along_w = np.sqrt((wn * (dev @ tng) ** 2).sum())
        along_u = np.std(dev @ tng)
        along_resolved = cfg.use_along_update and along_w < 0.5 * along_u and along_w < 30.0
        if (cfg.confidence_gate_along or cfg.confidence_policy) and action != "hard_gated_update":
            along_resolved = False                           # along-track corrections only when confident
        soft_conf = cfg.confidence_policy and action == "soft_map_update"
        if soft_conf:
            # medium confidence: inflated R, clipped innovation, no heading snap
            strong = False
            x_lag = ekf.state_at(k_lag)[0][[IE, IN]]
            y = mu - x_lag
            mu = x_lag + np.clip(y @ nrm, -cfg.soft_clip_cross_m, cfg.soft_clip_cross_m) * nrm \
                + np.clip(y @ tng, -cfg.soft_clip_along_m, cfg.soft_clip_along_m) * tng
        sig_c = (2.0 + half_w / 2) if strong else 3.0 * (2.0 + half_w)
        if soft_conf:
            sig_c *= cfg.soft_sig_mult
        sig_a = max(5.0, along_w) if along_resolved else 1e4
        R = sig_c ** 2 * np.outer(nrm, nrm) + sig_a ** 2 * np.outer(tng, tng)
        H = np.zeros((2, NS))
        H[0, IE] = H[1, IN] = 1.0
        # gate only on the cross-track component: along-track is allowed to correct large speed errors
        ok = ekf.delayed_update(k_lag, mu, H, R, gate=None, affect=cfg.pos_update_affect)
        applied = "strong" if strong else "soft"
        if ok:
            self.stats[applied] += 1
            if along_resolved:
                self.stats["along"] += 1
                self.dist_since_fix = 0.0
        else:
            self.stats["rejected"] += 1
            applied = "rejected"
        if ok and strong and straight and cfg.use_heading_update and \
                abs(wrap(b_road - ekf.state_at(k_lag)[0][IPSI])) < cfg.heading_gate:
            # vehicle heading equals the road bearing on a straight road (psi observable -> gyro bias observable)
            Hh = np.zeros((1, NS))
            Hh[0, IPSI] = 1.0
            ekf.delayed_update(k_lag, np.array([b_road]), Hh, np.array([[np.radians(2.0) ** 2]]),
                               angle_rows=(0,), gate=9.0, affect=cfg.heading_update_affect)
        return {"conf": conf, "edge": int(e_major), "p_road": p_road, "k": k_lag, "applied": applied}

    def _positions(self, e, s, d):
        rm = self.rm
        j = np.clip((s // STEP).astype(np.int64), 0, rm.e_n[e] - 1)
        a = rm.e_start[e] + j
        b = rm.bearing[a]
        return rm.pts[a] + d[:, None] * np.column_stack([np.cos(b), -np.sin(b)])

    def snapped_pose(self):
        """Display pose: current EKF position projected onto the dominant current road when confident."""
        pts, brg, _, _ = self._road_geom()
        keys = np.array([self._road_key(e) for e in self.edge])
        uniq, inv = np.unique(keys, return_inverse=True)
        share = np.bincount(inv, weights=self.w)
        b = int(np.argmax(share))
        if share[b] < self.cfg.strong_conf:
            return self.ekf.pos.copy()
        sel = inv == b
        return np.average(pts[sel], axis=0, weights=self.w[sel])


NHC_STATE_SCALE = {0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0, 4: 3.0}     # entry / exit x3; circulating x2 when |a_y| > 2


def run_outage(rm: RoadMap, p0, psi0, v0, w, a, a_lat=None, vib=None, p_stop=None, route_edges=(),
               nav_cfg=None, mm_cfg=None, use_map=True, seed=0, roundabout=False, map_free_circle=False,
               return_trace=False, nhc_state=None, ring_mode=None, detect=None, landmark=False, mm_overrides=None,
               tft_resid=None):
    """Dead-reckon one outage. w, a: (T,) corrected yaw rate (CCW+) and forward accel at 10 Hz. Returns (T,2) track.

    roundabout: detector-driven NHC inflation + roundabout mode in the particle filter.
    map_free_circle: when the detector sees circulation but no mapped ring is near, fit a circle to the recent
    dead-reckoned track and apply it as a soft radial constraint.
    """
    from idr.circle_fit import ransac
    from idr.roundabout_detector import ring_distance, rule_predict, window_features
    T = len(w)
    a_lat = np.zeros(T) if a_lat is None else a_lat
    vib = np.zeros(T) if vib is None else vib
    p_stop = np.zeros(T) if p_stop is None else p_stop
    ekf = NavEKF(p0[0], p0[1], psi0, v0, cfg=nav_cfg)
    nhc_state = roundabout if nhc_state is None else nhc_state      # detector-driven NHC inflation
    detect = (roundabout or nhc_state or map_free_circle) if detect is None else detect   # log-only when alone
    ring_mode = roundabout if ring_mode is None else ring_mode      # exit hypotheses + ring residuals in the PF
    cfg = mm_cfg or MMConfig(roundabout_mode=ring_mode, landmark_scoring=landmark, **(mm_overrides or {}))
    mm = MapMatcher(rm, ekf, cfg, route_edges, np.random.default_rng(seed)) if use_map else None
    track = np.zeros((T, 2))
    snapped = np.zeros((T, 2))
    speeds = np.zeros(T)
    dist_log = np.zeros(T)
    ds_log = np.zeros(T)
    feats, states = [], []
    v_ol = v0                                   # open-loop speed: same physics as training, never map-corrected
    state, nhc_scale = 0, 1.0
    for t in range(T):
        d0, psi_prev = ekf.dist, ekf.x[IPSI]
        ekf.propagate(w[t], a[t], a_lat[t], vib[t], p_stop[t], nhc_scale if nhc_state else 1.0)
        speeds[t] = max(ekf.x[IVF], 0.0)
        track[t] = ekf.pos
        dist_log[t] = ekf.dist
        ds_log[t] = ekf.x[7]
        v_ol_new = v_ol + a[t] * 0.1
        dd_ol = max(v_ol_new, 0.0) * (1.0 - p_stop[t]) * 0.1
        v_ol = v_ol_new
        if mm is not None:
            dd = dd_ol if cfg.open_loop_odometry else ekf.dist - d0
            mm.advance(dd, wrap(ekf.x[IPSI] - psi_prev))
        if t % 10 == 9:
            if detect:
                rd = ring_distance(rm, ekf.pos) if (rm is not None and use_map) else np.nan
                feats.append(window_features(w, speeds, a_lat, track, t, rd))
                state = int(rule_predict(np.array(feats[-10:]), use_map=bool(use_map and rm is not None))[-1])
                states.append(state)
                nhc_scale = NHC_STATE_SCALE[state] * (2.0 if state == 3 and abs(a_lat[t]) > 2 else 1.0)
                no_ring = not (use_map and rm is not None and rm.rings) or not (rd < 40)
                if map_free_circle and state == 3 and no_ring and t >= 50:
                    fit = ransac(track[t - 50:t + 1:2], thresh=2.0, iters=60)
                    if fit is not None and 8 < fit["r"] < 45 and fit["rms"] < 3:
                        c = np.array([fit["cx"], fit["cy"]])
                        u = ekf.pos - c
                        nu = np.hypot(*u)
                        if nu > 1e-3:
                            H = np.zeros((1, NS))
                            H[0, IE], H[0, IN] = u / nu
                            ekf.update(np.array([fit["r"] + (u / nu) @ c]), H, np.array([[3.0 ** 2]]), affect=MAP_AFFECT)
            if mm is not None:
                v = max(ekf.x[IVF], 1.0)
                if tft_resid is not None:
                    mm.tft_resid_now = float(np.mean(tft_resid[max(0, t - 9):t + 1]))
                mm.measure(-(w[t] - ekf.x[IBG]) / v)
                if mm.trace:
                    mm.trace[-1]["state"] = state
        if mm is not None:
            snapped[t] = mm.snapped_pose()
        track[t] = ekf.pos
    if mm is None:
        snapped = track.copy()
    stats = dict(mm.stats) if mm else {}
    if return_trace:
        return track, snapped, stats, {"states": states, "mm_trace": mm.trace if mm else [], "dist": dist_log,
                                       "speed_scale": 1.0 + ds_log, "along_events": mm.along_events if mm else []}
    return track, snapped, stats


# ------------------------------------------------------------------------------------------ synthetic self-tests
def _synthetic_map(polylines):
    from idr.osm_map import assemble
    kept, use = [], {}
    for item in polylines:
        refs, xy, hw, ow = item[:4]
        kept.append((hw, ow, refs, np.asarray(xy, float), item[4] if len(item) > 4 else False))
        for r in refs:
            use[r] = use.get(r, 0) + 1
    return assemble(kept, use)


if __name__ == "__main__":
    DT = 0.1
    # 1) straight 2 km road north, 3 deg initial heading error, 15 m/s
    rm = _synthetic_map([([1, 2, 3], [[0, -100], [0, 1000], [0, 2000]], "primary", 0)])
    T = 700
    w, a = np.zeros(T), np.zeros(T)
    track, _, st = run_outage(rm, np.array([0.0, 0.0]), np.radians(3), 15.0, w, a)
    base, _, _ = run_outage(rm, np.array([0.0, 0.0]), np.radians(3), 15.0, w, a, use_map=False)
    xt_mm, xt_raw = abs(track[-1, 0]), abs(base[-1, 0])
    assert xt_mm < 5.0 and xt_raw > 40, (xt_mm, xt_raw)
    print(f"[1] straight road, 3 deg heading error, {15 * T * DT:.0f} m: cross-track {xt_raw:.1f} m -> {xt_mm:.1f} m  {st}")

    # 2) T-junction at (0,500): left branch and right branch; true path goes RIGHT, planned route goes LEFT
    rm = _synthetic_map([([1, 2], [[0, -200], [0, 500]], "secondary", 0),
                         ([2, 3], [[0, 500], [-800, 500]], "secondary", 0),
                         ([2, 4], [[0, 500], [800, 500]], "secondary", 0)])
    left = [e for e in range(len(rm.e_u)) if rm.e_u[e] == 2 and rm.e_v[e] == 3]
    T = 800
    w = np.zeros(T)
    k_turn = int(500 / 10 / DT)                    # reach the junction at 50 s (10 m/s)
    w[k_turn:k_turn + 30] = -np.radians(90) / 3.0   # 90 deg right turn over 3 s (CCW-positive -> negative)
    track, _, st = run_outage(rm, np.array([0.0, 0.0]), 0.0, 10.0, w, np.zeros(T), route_edges=left)
    end_true = np.array([800 * 0 + (T - k_turn - 15) * 1.0, 500.0])
    assert track[-1, 0] > 150 and abs(track[-1, 1] - 500) < 15, track[-1]
    print(f"[2] T-junction, planned route LEFT, driver turns RIGHT: final pos {track[-1].round(1)} (on right branch)  {st}")

    # 3) 60 m initial offset from the road still converges (no fixed cut-off)
    rm = _synthetic_map([([1, 2], [[0, -100], [0, 3000]], "primary", 0)])
    T = 900
    track, _, st = run_outage(rm, np.array([60.0, 0.0]), 0.0, 15.0, np.zeros(T), np.zeros(T),
                              nav_cfg=None)
    assert abs(track[-1, 0]) < 10, track[-1]
    print(f"[3] 60 m initial offset: final cross-track {abs(track[-1, 0]):.1f} m  {st}")
    # ---- roundabout tests: clockwise ring r=20 m at the origin, arms S(entry) W N E; driver takes the 3rd exit (E)
    R = 20.0
    node_b = {1: np.pi, 2: 1.5 * np.pi, 3: 0.0, 4: 0.5 * np.pi}           # bearing of each ring node from center
    ring_pt = lambda b: np.array([R * np.sin(b), R * np.cos(b)])
    polys = []
    order = [1, 2, 3, 4, 1]
    for a_, b_ in zip(order[:-1], order[1:]):
        ba, bb = node_b[a_], node_b[b_]
        bb = bb if bb > ba else bb + 2 * np.pi
        arc = [ring_pt(x) for x in np.linspace(ba, bb, 16)]
        polys.append(([a_] + [1000 + 20 * a_ + k for k in range(14)] + [b_], arc, "primary", 1, True))
    for nid, b in node_b.items():
        polys.append(([nid, 100 + nid], [ring_pt(b), ring_pt(b) * 16], "primary", 0))
    rmr = _synthetic_map(polys)
    assert len(rmr.rings) == 1 and abs(rmr.rings[0]["r"] - R) < 0.5 and rmr.rings[0]["dir"] > 0

    def truth_path():
        pts = [np.array([0.0, -320.0]) + np.array([0.0, 1.0]) * u for u in np.arange(0, 300, 0.5)]
        pts += [ring_pt(b) for b in np.linspace(np.pi, 2.5 * np.pi, 200)]            # 270 deg clockwise
        pts += [np.array([R, 0.0]) + np.array([1.0, 0.0]) * u for u in np.arange(0.5, 300, 0.5)]
        P = np.array(pts)
        sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        return P, sl
    P, sl = truth_path()
    v_true = 8.0
    tt = np.arange(0, sl[-1] / v_true, DT)
    Pt = np.column_stack([np.interp(tt * v_true, sl, P[:, 0]), np.interp(tt * v_true, sl, P[:, 1])])
    hd = np.unwrap(np.arctan2(np.gradient(Pt[:, 0]), np.gradient(Pt[:, 1])))
    yaw = -np.gradient(hd, DT)                                       # CCW-positive yaw rate
    T = len(tt)
    k_exit = int(np.argmax(Pt[:, 0] > R + 1))
    east = next(x["edge"] for x in rmr.rings[0]["exits"] if abs(rmr.bearing[rmr.start if False else rmr.e_start[x["edge"]]] - np.pi / 2) < 0.3)
    north = next(x["edge"] for x in rmr.rings[0]["exits"] if abs(wrap(rmr.bearing[rmr.e_start[x["edge"]]])) < 0.3)

    tr, _, st, tri = run_outage(rmr, Pt[0], 0.0, v_true * 1.10, yaw, np.zeros(T), return_trace=True, ring_mode=True)
    probs = tri["mm_trace"][-1]["exit_probs"]
    assert Pt[-1][0] > 150 and abs(tr[-1, 1]) < 10 and tr[-1, 0] > 100, tr[-1]
    # exit hypotheses at the moment the vehicle actually exits: exit 3 (E) must dominate, exits 1/2 must not
    at_exit = [x for x in tri["mm_trace"] if x["k"] >= k_exit - 10]
    p_e = at_exit[0]["exit_probs"].get(int(east), 0.0) if at_exit else 0.0
    print(f"    P(exit E) when the car exits: {p_e:.2f}; trace ring flags around exit: "
          f"{[(x['k'], x['ring'], round(x['conf'], 2)) for x in tri['mm_trace'] if abs(x['k'] - k_exit) < 60]}")
    print(f"[4] roundabout, 3rd exit, +10% speed error: end {tr[-1].round(1)} (truth {Pt[-1].round(1)}), "
          f"exit probs at end {({('E' if e == east else e): round(p_, 2) for e, p_ in probs.items()})}  {st}")

    E_arm = {int(east)} | ({int(rmr.e_rev[east])} if rmr.e_rev[east] >= 0 else set())

    def decision(route, seed):
        _, _, _, tri_ = run_outage(rmr, Pt[0], 0.0, v_true * 1.10, yaw, np.zeros(T), route_edges=route, seed=seed,
                                   return_trace=True, ring_mode=True)
        ks = [x for x in tri_["mm_trace"] if x["k"] >= k_exit]
        dec = next((x["k_now"] for i, x in enumerate(ks) if all(y["edge"] in E_arm for y in ks[i:])), None)
        wrong = any(y["conf"] >= 0.8 and y["ring"] < 0 and y["edge"] not in E_arm and y["k"] >= k_exit + 10 for y in ks)
        return (None if dec is None else (dec - k_exit) * DT), wrong
    for label, route in [("no route", []), ("planned route = exit 2 (N), driver takes exit 3", [north])]:
        res = [decision(route, sd) for sd in range(5)]
        lat = [r_[0] for r_ in res]
        assert all(l_ is not None and l_ <= 8.0 for l_ in lat) and not any(r_[1] for r_ in res), res
        print(f"[5] {label}: correct-exit decision {min(lat):.1f}-{max(lat):.1f} s after exiting (incl. 3 s lag), "
              f"never committed to a wrong exit (5 seeds)")
    # ---- [7] two nearby junctions, +10% speed: gyro turn event must pick the 2nd junction and fix along-track
    rmj = _synthetic_map([([1, 2], [[0, -200], [0, 500]], "secondary", 0),
                          ([2, 3], [[0, 500], [0, 600]], "secondary", 0),
                          ([3, 4], [[0, 600], [0, 1500]], "secondary", 0),
                          ([2, 5], [[0, 500], [600, 500]], "secondary", 0),
                          ([3, 6], [[0, 600], [600, 600]], "secondary", 0)])
    vj = 10.0
    Pj = [np.array([0.0, float(y)]) for y in np.arange(0, 590, 0.5)]
    Pj += [np.array([10 - 10 * np.cos(a), 590 + 10 * np.sin(a)]) for a in np.linspace(0, np.pi / 2, 40)]
    Pj += [np.array([10 + x, 600.0]) for x in np.arange(0.5, 400, 0.5)]
    Pj = np.array(Pj)
    slj = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Pj, axis=0).T))])
    ttj = np.arange(0, slj[-1] / vj, DT)
    Ptj = np.column_stack([np.interp(ttj * vj, slj, Pj[:, 0]), np.interp(ttj * vj, slj, Pj[:, 1])])
    hdj = np.unwrap(np.arctan2(np.gradient(Ptj[:, 0]), np.gradient(Ptj[:, 1])))
    yawj = -np.gradient(hdj, DT)
    Tj = len(ttj)
    k_turn = int(np.argmax(Ptj[:, 0] > 10))
    for lm, alm in ((False, False), (True, False), (False, True)):
        ok, errs = 0, []
        for sd in range(5):
            trj, snj, stj = run_outage(rmj, Ptj[0], 0.0, vj * 1.10, yawj, np.zeros(Tj), seed=sd, landmark=lm,
                                       mm_overrides={"along_landmark": alm})
            k_chk = min(Tj - 1, k_turn + 150)                            # 15 s after the turn
            on_branch = abs(snj[k_chk, 1] - 600) < 8 and snj[k_chk, 0] > 20
            ok += on_branch
            errs.append(abs(trj[k_chk, 0] - Ptj[k_chk, 0]))
        print(f"[7] two junctions 100 m apart, turn at the 2nd, +10% speed, landmark={lm} along_landmark={alm}: "
              f"correct branch {ok}/5 seeds, along-track error 15 s after the turn median {np.median(errs):.1f} m "
              f"{ {k: v for k, v in stj.items() if k.startswith('alm')} if alm else ''}")
        if alm:
            assert ok >= 4, (ok, errs)
        if lm:
            assert ok >= 4, (ok, errs)       # along-track < 10 m target NOT met yet (13 m): speed scale adapts slowly

    # ---- [8] two identical parallel roads 12 m apart, car starts between them: genuinely ambiguous. The confidence
    # policy must keep both hypotheses alive (tempered likelihood, no EKF snap) at least as long as the plain filter.
    rmp = _synthetic_map([([11, 12], [[0, -100], [0, 900]], "secondary", 0),
                          ([21, 22], [[12, -100], [12, 900]], "secondary", 0)])
    Tp = 600
    res8 = {}
    for pol in (False, True):
        alive, lock_wrong = [], 0
        for sd in range(5):
            _, _, stp, trp = run_outage(rmp, np.array([6.0, 0.0]), 0.0, 10.0, np.zeros(Tp), np.zeros(Tp), seed=sd,
                                        return_trace=True, mm_overrides={"confidence_policy": pol, "confidence_log": True})
            n_alive = [x["n_roads"] >= 2 and x["conf"] < 0.9 for x in trp["mm_trace"]]
            alive.append(float(np.mean(n_alive)))
        res8[pol] = float(np.mean(alive))
    print(f"[8] ambiguous parallel roads: fraction of 1 Hz steps with >=2 live road hypotheses: plain "
          f"{res8[False]:.2f}, confidence policy {res8[True]:.2f}")
    assert res8[True] >= res8[False] - 1e-9

    # map-free: no map, gyro scale error +15% on the ring; the fitted-circle constraint keeps it circular
    ring_k = (Pt[:, 1] > -R - 1) & (Pt[:, 0] < R + 1)
    yaw_bad = yaw + np.where(ring_k, np.random.default_rng(3).normal(0, 0.15, T), 0.0)   # heavy gyro noise on ring
    def radial_err(tr_):
        m = ring_k & (np.arange(T) > np.argmax(ring_k) + 80)
        return float(np.mean(np.abs(np.hypot(*(tr_[m] - tr_[m].mean(0)).T) - np.hypot(*(tr_[m] - tr_[m].mean(0)).T).mean())))
    trf, _, _ = run_outage(None, Pt[0], 0.0, v_true, yaw_bad, np.zeros(T), use_map=False, map_free_circle=True)
    trn, _, _ = run_outage(None, Pt[0], 0.0, v_true, yaw_bad, np.zeros(T), use_map=False, map_free_circle=False)
    from idr.circle_fit import kasa, refine
    circ = lambda tr_: refine(tr_[ring_k][20:], kasa(tr_[ring_k][20:]))[2]
    e_in, e_out = circ(trf), circ(trn)
    print(f"[6] no map, noisy gyro on ring: radial RMS about best-fit circle {e_out:.2f} m (plain DR) -> {e_in:.2f} m "
          f"(map-free circle mode); end error {np.hypot(*(trn[-1] - Pt[-1])):.0f} -> {np.hypot(*(trf[-1] - Pt[-1])):.0f} m")
    assert e_in < 3.0 and e_in <= e_out + 0.05
    print("map_matching OK")
