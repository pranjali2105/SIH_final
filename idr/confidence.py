"""Confidence layer for the map-aided filter: gates and softens map corrections, never hard-snaps.

Three separate confidences, then an action from explicit gates (no learned weighted sum yet):
  dead reckoning   EKF covariance (position, heading, speed scale), distance since the last map fix, NHC innovation,
                   TFT residual magnitude
  map match        weighted map NLL components R_d (cross), R_s (along), R_psi, R_kappa, R_T (turn), EKF-vs-road
                   cross-track / heading residual, particle ESS and cloud dispersion
  decision         best / second-best road-hypothesis probabilities, margin and log ratio

Actions:
  hard_gated_update    normal map likelihood + EKF map updates + gated along-track (junction) updates
  soft_map_update      tempered likelihood, inflated / clipped EKF cross-track update, no along-track update
  preserve_hypotheses  tempered likelihood only (the map can reject impossible roads, never snaps the EKF)
  expand_support       as preserve + extra, wider reseeding; wait for a discriminative event
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from idr.nav_filter import IDS, IE, IN, IPSI, wrap

ACTIONS = ("hard_gated_update", "soft_map_update", "preserve_hypotheses", "expand_support")
BETA = {"hard_gated_update": 1.0, "soft_map_update": 0.7, "preserve_hypotheses": 0.4, "expand_support": 0.4}


@dataclass
class ConfGates:
    pos_sigma_hard_m: float = 25.0
    psi_sigma_hard_deg: float = 20.0
    ess_hard: float = 0.20
    p1_hard: float = 0.60
    margin_hard: float = 0.20
    cross_hard_m: float = 15.0
    psi_err_hard_deg: float = 25.0
    p1_soft: float = 0.40
    ess_soft: float = 0.10
    pos_sigma_soft_m: float = 50.0
    p1_expand: float = 0.30
    ess_expand: float = 0.05
    cloud_expand_m: float = 60.0
    resid_expand: float = 9.0          # best road's mean (R_d + R_psi), in squared-sigma units (3 sigma)
    tau_ekf_m2: float = 400.0          # c_EKF = exp(-u_EKF / tau), u_EKF = tr(P_xy) + lam_psi * P_psi
    lam_psi: float = 1000.0            # m^2 per rad^2 (1 deg heading ~ 0.3 m^2)
    tau_map: float = 3.0               # c_map = exp(-(R_d + R_psi + R_s) / tau)


@dataclass
class ConfidenceReport:
    ekf_confidence: float
    map_confidence: float
    route_confidence: float
    ess_ratio: float
    cloud_sigma_m: float
    pos_sigma_m: float
    heading_sigma_deg: float
    speed_scale_sigma: float
    dist_since_fix_m: float
    nhc_nis: float
    tft_resid: float | None
    cross_track_residual_m: float
    heading_residual_rad: float
    along_track_residual_m: float
    R_map: dict = field(default_factory=dict)
    best_route_probability: float = 0.0
    route_margin: float = 0.0
    route_log_ratio: float = 0.0
    action: str = "hard_gated_update"

    def compact(self):
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def decide_action(r: ConfidenceReport, g: ConfGates, best_resid: float) -> str:
    if (r.best_route_probability < g.p1_expand or r.ess_ratio < g.ess_expand or r.cloud_sigma_m > g.cloud_expand_m
            or best_resid > g.resid_expand):
        return "expand_support"
    if (r.pos_sigma_m < g.pos_sigma_hard_m and r.heading_sigma_deg < g.psi_sigma_hard_deg and r.ess_ratio > g.ess_hard
            and r.best_route_probability > g.p1_hard and r.route_margin > g.margin_hard
            and r.cross_track_residual_m < g.cross_hard_m and np.degrees(abs(r.heading_residual_rad)) < g.psi_err_hard_deg):
        return "hard_gated_update"
    if r.best_route_probability > g.p1_soft and r.ess_ratio > g.ess_soft and r.pos_sigma_m < g.pos_sigma_soft_m:
        return "soft_map_update"
    return "preserve_hypotheses"


def evaluate(mm, g: ConfGates | None = None) -> ConfidenceReport:
    """Confidence of the current map-matcher state (called at the start of the 1 Hz measure step)."""
    g = g or ConfGates()
    ekf, rm = mm.ekf, mm.rm
    P = ekf.P
    u_ekf = P[IE, IE] + P[IN, IN] + g.lam_psi * P[IPSI, IPSI]
    # particle cloud
    w = mm.w / mm.w.sum()
    pos = mm._positions(mm.edge, mm.s, mm.d)
    mu = (w[:, None] * pos).sum(0)
    cloud = float(np.sqrt((w * ((pos - mu) ** 2).sum(1)).sum()))
    # road hypotheses
    keys = np.array([mm._road_key(e) for e in mm.edge])
    uniq, inv = np.unique(keys, return_inverse=True)
    share = np.bincount(inv, weights=w)
    order = np.argsort(share)[::-1]
    p1 = float(share[order[0]])
    p2 = float(share[order[1]]) if len(order) > 1 else 0.0
    # EKF vs the best road (weighted over its particles)
    sel = inv == order[0]
    pts, brg, _, _ = mm._road_geom()
    ws = w[sel] / w[sel].sum()
    delta = ekf.pos - pts[sel]
    nrm = np.column_stack([np.cos(brg[sel]), -np.sin(brg[sel])])
    tng = np.column_stack([np.sin(brg[sel]), np.cos(brg[sel])])
    cross = float(abs((ws * ((delta * nrm).sum(1) - mm.d[sel])).sum()))
    along = float(abs((ws * (delta * tng).sum(1)).sum()))
    b_best = float(np.angle(np.sum(ws * np.exp(1j * brg[sel]))))
    dpsi = float(wrap(ekf.x[IPSI] - b_best))
    R = dict(getattr(mm, "_R_map", {}) or {})
    R_best = getattr(mm, "_R_best", None)
    best_resid = float(R_best) if R_best is not None else 0.0
    c_map = float(np.exp(-(R.get("d", 0.0) + R.get("psi", 0.0) + R.get("s", 0.0)) / g.tau_map)) if R else 1.0
    rep = ConfidenceReport(
        ekf_confidence=float(np.exp(-u_ekf / g.tau_ekf_m2)), map_confidence=c_map, route_confidence=p1 - p2,
        ess_ratio=float(mm._ess_post / len(w)), cloud_sigma_m=cloud, pos_sigma_m=float(np.sqrt(P[IE, IE] + P[IN, IN])),
        heading_sigma_deg=float(np.degrees(np.sqrt(P[IPSI, IPSI]))), speed_scale_sigma=float(np.sqrt(P[IDS, IDS])),
        dist_since_fix_m=float(mm.dist_since_fix), nhc_nis=float(getattr(ekf, "nhc_nis", np.nan)),
        tft_resid=getattr(mm, "tft_resid_now", None), cross_track_residual_m=cross, heading_residual_rad=dpsi,
        along_track_residual_m=along, R_map=R, best_route_probability=p1, route_margin=p1 - p2,
        route_log_ratio=float(np.log((p1 + 1e-3) / (p2 + 1e-3))))
    rep.action = decide_action(rep, g, best_resid)
    return rep


def junction_confidence(d_perp, d_psi, turn_err, dt_s, p1, p2, sig_perp=5.0, sig_psi_deg=10.0, sig_turn_deg=15.0,
                        sig_t=2.0):
    """C_junction = C_edge * C_turn * C_timing * C_cloud, with the factors."""
    c_edge = float(np.exp(-0.5 * (d_perp / sig_perp) ** 2) * np.exp(-0.5 * (d_psi / np.radians(sig_psi_deg)) ** 2))
    c_turn = float(np.exp(-0.5 * (turn_err / np.radians(sig_turn_deg)) ** 2))
    c_time = float(np.exp(-0.5 * (dt_s / sig_t) ** 2))
    c_cloud = float(max(p1 - p2, 0.0))
    return {"C_edge": c_edge, "C_turn": c_turn, "C_timing": c_time, "C_cloud": c_cloud,
            "C_junction": c_edge * c_turn * c_time * c_cloud}
