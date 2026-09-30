"""Planar navigation EKF: x = [E, N, psi, v_f, v_l, b_g, b_a, ds, b_s] (psi clockwise from North, yaw CCW-positive).

v_f is the IMU (TFT) speed integrator. The speed used for position is  v_pos = clip(1+ds, 0.6, 1.4)*max(v_f,0)*gate + b_s
with ds (speed-scale error, lambda_v - 1) and b_s (along-track distance-rate bias) slow random walks. Map pseudo-
measurements are only allowed to change position/heading/ds/b_s (not v_f), so they can recalibrate distance but
cannot drive the IMU speed to zero (the collapse seen in 7.5% of map-aided outages).

Propagated at 10 Hz from (denoised) yaw rate and forward acceleration. Soft, adaptive non-holonomic constraint on the
lateral velocity, ZUPT from the stationary head, and delayed measurements (map corrections for t - lag) applied by
rewinding to that step and re-propagating from buffered inputs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

DT = 0.1
IE, IN, IPSI, IVF, IVL, IBG, IBA, IDS, IBS = range(9)
NS = 9
MAP_AFFECT = (IE, IN, IPSI, IBG, IDS, IBS)      # states a map pseudo-measurement may change


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class NavConfig:
    sig_w: float = 0.02         # gyro noise after denoising, rad/s (per-sample)
    sig_a: float = 0.3          # forward accel noise, m/s^2
    sig_bg_rw: float = 2e-4     # gyro bias random walk, rad/s per sqrt(s)
    sig_ba_rw: float = 5e-3
    sig_ds_rw: float = 2e-3     # speed-scale random walk per sqrt(s) (slow calibration drift, not manoeuvres)
    sig_bs_rw: float = 0.0      # along-track bias rate random walk, m/s per sqrt(s) (0 = disabled; Step 3 used 0.05)
    nu_lat: float = 2.0         # lateral velocity decay rate, 1/s
    sig_vl_proc: float = 0.2
    # adaptive NHC: sigma = base * (1 + k_yaw*|w|/0.1 + k_acc*|a_f| + k_lat*|a_l| + k_vib*vib)
    nhc_base: float = 0.05
    k_yaw: float = 1.0
    k_acc: float = 0.5
    k_lat: float = 0.5
    k_vib: float = 1.0
    nhc_fixed: bool = False
    student_nu: float = 4.0     # heavy-tailed reweighting of NHC innovations
    lag_steps: int = 30         # history kept for delayed corrections (3 s)
    zupt: bool = False          # hard ZUPT reset; off by default: the TFT's soft stop gate already matches training
    sig_ds0: float = 0.02       # initial speed-scale (lambda_v - 1) uncertainty (Step-0 value; Step 3 used 0.10)
    sig_bs0: float = 0.0        # initial along-track bias-rate uncertainty, m/s (0 = disabled; Step 3 used 0.3)
    speed_cal: dict | None = None   # speed calibration g(v) fitted on training drives (idr.speed_calibration); None = off


class NavEKF:
    def __init__(self, e, n, psi, v, cfg: NavConfig | None = None, bg0=0.0, sig_psi0=np.radians(1.0),
                 sig_v0=0.3, sig_bg0=0.01, sig_ds0=None, sig_bs0=None):
        self.cfg = cfg or NavConfig()
        sig_ds0 = self.cfg.sig_ds0 if sig_ds0 is None else sig_ds0
        sig_bs0 = self.cfg.sig_bs0 if sig_bs0 is None else sig_bs0
        self.x = np.array([e, n, psi, v, 0.0, bg0, 0.0, 0.0, 0.0], dtype=float)
        # sig_ds0 ~ realistic speed-scale error of the (denoised) IMU speed; lets along-track uncertainty grow with
        # distance so the map matcher does not over-trust a position that is off by a few percent of distance
        self.P = np.diag([3.0, 3.0, sig_psi0, sig_v0, 0.1, sig_bg0, 0.1, sig_ds0, sig_bs0]) ** 2
        self.v0 = float(v)                                  # GNSS speed at outage start (anchor for speed calibration)
        self.k = 0
        self.hist_len = self.cfg.lag_steps + 1              # longer only for the turn-landmark replay (map_matching)
        self.hist = deque(maxlen=self.hist_len)             # (k, x, P, input) before propagating step k
        self.dist = 0.0
        self.nhc_nis = 1.0                                  # EMA of the NHC normalised innovation (log only)

    # ---- propagation ----
    def _f(self, x, w, a, p_stop=0.0):
        psi, vf, vl, bg, ba, ds, bs = x[IPSI], x[IVF], x[IVL], x[IBG], x[IBA], x[IDS], x[IBS]
        c = self.cfg
        s, co = np.sin(psi), np.cos(psi)
        xn = x.copy()
        # IMU speed integrator: same kinematics as idr.physics.integrate (what the TFT was trained through)
        vf_new = vf + (a - ba) * DT
        gate = (1.0 - p_stop) if vf_new > 0 else 0.0
        lam = float(np.clip(1.0 + ds, 0.6, 1.4))
        d_lam = 1.0 if 0.6 < 1.0 + ds < 1.4 else 0.0
        vb = lam * max(vf_new, 0.0)
        if c.speed_cal is not None:
            from idr.speed_calibration import apply_cal, cal_slope
            g, slope = float(apply_cal(vb, c.speed_cal, v0=self.v0)), cal_slope(vb, c.speed_cal)
        else:
            g, slope = vb, 1.0
        vpos = g * gate + bs
        xn[IVF] = vf_new
        xn[IE] += (vpos * s + vl * co) * DT
        xn[IN] += (vpos * co - vl * s) * DT
        xn[IPSI] = wrap(psi - (w - bg) * DT)
        xn[IVL] = vl * (1 - c.nu_lat * DT)
        F = np.eye(NS)
        F[IE, IPSI] = (vpos * co - vl * s) * DT
        F[IE, IVF] = slope * lam * gate * s * DT
        F[IE, IVL] = co * DT
        F[IE, IDS] = slope * d_lam * vf_new * gate * s * DT
        F[IE, IBS] = s * DT
        F[IN, IPSI] = (-vpos * s - vl * co) * DT
        F[IN, IVF] = slope * lam * gate * co * DT
        F[IN, IVL] = -s * DT
        F[IN, IDS] = slope * d_lam * vf_new * gate * co * DT
        F[IN, IBS] = co * DT
        F[IPSI, IBG] = DT
        F[IVF, IBA] = -DT
        F[IVL, IVL] = 1 - c.nu_lat * DT
        return xn, F

    def _Q(self):
        c = self.cfg
        q = np.zeros(NS)
        q[IPSI] = (c.sig_w * DT) ** 2
        q[IVF] = (c.sig_a * DT) ** 2
        q[IVL] = (c.sig_vl_proc * DT) ** 2
        q[IBG] = c.sig_bg_rw ** 2 * DT
        q[IBA] = c.sig_ba_rw ** 2 * DT
        q[IDS] = c.sig_ds_rw ** 2 * DT
        q[IBS] = c.sig_bs_rw ** 2 * DT
        q[IE] = q[IN] = 0.01 * DT
        return np.diag(q)

    def _step(self, inp):
        w, a, a_lat, vib, p_stop, nhc_scale = inp
        self.x, F = self._f(self.x, w, a, p_stop)
        self.P = F @ self.P @ F.T + self._Q()
        self._nhc(w, a, a_lat, vib, nhc_scale)
        if self.cfg.zupt and p_stop > 0.7:
            self.update(np.array([0.0, 0.0]), self._H([IVF, IVL]), np.diag([0.05, 0.05]) ** 2)
        lam = float(np.clip(1.0 + self.x[IDS], 0.6, 1.4))
        vb = lam * max(self.x[IVF], 0.0)
        if self.cfg.speed_cal is not None:
            from idr.speed_calibration import apply_cal
            vb = float(apply_cal(vb, self.cfg.speed_cal, v0=self.v0))
        self.dist += max(vb * (1.0 - p_stop) + self.x[IBS], 0.0) * DT

    def propagate(self, w, a, a_lat=0.0, vib=0.0, p_stop=0.0, nhc_scale=1.0):
        """nhc_scale: manoeuvre-state multiplier on the NHC sigma (e.g. x3 at roundabout entry/exit)."""
        inp = (float(w), float(a), float(a_lat), float(vib), float(p_stop), float(nhc_scale))
        self.hist.append((self.k, self.x.copy(), self.P.copy(), self.dist, inp))
        self._step(inp)
        self.k += 1

    # ---- measurements ----
    @staticmethod
    def _H(idx):
        H = np.zeros((len(idx), NS))
        for r, i in enumerate(idx):
            H[r, i] = 1.0
        return H

    def nhc_sigma(self, w, a, a_lat, vib):
        c = self.cfg
        if c.nhc_fixed:
            return c.nhc_base
        return c.nhc_base * (1 + c.k_yaw * abs(w) / 0.1 + c.k_acc * abs(a) + c.k_lat * abs(a_lat) + c.k_vib * vib)

    def _nhc(self, w, a, a_lat, vib, nhc_scale=1.0):
        sig = self.nhc_sigma(w, a, a_lat, vib) * nhc_scale
        H = self._H([IVL])
        S = float((H @ self.P @ H.T)[0, 0]) + sig ** 2
        r = -self.x[IVL]
        # Student-t reweighting: inflate R when the innovation is implausibly large (heavy-tailed NHC noise)
        nu = self.cfg.student_nu
        scale = (nu + 1) / (nu + r * r / S)
        self.nhc_nis = 0.98 * self.nhc_nis + 0.02 * r * r / S
        self.update(np.array([0.0]), H, np.array([[sig ** 2 / max(scale, 1e-3)]]))

    def update(self, z, H, R, angle_rows=(), gate=None, affect=None):
        """affect: indices of states this measurement may change (Schmidt / 'consider' update). Map pseudo-
        measurements use it so they correct position/heading but never feed back into speed, which would create a
        loop (slower EKF -> particles lag -> map pulls EKF back -> slower EKF)."""
        y = z - H @ self.x
        for i in angle_rows:
            y[i] = wrap(y[i])
        S = H @ self.P @ H.T + R
        if gate is not None and float(y @ np.linalg.solve(S, y)) > gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        if affect is not None:
            mask = np.zeros(NS, dtype=bool)
            mask[list(affect)] = True
            K[~mask] = 0.0
        self.x = self.x + K @ y
        self.x[IPSI] = wrap(self.x[IPSI])
        I_KH = np.eye(NS) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        return True

    def delayed_update(self, k_meas, z, H, R, angle_rows=(), gate=None, affect=None) -> bool:
        """Apply a measurement valid at step k_meas (< now) and re-propagate buffered inputs to the present."""
        items = list(self.hist)
        idx = next((i for i, it in enumerate(items) if it[0] == k_meas), None)
        if idx is None:
            return self.update(z, H, R, angle_rows, gate, affect)
        cur = (self.x.copy(), self.P.copy(), self.dist)
        _, self.x, self.P, self.dist, _ = (items[idx][0], items[idx][1].copy(), items[idx][2].copy(),
                                           items[idx][3], None)
        ok = self.update(z, H, R, angle_rows, gate, affect)
        if not ok:
            self.x, self.P, self.dist = cur
            return False
        new_hist = items[:idx]
        for kk, _, _, _, inp in items[idx:]:
            new_hist.append((kk, self.x.copy(), self.P.copy(), self.dist, inp))
            self._step(inp)
        self.hist = deque(new_hist, maxlen=self.hist_len)
        return True

    def state_at(self, k):
        for kk, x, P, d, _ in self.hist:
            if kk == k:
                return x, P, d
        return self.x, self.P, self.dist

    @property
    def pos(self):
        return self.x[[IE, IN]]

    def pos_sigma(self):
        return float(np.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1) gyro bias observable from periodic heading fixes on a straight road
    bias = np.radians(0.5)
    f = NavEKF(0, 0, 0.0, 15.0)
    for k in range(1200):
        f.propagate(w=bias + rng.normal(0, 0.01), a=0.0)
        if k % 10 == 9:
            f.update(np.array([0.0]), f._H([IPSI]), np.array([[np.radians(2.0) ** 2]]), angle_rows=(0,))
    assert abs(f.x[IBG] - bias) < np.radians(0.1), np.degrees(f.x[IBG])
    # 2) adaptive NHC sigma inflates in a sharp turn
    s_straight, s_turn = f.nhc_sigma(0.0, 0.0, 0.0, 0.0), f.nhc_sigma(np.radians(30), 0.0, 3.0, 0.0)
    assert s_turn > 5 * s_straight
    # 3) delayed update + re-propagation == applying it at the right time in a batch run
    ws = rng.normal(0, 0.05, 60)
    a = NavEKF(0, 0, 0.3, 10.0)
    b = NavEKF(0, 0, 0.3, 10.0)
    z, H, R = np.array([5.0]), NavEKF._H([IE]), np.array([[4.0]])
    for k in range(60):
        b.propagate(ws[k], 0.1)
        if k == 29:
            b.update(z, H, R)
    for k in range(60):
        a.propagate(ws[k], 0.1)
    a.delayed_update(30, z, H, R)
    assert np.allclose(a.x, b.x, atol=1e-9) and np.allclose(a.P, b.P, atol=1e-9)
    print(f"nav_filter OK: bias est {np.degrees(f.x[IBG]):.3f} deg/s (true 0.5), "
          f"NHC sigma straight {s_straight:.3f} vs turn {s_turn:.3f} m/s, delayed update == batch")
