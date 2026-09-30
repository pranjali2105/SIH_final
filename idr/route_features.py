"""Destination + planned-route features. The app must build exactly these tensors from its route polyline.

Route tensor (N_PTS x 4), sampled every STEP_M metres of arc length starting at the last GNSS fix:
    [sin(rel_bearing), cos(rel_bearing), curvature (rad/m * 20), valid]
rel_bearing = bearing of the route segment starting at that point minus psi0 (last GNSS heading), radians,
bearing clockwise from North. Points past the destination have valid=0 and zeros elsewhere.

Destination features (4): [log1p(dist_m/100)/5, sin(rel_bearing_to_dest), cos(rel_bearing_to_dest), route_available]
"""
from __future__ import annotations

import numpy as np

STEP_M = 20.0
N_PTS = 100


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resample_polyline(e: np.ndarray, n: np.ndarray, step: float = STEP_M):
    """Resample a polyline (metres, ENU) at equal arc-length spacing. Returns (E, N, s_grid) of the samples."""
    seg = np.hypot(np.diff(e), np.diff(n))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step:
        return e[:1], n[:1], s[:1]
    grid = np.arange(0.0, s[-1], step)
    return np.interp(grid, s, e), np.interp(grid, s, n), grid


def polyline_bearings(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Bearing (clockwise from North) of each resampled segment; the last point repeats the previous bearing."""
    if len(e) < 2:
        return np.zeros(len(e))
    b = np.arctan2(np.diff(e), np.diff(n))
    return np.concatenate([b, b[-1:]])


def route_tensor(bearings: np.ndarray, psi0: float, n_valid: int) -> np.ndarray:
    """bearings: absolute bearings of consecutive 20 m segments starting at the last fix (any length)."""
    out = np.zeros((N_PTS, 4), dtype=np.float32)
    k = int(min(n_valid, len(bearings), N_PTS))
    if k == 0:
        return out
    rel = wrap(bearings[:k] - psi0)
    curv = np.zeros(k)
    if k > 1:
        curv[:-1] = wrap(np.diff(bearings[:k])) / STEP_M
    out[:k, 0], out[:k, 1] = np.sin(rel), np.cos(rel)
    out[:k, 2] = curv * STEP_M
    out[:k, 3] = 1.0
    return out


def destination_features(p0: np.ndarray, p_dest: np.ndarray, psi0: float, route_available: bool) -> np.ndarray:
    d = p_dest - p0
    dist = float(np.hypot(*d))
    rel = wrap(np.arctan2(d[0], d[1]) - psi0) if dist > 1.0 else 0.0
    return np.array([np.log1p(dist / 100.0) / 5.0, np.sin(rel), np.cos(rel), float(route_available)], dtype=np.float32)


def app_route_features(route_e: np.ndarray, route_n: np.ndarray, psi0: float,
                       dest_e: float, dest_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation for the app: route polyline (ENU metres, starting at the last GNSS fix)."""
    e, n, _ = resample_polyline(route_e, route_n)
    b = polyline_bearings(e, n)
    route = route_tensor(b, psi0, len(b))
    dest = destination_features(np.array([route_e[0], route_n[0]]), np.array([dest_e, dest_n]), psi0, True)
    return route, dest


if __name__ == "__main__":
    # straight route north, psi0 = north: constant bearing 0, zero curvature
    r, d = app_route_features(np.zeros(50), np.linspace(0, 1000, 50), 0.0, 0.0, 1000.0)
    assert np.allclose(r[r[:, 3] > 0, 0], 0) and np.allclose(r[r[:, 3] > 0, 2], 0), r[:5]
    # L-shaped: 500 m north then 500 m east -> one 90 deg step at index 25
    e = np.r_[np.zeros(26), np.linspace(20, 500, 25)]
    n = np.r_[np.linspace(0, 500, 26), np.full(25, 500.0)]
    r, _ = app_route_features(e, n, 0.0, 500.0, 500.0)
    rel = np.degrees(np.arctan2(r[:, 0], r[:, 1]))[r[:, 3] > 0]
    jumps = np.flatnonzero(np.abs(np.diff(rel)) > 1)
    assert len(jumps) == 1 and abs(rel[jumps[0] + 1] - 90) < 1e-6, (rel, jumps)
    print("route_features OK: straight + L-shape; corner at index", jumps[0] + 1)
