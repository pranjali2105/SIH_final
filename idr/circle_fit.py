"""Robust circle fitting: algebraic (Kasa) -> weighted Gauss-Newton refinement -> RANSAC wrapper."""
from __future__ import annotations

import numpy as np


def kasa(p: np.ndarray):
    """Algebraic least squares: (x-cx)^2+(y-cy)^2=r^2 linearised. Returns (cx, cy, r)."""
    A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = (p ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c[2] + c[0] ** 2 + c[1] ** 2
    return float(c[0]), float(c[1]), float(np.sqrt(max(r2, 1e-9)))


def refine(p: np.ndarray, c0, w: np.ndarray | None = None, iters: int = 10):
    """Weighted Gauss-Newton on geometric residuals ||p - c|| - r. Returns (cx, cy, r), covariance (3x3), rms."""
    cx, cy, r = c0
    w = np.ones(len(p)) if w is None else w
    cov = np.full((3, 3), np.nan)
    for _ in range(iters):
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        d = np.hypot(dx, dy) + 1e-9
        res = d - r
        J = np.column_stack([-dx / d, -dy / d, -np.ones(len(p))])
        JW = J * w[:, None]
        H = J.T @ JW
        try:
            step = np.linalg.solve(H, -(JW.T @ res))
        except np.linalg.LinAlgError:
            break
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-4:
            break
    dx, dy = p[:, 0] - cx, p[:, 1] - cy
    res = np.hypot(dx, dy) - r
    rms = float(np.sqrt(np.average(res ** 2, weights=w)))
    try:
        dof = max(len(p) - 3, 1)
        cov = np.linalg.inv(H) * (np.sum(w * res ** 2) / dof)
    except Exception:
        pass
    return (float(cx), float(cy), float(abs(r))), cov, rms


def ransac(p: np.ndarray, w: np.ndarray | None = None, thresh: float = 2.0, iters: int = 200,
           rng=None, min_inlier_frac: float = 0.5):
    """RANSAC over 3-point circles, then refine on inliers. Returns dict or None."""
    rng = rng or np.random.default_rng(0)
    n = len(p)
    if n < 5:
        return None
    best_in = None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        try:
            c = kasa(p[i])
        except Exception:
            continue
        if not np.isfinite(c).all() or c[2] > 1e4:
            continue
        inl = np.abs(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - c[2]) < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None or best_in.mean() < min_inlier_frac:
        return None
    c, cov, rms = refine(p[best_in], kasa(p[best_in]), None if w is None else w[best_in])
    return {"cx": c[0], "cy": c[1], "r": c[2], "cov": cov, "rms": rms, "inlier_frac": float(best_in.mean()),
            "inliers": best_in}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    th = np.linspace(0.3, 3.5, 60)                               # ~180 deg partial arc
    p = np.column_stack([5 + 20 * np.sin(th), -3 + 20 * np.cos(th)]) + rng.normal(0, 1.0, (60, 2))
    out = rng.choice(60, 18, replace=False)                      # 30% outliers
    p[out] += rng.normal(0, 15, (18, 2))
    f = ransac(p, rng=rng)
    ce, re = np.hypot(f["cx"] - 5, f["cy"] + 3), abs(f["r"] - 20)
    assert ce < 1.0 and re < 0.5, (ce, re, f)
    print(f"circle_fit OK: center error {ce:.2f} m, radius error {re:.2f} m, rms {f['rms']:.2f} m, "
          f"inliers {f['inlier_frac']:.2f}")
