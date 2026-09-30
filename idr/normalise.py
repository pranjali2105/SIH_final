"""Phone -> vehicle frame alignment, bias removal and standardisation.

The IO-VNBD phone axes do not follow a clean convention (e.g. yaw shows up on gyro Y while gravity sits on accel Z),
so instead of assuming a mount geometry we *calibrate* the projection from phone axes to vehicle axes against
GNSS-observable motion, exactly what the app can do while GNSS is still available:
  - gyro yaw axis   <- least squares of GNSS course rate (here: vehicle yaw rate) on the 3 gyro axes
  - accel forward   <- least squares of GNSS speed derivative on the 3 accel axes (+ intercept)
  - accel lateral   <- least squares of centripetal accel v*yaw_rate on the 3 accel axes (+ intercept)
  - up              <- mean accel direction (gravity)
Calibration is fitted once per phone recording segment (the mount can change between recordings).

Run `python -m idr.normalise` to align every drive, cache the result and print alignment quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from idr.data import CACHE_DIR, DT, load_all

ALIGNED_DIR = CACHE_DIR / "aligned"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
STAT_COLS = ["a_fwd", "a_lat", "a_up", "w_roll", "w_pitch", "w_yaw", "a_hmag", "w_mag"]
CLIP_SIGMA = 6.0


def _lstsq(X: np.ndarray, y: np.ndarray, intercept: bool) -> np.ndarray:
    if intercept:
        X = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Contiguous runs of imu_valid=1, split where the phone clock resets (new recording/mount)."""
    v = df["imu_valid"].to_numpy() > 0
    t = df["s_t_ms"].to_numpy()
    cut = np.zeros(len(df), dtype=bool)
    cut[1:] = (np.diff(t) < 0) | (np.diff(t) > 1000) | (v[1:] != v[:-1])
    starts = np.flatnonzero(cut).tolist()
    bounds = [0, *starts, len(df)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if v[a] and b - a >= 300]


def calibrate(seg: pd.DataFrame) -> dict:
    """Fit phone->vehicle projections on one segment. Returns vectors usable on-device."""
    A = seg[["ax", "ay", "az"]].to_numpy()
    W = seg[["wx", "wy", "wz"]].to_numpy()
    v = seg["v_true"].to_numpy()
    yaw = seg["yaw_rate_true"].to_numpy()
    moving = v > 1.0
    still = uniform_filter1d((v < 0.1).astype(float), 20) > 0.99

    gyro_bias = W[still].mean(0) if still.sum() > 20 else np.median(W, 0)
    Wc = W - gyro_bias
    Ws = uniform_filter1d(Wc, 5, axis=0)
    u_yaw = _lstsq(Ws[moving], yaw[moving], intercept=False)            # rad/s per phone axis (sign+scale)

    up = A.mean(0)
    up /= np.linalg.norm(up)
    As = uniform_filter1d(A, 10, axis=0)                                 # 1 s low-pass for the fit only
    dv = np.gradient(uniform_filter1d(v, 10), DT)
    a_c = v * yaw                                                        # centripetal (lateral) accel
    c_fwd = _lstsq(As[moving], dv[moving], intercept=True)
    c_lat = _lstsq(As[moving], a_c[moving], intercept=True)

    # Two unit axes orthogonal to the yaw axis for roll/pitch rates (sign/order arbitrary but fixed).
    k = u_yaw / (np.linalg.norm(u_yaw) + 1e-9)
    e1 = np.cross(k, [1.0, 0, 0]) if abs(k[0]) < 0.9 else np.cross(k, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)

    M = seg[["mx", "my", "mz"]].to_numpy()
    return {
        "gyro_bias": gyro_bias.tolist(), "u_yaw": u_yaw.tolist(), "e_roll": e1.tolist(), "e_pitch": e2.tolist(),
        "up": up.tolist(), "c_fwd": c_fwd.tolist(), "c_lat": c_lat.tolist(),
        "mag_hard_iron": M.mean(0).tolist(),
    }


def apply_calibration(seg: pd.DataFrame, cal: dict) -> pd.DataFrame:
    A = seg[["ax", "ay", "az"]].to_numpy()
    W = seg[["wx", "wy", "wz"]].to_numpy() - np.asarray(cal["gyro_bias"])
    M = seg[["mx", "my", "mz"]].to_numpy() - np.asarray(cal["mag_hard_iron"])
    c_fwd, c_lat, up = np.asarray(cal["c_fwd"]), np.asarray(cal["c_lat"]), np.asarray(cal["up"])
    out = pd.DataFrame(index=seg.index)
    out["a_fwd"] = A @ c_fwd[:3] + c_fwd[3]
    out["a_lat"] = A @ c_lat[:3] + c_lat[3]
    out["a_up"] = A @ up - 9.80665
    out["w_yaw"] = W @ np.asarray(cal["u_yaw"])
    out["w_roll"] = W @ np.asarray(cal["e_roll"])
    out["w_pitch"] = W @ np.asarray(cal["e_pitch"])
    out["a_hmag"] = np.hypot(out["a_fwd"], out["a_lat"])
    out["w_mag"] = np.linalg.norm(W, axis=1)
    # Magnetometer azimuth in the vehicle frame: project onto forward / lateral directions of the phone.
    f = c_fwd[:3] / (np.linalg.norm(c_fwd[:3]) + 1e-9)
    lat = c_lat[:3] / (np.linalg.norm(c_lat[:3]) + 1e-9)
    out["psi_mag_raw"] = np.arctan2(M @ lat, M @ f)   # offset vs true north is removed per sample from GNSS context
    return out


def align_drive(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    cols = STAT_COLS + ["psi_mag_raw"]
    res = pd.DataFrame(np.nan, index=df.index, columns=cols)
    cals = []
    for a, b in _segments(df):
        seg = df.iloc[a:b]
        cal = calibrate(seg)
        res.iloc[a:b] = apply_calibration(seg, cal)[cols].to_numpy()
        cals.append({"start": a, "end": b, **cal})
    keep = ["drive", "v_true", "psi_true", "heading_valid", "vE_true", "vN_true", "yaw_rate_true", "pE", "pN",
            "lat", "lon", "imu_valid", "s_speed_ms", "s_course_deg"]
    out = pd.concat([df[keep].reset_index(drop=True), res.reset_index(drop=True)], axis=1)
    out["imu_valid"] = (out["imu_valid"] > 0) & out["a_fwd"].notna()
    return out, cals


def load_aligned(refresh: bool = False) -> dict[str, pd.DataFrame]:
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    out, all_cals = {}, {}
    raw = load_all()
    for name, df in raw.items():
        cache = ALIGNED_DIR / f"{name}.parquet"
        if cache.exists() and not refresh:
            out[name] = pd.read_parquet(cache)
            continue
        al, cals = align_drive(df)
        al.to_parquet(cache)
        out[name] = al
        all_cals[name] = cals
    if all_cals:
        (ALIGNED_DIR / "calibrations.json").write_text(json.dumps(all_cals, indent=1))
    return out


def fit_stats(drives: dict[str, pd.DataFrame], names: list[str]) -> dict:
    """Per-channel mean/std from TRAINING drives only (never per file)."""
    X = pd.concat([drives[n].loc[drives[n]["imu_valid"], STAT_COLS] for n in names if n in drives])
    stats = {c: {"mean": float(X[c].mean()), "std": float(X[c].std() + 1e-6)} for c in STAT_COLS}
    stats["_clip_sigma"] = CLIP_SIGMA
    stats["_speed_scale"] = 30.0      # velocity targets/v0 share one scale so E/N angles are preserved
    return stats


def standardise(x: np.ndarray, stats: dict, cols: list[str] = STAT_COLS) -> np.ndarray:
    mu = np.array([stats[c]["mean"] for c in cols], dtype=np.float32)
    sd = np.array([stats[c]["std"] for c in cols], dtype=np.float32)
    z = (x - mu) / sd
    return np.clip(z, -stats["_clip_sigma"], stats["_clip_sigma"])


if __name__ == "__main__":
    drives = load_aligned(refresh=True)
    rows = []
    for name, d in drives.items():
        m = d["imu_valid"].to_numpy() & (d["v_true"] > 1).to_numpy()
        if m.sum() < 600:
            continue
        dv = np.gradient(uniform_filter1d(d["v_true"].to_numpy(), 10), DT)
        r_yaw = np.corrcoef(d.loc[m, "w_yaw"], d.loc[m, "yaw_rate_true"])[0, 1]
        r_fwd = np.corrcoef(uniform_filter1d(d["a_fwd"].fillna(0).to_numpy(), 10)[m], dv[m])[0, 1]
        rows.append((name, int(m.sum()), round(r_yaw, 3), round(r_fwd, 3)))
    rep = pd.DataFrame(rows, columns=["drive", "moving_valid_rows", "r_yaw", "r_fwd_acc(1s)"])
    print(rep.to_string(index=False))
    w = rep["moving_valid_rows"]
    print(f"\nweighted mean r_yaw={np.average(rep.r_yaw, weights=w):.3f}  r_fwd={np.average(rep['r_fwd_acc(1s)'], weights=w):.3f}")
