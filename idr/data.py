"""Load and clean paired IO-VNBD drives (phone S-* + vehicle V-*) into one tidy table per drive.

Run `python -m idr.data` to rebuild the cache and print a per-drive summary with sanity checks.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNC_DIR = ROOT / "raw" / "Synchronised V abd S datasets" / "Uncategorised IOVNB Dataset"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DT = 0.1  # both sources are 10 Hz
EARTH_R = 6_378_137.0

V_COLS = [
    "sats_raw", "tod", "lat", "lon", "gps_speed_kmh", "heading_deg", "height_m", "vert_vel_kmh",
    "sample_period", "steer_deg", "ws_fl", "ws_fr", "ws_rl", "ws_rr", "yaw_rate_dps",
    "speed_kmh", "acc_long_g", "acc_lat_g", "handbrake", "gear_req", "gear", "rpm", "coolant_c",
    "clutch", "brake_psi", "brake_on", "battery_v", "air_c", "throttle_pct",
]
S_COLS = [
    "s_lat", "s_lon", "s_alt", "s_speed_ms", "s_acc_m", "s_course_deg", "s_sats", "s_t_ms", "s_date",
    "ax", "ay", "az", "gx", "gy", "gz", "wx", "wy", "wz", "mx", "my", "mz",
    "o_azimuth", "o_pitch", "o_roll",
]


def _files(sub: str, prefix: str) -> dict[str, Path]:
    # Filenames differ in case between S and V (S-Vta10 vs V-vta10), so match case-insensitively.
    return {p.name[2:-4].lower(): p for p in (SYNC_DIR / sub).glob(f"{prefix}-*.csv")}


def drive_names() -> list[str]:
    s, v = _files("S-Dataset", "S"), _files("V-Dataset", "V")
    return sorted(s[k].name[2:-4] for k in s.keys() & v.keys())


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True, low_memory=False)
    if df.shape[1] != len(cols):
        raise ValueError(f"{path.name}: expected {len(cols)} columns, got {df.shape[1]}")
    df.columns = cols
    return df


def enu_from_latlon(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local tangent-plane East/North in metres (equirectangular; fine over tens of km)."""
    e = np.deg2rad(lon - lon0) * EARTH_R * np.cos(np.deg2rad(lat0))
    n = np.deg2rad(lat - lat0) * EARTH_R
    return e, n


PHONE_SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz", "wx", "wy", "wz", "mx", "my", "mz",
                 "o_azimuth", "o_pitch", "o_roll", "s_lat", "s_lon", "s_alt", "s_speed_ms", "s_acc_m",
                 "s_course_deg", "s_sats_used", "s_sats_visible"]
MAX_LAG = 1500      # ±150 s search window
CHUNK = 3000        # 5 min verification chunks


def _xcorr_best(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[int, float]:
    """Lag L maximising |corr(a[i+L], b[i])| (positive L: `a` runs late)."""
    from scipy.signal import correlate
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = correlate(a, b, mode="full", method="fft") / len(a)
    lags = np.arange(-len(b) + 1, len(a))
    m = np.abs(lags) <= maxlag
    i = int(np.argmax(np.abs(c[m])))
    return int(lags[m][i]), float(c[m][i])


def _corr_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag >= 0:
        x, y = a[lag:], b[: len(b) - lag]
    else:
        x, y = a[:lag], b[-lag:]
    if len(x) < 50 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _best_axis_lag(W: np.ndarray, yaw: np.ndarray, maxlag: int) -> tuple[int, float]:
    from scipy.ndimage import uniform_filter1d
    best = (0, 0.0)
    for k in range(3):
        lag, r = _xcorr_best(uniform_filter1d(W[:, k], 5), yaw, maxlag)
        if abs(r) > abs(best[1]):
            best = (lag, r)
    return best


def resync_phone(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Re-align phone signals to the vehicle clock.

    The "synchronised" IO-VNBD pairs are offset by 0.3 s to >100 s and some phone files are several recordings
    stitched together. Per phone segment (split at phone-clock resets) we find the lag that best matches phone gyro
    to the vehicle yaw rate, shift the phone signals, and mark 5-min chunks whose match stays weak as imu_valid=0.
    """
    from scipy.ndimage import uniform_filter1d
    n = len(df)
    t = df["s_t_ms"].to_numpy()
    breaks = np.where((np.diff(t) < 0) | (np.diff(t) > 1000))[0] + 1
    bounds = [0, *breaks.tolist(), n]
    out = {c: np.full(n, np.nan) for c in PHONE_SIGNALS}
    valid = np.zeros(n, dtype=np.float32)
    yaw = df["yaw_rate_true"].to_numpy()
    W_all = df[["wx", "wy", "wz"]].to_numpy()
    report = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if s1 - s0 < 600:
            continue
        # Segment lag: consensus of chunk-local lags (robust to straight stretches and bad chunks).
        votes = []
        for c0 in range(s0, s1, CHUNK):
            c1 = min(c0 + CHUNK, s1)
            if c1 - c0 >= 600 and yaw[c0:c1].std() > 0.02:
                votes.append(_best_axis_lag(W_all[c0:c1], yaw[c0:c1], min(MAX_LAG, c1 - c0 - 600)))
        good = [(l, abs(r)) for l, r in votes if abs(r) >= 0.2]
        if good:
            lags = np.array([l for l, _ in good])
            support = [sum(w for l2, w in good if abs(l2 - l) <= 3) for l in lags]
            lag = int(np.median(lags[np.abs(lags - lags[int(np.argmax(support))]) <= 3]))
        else:
            lag, _ = _best_axis_lag(W_all[s0:s1], yaw[s0:s1], min(MAX_LAG, (s1 - s0) // 3))
        # phone row (i + lag) belongs to vehicle row i
        src = np.arange(s0, s1) + lag
        ok = (src >= s0) & (src < s1)
        dst = np.arange(s0, s1)[ok]
        for c in PHONE_SIGNALS:
            out[c][dst] = df[c].to_numpy()[src[ok]]
        # Verify chunk-by-chunk after the shift: local best lag must be ~0.
        Wsh = np.stack([out["wx"], out["wy"], out["wz"]], 1)
        status = []  # 1 verified, 0 rejected, None unverifiable (no turning)
        for c0 in range(s0, s1, CHUNK):
            c1 = min(c0 + CHUNK, s1)
            m = ~np.isnan(Wsh[c0:c1, 0])
            if m.sum() < 300 or yaw[c0:c1][m].std() < 0.02:
                status.append(None)
                continue
            l_loc, r_loc = _best_axis_lag(Wsh[c0:c1][m], yaw[c0:c1][m], 300)
            status.append(1 if abs(l_loc) <= 3 and abs(r_loc) >= 0.2 else 0)
        checked = [x for x in status if x is not None]
        seg_ok = bool(checked) and np.mean(checked) >= 0.5
        for i, c0 in enumerate(range(s0, s1, CHUNK)):
            c1 = min(c0 + CHUNK, s1)
            if status[i] == 1 or (status[i] is None and seg_ok):
                valid[c0:c1] = (~np.isnan(Wsh[c0:c1, 0])).astype(np.float32)
        report.append({"start": s0, "end": s1, "lag": lag, "verified": sum(x == 1 for x in status),
                       "rejected": sum(x == 0 for x in status), "unverifiable": sum(x is None for x in status)})
    new = df.copy()
    for c in PHONE_SIGNALS:
        new[c] = out[c]
    new["imu_valid"] = valid
    return new, report


def load_pair(name: str) -> pd.DataFrame:
    """One cleaned, row-aligned DataFrame for a drive (vehicle ground truth + phone sensors)."""
    v = _read(_files("V-Dataset", "V")[name.lower()], V_COLS)
    s = _read(_files("S-Dataset", "S")[name.lower()], S_COLS)
    n = min(len(v), len(s))
    v, s = v.iloc[:n].reset_index(drop=True), s.iloc[:n].reset_index(drop=True)

    sats = s["s_sats"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)").astype(float)
    s = s.drop(columns=["s_sats", "s_date"]).apply(pd.to_numeric, errors="coerce")
    s["s_sats_used"], s["s_sats_visible"] = sats[0], sats[1]
    v = v.apply(pd.to_numeric, errors="coerce")
    v["sats"] = v.pop("sats_raw").fillna(0).astype(int) & 0x7F

    df = pd.concat([v, s], axis=1)
    df = df.interpolate(limit_direction="both")

    # Ground truth: speed from the vehicle speedometer, direction from GNSS course.
    df["v_true"] = (df["speed_kmh"] / 3.6).clip(lower=0)
    psi = np.deg2rad(df["heading_deg"].to_numpy())
    valid = (df["v_true"].to_numpy() > 0.5) & (df["sats"].to_numpy() >= 4)
    # Hold the last valid heading through stops so the direction target stays defined.
    psi_held = pd.Series(np.where(valid, psi, np.nan)).ffill().bfill().to_numpy()
    df["psi_true"] = psi_held
    df["heading_valid"] = valid.astype(np.float32)
    df["vE_true"] = df["v_true"] * np.sin(psi_held)
    df["vN_true"] = df["v_true"] * np.cos(psi_held)
    df["yaw_rate_true"] = np.deg2rad(df["yaw_rate_dps"])

    e, nn = enu_from_latlon(df["lat"].to_numpy(), df["lon"].to_numpy(), df["lat"].iloc[0], df["lon"].iloc[0])
    df["pE"], df["pN"] = e, nn
    df, report = resync_phone(df)
    df.attrs["sync"] = report
    df["drive"] = name
    return df


def load_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(exist_ok=True)
    out, sync = {}, {}
    for name in drive_names():
        cache = CACHE_DIR / f"{name}.parquet"
        if cache.exists() and not refresh:
            out[name] = pd.read_parquet(cache)
        else:
            df = load_pair(name)
            sync[name] = df.attrs.pop("sync")
            df.to_parquet(cache)
            out[name] = df
    if sync:
        import json
        (CACHE_DIR / "sync_report.json").write_text(json.dumps(sync, indent=1))
    return out


def moving_seconds(df: pd.DataFrame) -> float:
    return float((df["v_true"] > 0.5).sum() * DT)


def _angle_deg(a_e, a_n, b_e, b_n):
    return np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(a_e, a_n) - np.arctan2(b_e, b_n))))))


if __name__ == "__main__":
    drives = load_all(refresh=True)
    rows = []
    r_speed, ang = [], []
    for name, df in drives.items():
        km = df["v_true"].sum() * DT / 1000
        iv = df["imu_valid"] > 0
        rows.append((name, len(df), round(len(df) * DT / 3600, 2), round(km, 1), round(moving_seconds(df)),
                     round(iv.mean(), 2), round(df.loc[iv, "v_true"].sum() * DT / 1000, 1)))
        if iv.sum() > 100:
            r_speed.append(np.corrcoef(df.loc[iv, "s_speed_ms"] * 3.6, df.loc[iv, "speed_kmh"])[0, 1])
        # Velocity from heading vs from differentiated position (only while moving with a good fix).
        dE = np.gradient(df["pE"].to_numpy(), DT)
        dN = np.gradient(df["pN"].to_numpy(), DT)
        m = (df["v_true"] > 3).to_numpy() & (df["heading_valid"] > 0).to_numpy()
        if m.sum() > 100:
            ang.append(np.median(_angle_deg(df["vE_true"].to_numpy()[m], df["vN_true"].to_numpy()[m], dE[m], dN[m])))
    summ = pd.DataFrame(rows, columns=["drive", "rows", "hours", "km", "moving_s", "imu_ok", "km_ok"])
    print(summ.to_string(index=False))
    print(f"\npairs={len(summ)}  hours={summ.hours.sum():.1f}  km={summ.km.sum():.1f}  "
          f"usable after resync: km={summ.km_ok.sum():.1f}")
    import json
    rep = json.loads((CACHE_DIR / "sync_report.json").read_text())
    for name in ["M", "S1", "S4", "Vw4", "Y1", "Vtb5"]:
        print(name, rep.get(name))
    print(f"phone GPS speed (m/s ×3.6) vs vehicle km/h: median r = {np.nanmedian(r_speed):.3f}")
    print(f"heading-velocity vs position-derivative: median angle diff = {np.median(ang):.2f}°")
