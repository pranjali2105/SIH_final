"""Per-outage calibration modes for GNSS-outage evaluation.

    causal_pre_outage  (DEFAULT) phone->vehicle axes, gyro bias, forward-accel bias and magnetometer hard iron are
                       fitted ONLY on the same recording segment's samples with timestamp < outage_start - guard_s.
    offline_oracle     legacy behaviour: one calibration per recording segment fitted on the WHOLE segment,
                       including the outage itself. Upper-bound / debug only; must be requested explicitly.

Why: idr.normalise.calibrate() regresses the phone axes on GNSS speed derivative, GNSS-derived centripetal
acceleration and stationary periods detected from GNSS speed. Fitted over the whole recording, that puts GNSS from
inside each evaluated outage into the "IMU-only" inputs.

What a causal view changes: for one outage starting at row t0, rows [t0 - CTX, t0 + HORIZON) of the calibrated IMU
arrays (raw_u, standardised feats, psi_mag) are recomputed with the prefix calibration. All other rows of the view
are NaN, so anything that silently reads outside the window fails loudly instead of using oracle values.
Normalisation statistics are the global training-drive statistics (no evaluation-drive data at all).

Quality (decided from the prefix only):
    orientation   estimated       >= min_moving_full moving GNSS samples (v > 1 m/s) in the prefix
                  estimated_weak  >= min_moving_partial
                  prior           fewer -> training-mean axes (documented UNRELIABLE: forward axis median 47 deg off
                                  across training mounts), outage marked cold_start and reported separately
    gyro bias     stationary      >= min_still samples inside GNSS-detected stops (calibrate() rule)
                  median          no stop but >= min_moving_partial prefix samples (median gyro, calibrate() rule)
                  prior           otherwise: zero bias
    accel bias    follows the forward-axis fit (intercept of the same regression): estimated / estimated_weak / prior
    calibration_quality = full (orientation estimated AND gyro stationary) | partial (orientation estimated or
                          estimated_weak) | cold_start (orientation prior)
There is NO fallback to oracle calibration.

Assumptions kept as dataset preprocessing (flagged, not changed here): the phone/vehicle clock re-sync and its
weak-match imu_valid masking (idr.data.resync_phone) use the whole recording; the outage set is therefore identical
in both modes.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from idr.data import DT
from idr.normalise import ARTIFACTS, STAT_COLS, _segments, apply_calibration, calibrate, standardise

CAL_MODES = ("causal_pre_outage", "offline_oracle")
DEFAULT_MODE = "causal_pre_outage"
GNSS_COLS = ("v_true", "yaw_rate_true", "psi_true", "heading_valid", "vE_true", "vN_true", "pE", "pN", "lat", "lon",
             "s_speed_ms", "s_course_deg")
ORACLE_WARNING = ("*** offline_oracle calibration: phone calibration is fitted on whole recordings INCLUDING the "
                  "GNSS inside each outage. Results are an optimistic upper bound, not IMU-only performance. ***")


@dataclass(frozen=True)
class CalibrationConfig:
    mode: str = DEFAULT_MODE
    guard_s: float = 0.5
    min_moving_full: int = 600          # 60 s moving
    min_moving_partial: int = 300       # 30 s moving
    min_still: int = 20                 # calibrate(): stationary gyro-bias needs > 20 still samples
    min_prefix_up: int = 10             # samples to measure gravity direction from the prefix IMU
    inflate_partial: float = 1.5        # EKF input-noise multiplier (sig_w, sig_a) for partial outages
    inflate_cold: float = 2.0           # ... for cold_start outages

    def __post_init__(self):
        check_mode(self.mode)

    @property
    def guard_steps(self) -> int:
        return int(np.ceil(self.guard_s / DT - 1e-9))


def check_mode(mode: str) -> str:
    if mode not in CAL_MODES:
        raise ValueError(f"calibration mode must be one of {CAL_MODES}, got {mode!r}")
    return mode


def warn_if_oracle(mode: str) -> None:
    if check_mode(mode) == "offline_oracle":
        print("\n" + ORACLE_WARNING + "\n", flush=True)
        warnings.warn(ORACLE_WARNING, stacklevel=2)


def add_cli(ap) -> None:
    ap.add_argument("--calibration_mode", default=DEFAULT_MODE, choices=CAL_MODES,
                    help="causal_pre_outage (default) or offline_oracle (explicit upper-bound/debug only)")
    ap.add_argument("--calibration_guard_s", type=float, default=0.5)


def config_from_args(args) -> CalibrationConfig:
    warn_if_oracle(args.calibration_mode)
    return CalibrationConfig(mode=args.calibration_mode, guard_s=args.calibration_guard_s)


# ---------------------------------------------------------------- training prior (documented, cold_start only)
_PRIOR = None


def training_prior() -> dict:
    """Mean training-segment calibration. Used ONLY for cold_start outages (reported separately): the forward axis
    of training mounts is median 47 deg (p90 131 deg) from this mean, so it is not a usable estimate."""
    global _PRIOR
    if _PRIOR is not None:
        return _PRIOR
    path = ARTIFACTS / "calibration_prior.json"
    if path.exists():
        _PRIOR = json.loads(path.read_text())
        return _PRIOR
    from idr.data import load_all
    from idr.dataset import SCENARIO_TEST, TEST, VAL
    raw = load_all()
    cals = []
    for n, df in raw.items():
        if n in TEST + VAL + SCENARIO_TEST:
            continue
        for a, b in _segments(df):
            cals.append(calibrate(df.iloc[a:b]))
    keys = ["u_yaw", "e_roll", "e_pitch", "up", "c_fwd", "c_lat", "mag_hard_iron"]
    _PRIOR = {k: np.mean([c[k] for c in cals], axis=0).tolist() for k in keys}
    _PRIOR["gyro_bias"] = [0.0, 0.0, 0.0]
    _PRIOR["n_training_segments"] = len(cals)
    path.write_text(json.dumps(_PRIOR, indent=1))
    return _PRIOR


# ---------------------------------------------------------------- calibration
def segment_of(segments, t0):
    for a, b in segments:
        if a <= t0 < b:
            return a, b
    raise ValueError(f"outage start {t0} is not inside a valid IMU segment")


def calibrate_prefix(prefix: pd.DataFrame, cfg: CalibrationConfig) -> tuple[dict, dict]:
    """Calibration from a pre-outage prefix only (+ statuses). Never reads anything but `prefix`."""
    n = len(prefix)
    v = prefix["v_true"].to_numpy() if n else np.zeros(0)
    gnss_ok = np.isfinite(v)
    n_gnss = int(gnss_ok.sum())
    moving = gnss_ok & (v > 1.0)
    still = (uniform_filter1d((np.nan_to_num(v, nan=1e9) < 0.1).astype(float), 20) > 0.99) if n else np.zeros(0, bool)
    n_moving, n_still = int(moving.sum()), int(still.sum())
    if n_moving >= cfg.min_moving_partial:
        cal = calibrate(prefix)                                     # identical maths to the oracle fit
        orient = "estimated" if n_moving >= cfg.min_moving_full else "estimated_weak"
        gyro = "stationary" if n_still > cfg.min_still else "median"
        accel = orient
    else:
        prior = training_prior()
        cal = {k: list(prior[k]) for k in ("u_yaw", "e_roll", "e_pitch", "up", "c_fwd", "c_lat", "mag_hard_iron",
                                           "gyro_bias")}
        if n >= cfg.min_prefix_up:                                  # IMU-only: gravity direction and hard iron
            A = prefix[["ax", "ay", "az"]].to_numpy()
            up = A.mean(0)
            cal["up"] = (up / np.linalg.norm(up)).tolist()
            cal["mag_hard_iron"] = prefix[["mx", "my", "mz"]].to_numpy().mean(0).tolist()
        if n_still > cfg.min_still:
            W = prefix[["wx", "wy", "wz"]].to_numpy()
            cal["gyro_bias"] = W[still].mean(0).tolist()
            gyro = "stationary"
        else:
            gyro = "prior"
        orient, accel = "prior", "prior"
    quality = ("full" if orient == "estimated" and gyro == "stationary"
               else "partial" if orient in ("estimated", "estimated_weak") else "cold_start")
    meta = {"number_of_pre_outage_GNSS_samples": n_gnss, "number_of_pre_outage_moving_samples": n_moving,
            "number_of_pre_outage_stop_samples": n_still, "orientation_calibration_status": orient,
            "gyro_bias_calibration_status": gyro, "accel_bias_calibration_status": accel,
            "calibration_quality": quality}
    return cal, meta


def outage_calibration(raw_df: pd.DataFrame, t0: int, cfg: CalibrationConfig, segments=None) -> tuple[dict, dict]:
    """Calibration for the outage starting at row t0, under cfg.mode, plus its metadata."""
    segments = segments if segments is not None else _segments(raw_df)
    a, b = segment_of(segments, t0)
    if cfg.mode == "offline_oracle":
        cal = calibrate(raw_df.iloc[a:b])
        meta = {"calibration_mode": "offline_oracle", "outage_start_time_s": t0 * DT,
                "calibration_end_time_s": (b - 1) * DT, "prefix_duration_s": (b - a) * DT,
                "uses_samples_after_outage_start": True, "calibration_quality": "oracle",
                "orientation_calibration_status": "oracle", "gyro_bias_calibration_status": "oracle",
                "accel_bias_calibration_status": "oracle"}
        return cal, meta
    end = t0 - cfg.guard_steps                                      # exclusive: time(end-1) < t0*DT - guard_s
    prefix = raw_df.iloc[a:max(a, end)]
    cal, meta = calibrate_prefix(prefix, cfg)
    meta.update({"calibration_mode": "causal_pre_outage", "outage_start_time_s": t0 * DT,
                 "calibration_end_time_s": (end - 1) * DT if end > a else None, "guard_s": cfg.guard_s,
                 "prefix_duration_s": max(0, end - a) * DT, "uses_samples_after_outage_start": False})
    return cal, meta


# ---------------------------------------------------------------- per-outage drive views
_STATS = None


def norm_stats() -> dict:
    global _STATS
    if _STATS is None:
        _STATS = json.loads((ARTIFACTS / "norm_stats.json").read_text())
    return _STATS


def causal_view(d, raw_df: pd.DataFrame, t0: int, cfg: CalibrationConfig, segments=None, ctx=None, horizon=None):
    """Drive copy for ONE outage: calibrated IMU arrays recomputed in rows [t0 - ctx, t0 + horizon) with the
    outage's calibration; NaN elsewhere. Ground-truth arrays (v, psi, pos, u_true, ...) are shared, unchanged."""
    from idr.dataset import CTX, HORIZON
    ctx = CTX if ctx is None else ctx
    horizon = HORIZON if horizon is None else horizon
    cal, meta = outage_calibration(raw_df, t0, cfg, segments)
    lo, hi = t0 - ctx, min(t0 + horizon, int(d.run_end[t0]))
    win = apply_calibration(raw_df.iloc[lo:hi], cal)
    n = len(d.v)
    raw_u = np.full((n, 3), np.nan, dtype=np.float32)
    raw_u[lo:hi] = np.nan_to_num(win[["w_yaw", "a_fwd", "a_lat"]].to_numpy(dtype=np.float64)).astype(np.float32)
    feats = np.full((n, len(STAT_COLS)), np.nan, dtype=np.float32)
    feats[lo:hi] = standardise(np.nan_to_num(win[STAT_COLS].to_numpy(dtype=np.float32)), norm_stats()).astype(np.float32)
    psi_mag = np.full(n, np.nan)
    psi_mag[lo:hi] = np.nan_to_num(win["psi_mag_raw"].to_numpy())
    meta = {**meta, "window_rows": [int(lo), int(hi)],
            "norm_stats": "global training-drive statistics (artifacts/norm_stats.json); no evaluation-drive data"}
    return dataclasses.replace(d, name=f"{d.name}@{t0}", raw_u=raw_u, feats=feats, psi_mag=psi_mag), meta


def build_views(drives, specs, cfg: CalibrationConfig, raw=None):
    """Views + remapped specs for a list of outage specs (causal mode). Returns (view_drives, view_specs, metas)."""
    from idr.data import load_all
    raw = raw if raw is not None else load_all()
    seg_cache, views, vspecs, metas = {}, {}, [], []
    for sp in specs:
        name, t0 = sp["drive"], int(sp["t0"])
        if name not in seg_cache:
            seg_cache[name] = _segments(raw[name])
        v, meta = causal_view(drives[name], raw[name], t0, cfg, seg_cache[name])
        views[v.name] = v
        vspecs.append({**sp, "drive": v.name})
        metas.append(meta)
    return views, vspecs, metas


def oracle_meta(drives, specs, raw=None):
    """Metadata for offline_oracle outputs (the arrays themselves come from the cached whole-segment calibration)."""
    from idr.data import load_all
    raw = raw if raw is not None else load_all()
    seg_cache, out = {}, []
    for sp in specs:
        name, t0 = sp["drive"], int(sp["t0"])
        if name not in seg_cache:
            seg_cache[name] = _segments(raw[name])
        a, b = segment_of(seg_cache[name], t0)
        out.append({"calibration_mode": "offline_oracle", "outage_start_time_s": t0 * DT,
                    "calibration_end_time_s": (b - 1) * DT, "prefix_duration_s": (b - a) * DT,
                    "uses_samples_after_outage_start": True, "calibration_quality": "oracle",
                    "orientation_calibration_status": "oracle", "gyro_bias_calibration_status": "oracle",
                    "accel_bias_calibration_status": "oracle"})
    return out


def nav_cfg_for(meta: dict, base_cfg, cfg: CalibrationConfig):
    """EKF input-noise inflation for weaker calibrations (config only; EKF equations unchanged)."""
    q = meta.get("calibration_quality")
    k = cfg.inflate_partial if q == "partial" else cfg.inflate_cold if q == "cold_start" else 1.0
    if k == 1.0:
        return base_cfg
    return dataclasses.replace(base_cfg, sig_w=base_cfg.sig_w * k, sig_a=base_cfg.sig_a * k)


def window_vib(view, t0, horizon):
    """Vibration feature for the EKF from the view's window (same formula as evaluate_mm's oracle path)."""
    iv = STAT_COLS.index("a_up")
    lo, hi = view.feats.shape[0], 0
    rows = np.flatnonzero(np.isfinite(view.feats[:, iv]))
    lo, hi = rows[0], rows[-1] + 1
    v = np.clip(uniform_filter1d(view.feats[lo:hi, iv] ** 2, 10) ** 0.5 - 1.0, 0, 3)
    out = np.zeros(horizon)
    seg = v[t0 - lo:t0 - lo + horizon]
    out[:len(seg)] = seg
    return out
