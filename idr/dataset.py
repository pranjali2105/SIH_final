"""Simulated GNSS-outage samples for the residual (sensor-error) model.

Each sample = 5 s of context with GNSS + up to 60 s of outage. Every call draws a fresh outage start t0, a random
destination further along the drive, and route/no-route/wrong-route conditions, so one drive yields many passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import Dataset

from idr import route_features as rf
from idr.data import DT
from idr.normalise import ARTIFACTS, STAT_COLS, fit_stats, load_aligned, standardise

CTX = 50          # 5 s context with GNSS
HORIZON = 600     # 60 s outage
MIN_H = 100       # shortest usable outage (10 s)
SPEED_SCALE = 30.0

TEST = ["M", "Vw4", "Y1"]      # Y1 kept so results compare with the MARK-V benchmark table
SCENARIO_TEST = ["Vw16b", "Vw6", "Vw7", "Vw8"]   # hard brake / successive turns (Onyekpe et al. 2021 test sets)
VAL = ["S2", "Vtb5", "S3c", "Vta1a"]   # enlarged: 2 drives were too few for reliable checkpoint selection

SEQ_FEATURES = STAT_COLS + ["sin_dpsi_gyro", "cos_dpsi_gyro", "sin_dpsi_mag", "cos_dpsi_mag", "tau",
                            "gnss_flag", "gnss_speed", "sin_dpsi_gnss", "cos_dpsi_gnss"]
N_SEQ = len(SEQ_FEATURES)
N_STATIC = 5      # v0/30 + 4 destination features


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Drive:
    name: str
    raw_u: np.ndarray        # (N,3) aligned phone [w_yaw, a_fwd, a_lat]
    feats: np.ndarray        # (N,8) standardised STAT_COLS
    psi_mag: np.ndarray
    valid: np.ndarray        # imu_valid
    v: np.ndarray
    psi: np.ndarray
    hvalid: np.ndarray
    pos: np.ndarray          # (N,2) ENU metres
    u_true: np.ndarray       # (N,3) [yaw_ref, dv/dt, v*yaw_ref]
    s: np.ndarray            # arc length per row (m)
    grid_b: np.ndarray       # bearings of the whole drive resampled every 20 m
    starts: np.ndarray       # admissible outage starts (index of first outage step)
    run_end: np.ndarray      # for each row, end (exclusive) of the contiguous valid run it belongs to
    states: np.ndarray | None = None    # roundabout-state labels (roundabout_labels.STATES), -1 = unlabelled
    rb_starts: np.ndarray | None = None  # outage starts whose 60 s window contains roundabout entry/circ/exit


def _yaw_reference(df: pd.DataFrame) -> np.ndarray:
    """CAN yaw rate re-scaled per drive against the GNSS heading rate (CAN yaw alone drifts ~3 deg/min)."""
    yaw = df["yaw_rate_true"].to_numpy()
    psi = df["psi_true"].to_numpy()
    rate = -np.gradient(np.unwrap(psi), DT)          # counter-clockwise positive, like the CAN yaw
    m = (df["v_true"].to_numpy() > 3) & (df["heading_valid"].to_numpy() > 0)
    if m.sum() < 300:
        return yaw
    ys, rs = uniform_filter1d(yaw, 10), uniform_filter1d(rate, 10)
    X = np.column_stack([ys[m], np.ones(m.sum())])
    a, b = np.linalg.lstsq(X, rs[m], rcond=None)[0]
    ref = a * yaw + b if 0.8 < a < 1.2 else yaw
    # While moving, GNSS course rate is the better truth (CAN yaw drifts ~4 deg/min when integrated);
    # at low speed GNSS course is noise, so fall back to the rescaled CAN yaw there.
    gnss_ok = uniform_filter1d(m.astype(float), 5) > 0.99
    return np.where(gnss_ok, uniform_filter1d(rate, 5), ref)


def build_drive(name: str, df: pd.DataFrame, stats: dict) -> Drive | None:
    valid = df["imu_valid"].to_numpy().astype(bool)
    if valid.sum() < CTX + MIN_H:
        return None
    raw = df[["w_yaw", "a_fwd", "a_lat"]].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw)
    feats = standardise(np.nan_to_num(df[STAT_COLS].to_numpy(dtype=np.float32)), stats).astype(np.float32)
    v = df["v_true"].to_numpy()
    yaw_ref = _yaw_reference(df)
    dvdt = np.gradient(uniform_filter1d(v, 5), DT)
    u_true = np.column_stack([yaw_ref, dvdt, v * yaw_ref])
    pos = df[["pE", "pN"]].to_numpy()
    step = np.r_[0.0, np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))]
    step[(v < 0.3) | (step > 10)] = 0.0
    s = np.cumsum(step)
    e, n, _ = rf.resample_polyline(pos[:, 0], pos[:, 1])
    grid_b = rf.polyline_bearings(e, n)
    # contiguous valid runs
    run_end = np.zeros(len(df), dtype=np.int64)
    end = len(df)
    for i in range(len(df) - 1, -1, -1):
        if not valid[i]:
            end = i
        run_end[i] = end
    run_start = np.zeros(len(df), dtype=np.int64)
    st = 0
    for i in range(len(df)):
        if not valid[i]:
            st = i + 1
        run_start[i] = st
    idx = np.arange(len(df))
    ok = valid & (idx - run_start >= CTX) & (run_end - idx >= MIN_H)
    return Drive(name, raw.astype(np.float32), feats, np.nan_to_num(df["psi_mag_raw"].to_numpy()), valid, v,
                 df["psi_true"].to_numpy(), df["heading_valid"].to_numpy(), pos, u_true.astype(np.float32), s,
                 grid_b, np.flatnonzero(ok), run_end)


def load_drives(names: list[str] | None = None, stats: dict | None = None) -> tuple[dict[str, Drive], dict]:
    aligned = load_aligned()
    if stats is None:
        train_names = [n for n in aligned if n not in TEST + VAL + SCENARIO_TEST]
        stats = fit_stats(aligned, train_names)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "norm_stats.json").write_text(json.dumps(stats, indent=1))
    out = {}
    for n, df in aligned.items():
        if names is not None and n not in names:
            continue
        d = build_drive(n, df, stats)
        if d is not None and len(d.starts):
            out[n] = d
    return out, stats


def split_names(all_names) -> dict[str, list[str]]:
    return {
        "test": [n for n in TEST if n in all_names],
        "scenario": [n for n in SCENARIO_TEST if n in all_names],
        "val": [n for n in VAL if n in all_names],
        "train": [n for n in all_names if n not in TEST + VAL + SCENARIO_TEST],
    }


class OutageDataset(Dataset):
    """mode='train' draws fresh random samples; otherwise a fixed, seeded list of samples."""

    def __init__(self, drives: dict[str, Drive], names: list[str], n_samples: int, mode: str = "train",
                 seed: int = 0, route_mode: str | None = None, donors: dict[str, Drive] | None = None,
                 residual_std: np.ndarray | None = None):
        self.drives = {n: drives[n] for n in names if n in drives}
        self.names = list(self.drives)
        # weight drives by number of admissible starts so long drives are sampled proportionally
        w = np.array([len(self.drives[n].starts) for n in self.names], dtype=float)
        self.p = w / w.sum()
        self.n = n_samples
        self.mode = mode
        self.route_mode = route_mode      # None=random mix, or 'none' | 'dest' | 'route' | 'wrong'
        self.donors = [d for d in (donors or self.drives).values() if len(d.grid_b) > 120]
        self.residual_std = residual_std if residual_std is not None else np.array([0.02, 0.3, 0.3])
        self.fixed = None
        self.rb_frac = 0.0
        if mode != "train":
            rng = np.random.default_rng(seed)
            self.fixed = [self._draw_spec(rng) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    def _draw_spec(self, rng) -> dict:
        rb_names = [n for n in self.names if self.drives[n].rb_starts is not None and len(self.drives[n].rb_starts)]
        if self.mode == "train" and rb_names and rng.random() < self.rb_frac:
            # balanced sampling: oversample outages that contain a roundabout manoeuvre
            name = rb_names[rng.integers(len(rb_names))]
            d = self.drives[name]
            pool = d.rb_starts
        else:
            name = self.names[rng.choice(len(self.names), p=self.p)]
            d = self.drives[name]
            moving = d.starts[d.v[d.starts - 1] > 1.0]
            pool = moving if (len(moving) and rng.random() < 0.85) else d.starts
        t0 = int(pool[rng.integers(len(pool))])
        end = len(d.v) - 1
        if rng.random() < 0.7 or t0 + 50 >= end:
            td = int(rng.integers(min(t0 + 50, end), end + 1))
        else:
            td = int(rng.integers(t0 + 50, min(end, t0 + 700) + 1))
        r = rng.random()
        mode = self.route_mode or ("none" if r < 0.30 else "wrong" if r < 0.45 else "route")
        return {"drive": name, "t0": t0, "td": td, "route_mode": mode, "seed": int(rng.integers(1 << 31)),
                "h_long": int(rng.integers(MIN_H, HORIZON + 1))}

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.mode == "train" else None
        spec = self._draw_spec(rng) if self.fixed is None else self.fixed[i]
        return self.make_sample(spec, augment=self.mode == "train")

    decel_weight = 3.0

    def _decel_weights(self, d, t0, hi):
        """Per-step weight for speed losses: x decel_weight on braking, the 10 s after a turn and roundabout
        entry/circulation (where validation slicing found the largest positive speed bias)."""
        w = np.ones(HORIZON, dtype=np.float32)
        n = hi - t0
        dvdt = d.u_true[t0:hi, 1]
        turn = np.abs(d.u_true[max(0, t0 - 100):hi, 0]) > 0.1
        post = np.convolve(turn.astype(float), np.ones(101), mode="full")[:len(turn)][-n:] > 0
        hot = (dvdt < -0.5) | post
        if d.states is not None:
            st = d.states[t0:hi]
            hot |= (st == 2) | (st == 3)
        w[:n][hot] = self.decel_weight
        return w

    @staticmethod
    def _states(d, t0, hi):
        out = np.full(HORIZON, -1, dtype=np.int64)
        if d.states is not None:
            out[:hi - t0] = d.states[t0:hi]
        return out

    def make_sample(self, spec: dict, augment: bool = False) -> dict:
        rng = np.random.default_rng(spec["seed"])
        d = self.drives[spec["drive"]] if spec["drive"] in self.drives else None
        t0 = spec["t0"]
        lo, hi = t0 - CTX, min(t0 + HORIZON, d.run_end[t0])
        H = hi - t0
        L = CTX + HORIZON
        sl = slice(lo, hi)
        n = hi - lo

        raw = d.raw_u[sl].copy()
        feats = d.feats[sl].copy()
        if augment:
            noise = rng.normal(0, 0.5, size=raw.shape) * self.residual_std[None, :]
            raw += noise.astype(np.float32)
            # keep standardised copies of the same three channels consistent with the noisy raw signal
            for k, col in enumerate(["w_yaw", "a_fwd", "a_lat"]):
                j = STAT_COLS.index(col)
                sd = self._stats[col]["std"] if hasattr(self, "_stats") else 1.0
                feats[:, j] += (noise[:, k] / sd).astype(np.float32)

        psi0 = float(d.psi[t0 - 1])
        v0 = float(d.v[t0 - 1])
        p0 = d.pos[t0 - 1]

        # heading-change features, all relative to t0 (never to the model's own outputs)
        cs = np.cumsum(raw[:, 0].astype(np.float64)) * DT
        dpsi_gyro = -(cs - cs[CTX - 1])
        mag = d.psi_mag[sl]
        mag0 = np.angle(np.mean(np.exp(1j * mag[CTX - 10:CTX])))
        dpsi_mag = wrap(mag - mag0)
        tau = (np.arange(n) - CTX + 1) * DT / 60.0
        gnss = np.zeros(n)
        gnss[:CTX] = 1.0
        on = gnss > 0                       # GNSS samples after outage start are never read (np.where, not x*0)
        dpsi_gnss = np.where(on, wrap(np.where(on, d.psi[sl], psi0) - psi0), 0.0)

        seq = np.zeros((L, N_SEQ), dtype=np.float32)
        seq[:n, :8] = feats
        seq[:n, 8], seq[:n, 9] = np.sin(dpsi_gyro), np.cos(dpsi_gyro)
        seq[:n, 10], seq[:n, 11] = np.sin(dpsi_mag), np.cos(dpsi_mag)
        seq[:n, 12] = tau
        seq[:n, 13] = gnss
        seq[:n, 14] = np.where(on, np.where(on, d.v[sl], 0.0) / SPEED_SCALE, 0.0)
        seq[:n, 15], seq[:n, 16] = np.where(on, np.sin(dpsi_gnss), 0.0), np.where(on, np.cos(dpsi_gnss), 0.0)

        # outage-step raw signals + targets (length HORIZON, masked beyond H)
        out = slice(t0, hi)
        u_raw = np.zeros((HORIZON, 3), dtype=np.float32)
        u_raw[:H] = raw[CTX:]
        u_true = np.zeros((HORIZON, 3), dtype=np.float32)
        u_true[:H] = d.u_true[out]
        mask = np.zeros(HORIZON, dtype=np.float32)
        mask[:H] = 1.0
        v_true = np.zeros(HORIZON, dtype=np.float32)
        v_true[:H] = d.v[out]
        dpsi_true = np.zeros(HORIZON, dtype=np.float32)
        dpsi_true[:H] = wrap(d.psi[out] - psi0)
        hvalid = np.zeros(HORIZON, dtype=np.float32)
        hvalid[:H] = d.hvalid[out]
        rel = d.pos[out] - p0
        f = np.array([np.sin(psi0), np.cos(psi0)])
        r = np.array([np.cos(psi0), -np.sin(psi0)])
        pos_rel = np.zeros((HORIZON, 2), dtype=np.float32)
        pos_rel[:H, 0], pos_rel[:H, 1] = rel @ f, rel @ r
        dist = np.zeros(HORIZON, dtype=np.float32)
        dist[:H] = np.cumsum(d.v[out]) * DT
        long_mask = mask.copy()
        long_mask[spec["h_long"]:] = 0.0

        # destination + route
        td = spec["td"]
        mode = spec["route_mode"]
        route = np.zeros((rf.N_PTS, 4), dtype=np.float32)
        has_route = mode in ("route", "wrong")
        if has_route:
            offset = rng.uniform(-30, 30) if augment or self.mode != "train" else 0.0
            if "route_b" in spec:
                # externally planned route (idr.route_input, e.g. OSM shortest path from the last fix to the
                # destination); same start jitter, clipped at the route start; never reads the true track
                rb_ = np.asarray(spec["route_b"], dtype=float)
                k0 = int(np.clip(round(offset / rf.STEP_M), 0, max(len(rb_) - 1, 0)))
                b = rb_[k0:k0 + rf.N_PTS].copy()
                k0, kd = 0, int(spec["route_n_valid"]) - k0 - 1
            else:
                k0 = int(np.clip(round((d.s[t0 - 1] + offset) / rf.STEP_M), 0, len(d.grid_b) - 1))
                kd = int(d.s[td] // rf.STEP_M)
                b = d.grid_b[k0:k0 + rf.N_PTS].copy()
            if mode == "wrong" and "route_b" not in spec and len(b) > 10 and self.donors:
                donor = self.donors[rng.integers(len(self.donors))]
                j = int(rng.integers(0, len(donor.grid_b) - rf.N_PTS))
                db = np.diff(donor.grid_b[j:j + rf.N_PTS + 1])
                kf = int(rng.integers(5, min(60, len(b) - 1)))
                b[kf:] = b[kf] + np.cumsum(wrap(db[kf:kf + len(b) - kf]))
            k = np.arange(len(b))
            b = b + np.radians(rng.normal(0, 2.0, len(b)) + 3.0 * np.sin(2 * np.pi * k / rng.uniform(20, 60)
                                                                            + rng.uniform(0, 2 * np.pi)))
            route = rf.route_tensor(b, psi0, max(1, kd - k0 + 1))
        dest = rf.destination_features(p0, d.pos[td], psi0, has_route)
        if mode == "none":
            dest[3] = 0.0
        static = np.concatenate([[v0 / SPEED_SCALE], dest]).astype(np.float32)

        return {
            "seq": torch.from_numpy(seq), "static": torch.from_numpy(static), "route": torch.from_numpy(route),
            "u_raw": torch.from_numpy(u_raw), "u_true": torch.from_numpy(u_true), "mask": torch.from_numpy(mask),
            "long_mask": torch.from_numpy(long_mask), "v0": torch.tensor(v0, dtype=torch.float32),
            "psi0": torch.tensor(psi0, dtype=torch.float32), "v_true": torch.from_numpy(v_true),
            "dpsi_true": torch.from_numpy(dpsi_true), "hvalid": torch.from_numpy(hvalid),
            "pos_rel": torch.from_numpy(pos_rel), "dist": torch.from_numpy(dist),
            "state": torch.from_numpy(self._states(d, t0, hi)),
            "w_speed": torch.from_numpy(self._decel_weights(d, t0, hi)),
        }


def attach_labels(drives: dict[str, "Drive"], labels: dict) -> None:
    """Attach roundabout-state labels and roundabout-rich outage starts to each drive (from roundabout_labels)."""
    for n, d in drives.items():
        if n not in labels:
            continue
        d.states = labels[n]["states"].astype(np.int64)
        rb = (d.states >= 2).astype(np.int64)
        c = np.concatenate([[0], np.cumsum(rb)])
        ends = np.minimum(d.starts + HORIZON, len(d.v))
        has = (c[ends] - c[d.starts]) > 0
        d.rb_starts = d.starts[has]


def residual_std(drives: dict[str, Drive], names: list[str]) -> np.ndarray:
    e = np.concatenate([drives[n].raw_u[drives[n].valid] - drives[n].u_true[drives[n].valid] for n in names])
    return e.std(0)


if __name__ == "__main__":
    drives, stats = load_drives()
    sp = split_names(list(drives))
    for k, v in sp.items():
        km = sum(drives[n].v[drives[n].valid].sum() * DT / 1000 for n in v)
        print(f"{k:9s} drives={len(v):3d}  usable km={km:7.1f}")
    rs = residual_std(drives, sp["train"])
    raw_sd = np.concatenate([drives[n].raw_u[drives[n].valid] for n in sp["train"]]).std(0)
    print("residual std [w_yaw rad/s, a_fwd, a_lat]:", rs.round(4), " raw signal std:", raw_sd.round(4))

    # CAN-yaw reference check: 60 s integrated heading vs GNSS heading on training drives
    errs = []
    for n in sp["train"][:15]:
        d = drives[n]
        for t0 in d.starts[::300]:
            if d.run_end[t0] - t0 < HORIZON or d.v[t0] < 3 or d.v[t0 + HORIZON - 1] < 3:
                continue
            dp_ref = -d.u_true[t0:t0 + HORIZON, 0].sum() * DT
            dp_gnss = wrap(d.psi[t0 + HORIZON - 1] - d.psi[t0 - 1])
            errs.append(abs(np.degrees(wrap(dp_gnss - dp_ref))))
    print(f"yaw target vs GNSS heading over 60 s: median {np.median(errs):.2f} deg, p90 {np.percentile(errs, 90):.2f} deg (n={len(errs)})")

    ds = OutageDataset(drives, sp["train"], 1000, mode="val", seed=1, residual_std=rs)
    tds = [s["td"] > s["t0"] for s in ds.fixed]
    assert all(tds)
    ds2 = OutageDataset(drives, sp["train"], 1000, mode="val", seed=1, residual_std=rs)
    assert ds.fixed == ds2.fixed
    x = ds[0]
    print({k: tuple(v.shape) for k, v in x.items()})
    print("dataset OK")
