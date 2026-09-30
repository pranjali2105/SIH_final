"""Differentiable dead-reckoning integrator shared by training, evaluation and the app spec.

Frame: heading-relative to the last GNSS heading psi0. x = forward along psi0, y = to the right (clockwise positive),
so a relative heading d_psi maps to direction (cos d_psi, sin d_psi). ENU heading psi is clockwise from North.

    d_psi_t = -sum_{k<=t} w_yaw_k * dt          (vehicle yaw rate is counter-clockwise positive)
    v_t     = relu(v0 + sum_{k<=t} a_fwd_k * dt) * (1 - p_stop_t)
    vel_t   = v_t * (cos d_psi_t, sin d_psi_t)   (non-holonomic: no lateral slip)
    p_t     = sum_{k<=t} vel_k * dt
"""
from __future__ import annotations

import math

import torch

DT = 0.1


def integrate(w_yaw: torch.Tensor, a_fwd: torch.Tensor, v0: torch.Tensor, p_stop: torch.Tensor | None = None,
              dt: float = DT) -> dict[str, torch.Tensor]:
    """w_yaw, a_fwd, p_stop: (B, T); v0: (B,). Returns relative heading, speed, velocity and position (B, T[, 2])."""
    d_psi = -torch.cumsum(w_yaw, dim=1) * dt
    speed = torch.relu(v0[:, None] + torch.cumsum(a_fwd, dim=1) * dt)
    if p_stop is not None:
        speed = speed * (1.0 - p_stop)
    vel = torch.stack([speed * torch.cos(d_psi), speed * torch.sin(d_psi)], dim=-1)
    pos = torch.cumsum(vel, dim=1) * dt
    return {"d_psi": d_psi, "speed": speed, "vel": vel, "pos": pos}


def to_enu(rel: torch.Tensor, psi0: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 2) heading-relative (forward, right) vectors back to (East, North)."""
    s, c = torch.sin(psi0), torch.cos(psi0)
    while s.dim() < rel.dim() - 1:
        s, c = s[..., None], c[..., None]
    fwd, right = rel[..., 0], rel[..., 1]
    return torch.stack([fwd * s + right * c, fwd * c - right * s], dim=-1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
