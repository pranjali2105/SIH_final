"""Residual supervision + trajectory losses through the differentiable integrator."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from idr import physics

DEFAULT_WEIGHTS = {"res": 1.0, "inc": 1.0, "angle": 0.5, "mag": 0.5, "pos": 0.25, "stop": 0.2, "smooth": 0.01, "win": 0.0,
                   "cls": 0.2}


def angle_weights(v_true, hvalid, mask):
    """Angle is undefined when stopped: weight ramps in with speed (0 at 0 m/s, 1 above 2 m/s)."""
    return hvalid * torch.clamp(v_true / 2.0, max=1.0) * mask


def cos_angle(a, b, eps=1e-6):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def true_velocity(batch):
    v, dp = batch["v_true"], batch["dpsi_true"]
    return torch.stack([v * torch.cos(dp), v * torch.sin(dp)], -1)


def increment_loss(traj, batch, js=(16, 32)):
    """Brossard-style loss on integrated heading/speed increments over short non-overlapping windows."""
    B, H = batch["mask"].shape
    total = 0.0
    zeros = torch.zeros(B, 1, device=batch["mask"].device)
    dpsi_hat = torch.cat([zeros, traj["d_psi"]], 1)
    dpsi_true = torch.cat([zeros, batch["dpsi_true"]], 1)
    v_hat = torch.cat([batch["v0"][:, None], traj["speed_raw"]], 1)
    v_true = torch.cat([batch["v0"][:, None], batch["v_true"]], 1)
    hv = torch.cat([torch.ones_like(zeros), batch["hvalid"]], 1)
    m = torch.cat([torch.ones_like(zeros), batch["mask"]], 1)
    for j in js:
        idx0 = torch.arange(0, H - j + 1, j, device=m.device)
        idx1 = idx0 + j
        d_hat = dpsi_hat[:, idx1] - dpsi_hat[:, idx0]
        d_true = torch.remainder(dpsi_true[:, idx1] - dpsi_true[:, idx0] + math.pi, 2 * math.pi) - math.pi
        ok_h = m[:, idx1] * hv[:, idx1] * hv[:, idx0] * (v_true[:, idx0] > 2).float()
        ok_v = m[:, idx1]
        lh = F.huber_loss(d_hat / 0.02, d_true / 0.02, reduction="none", delta=1.0)
        lv = F.huber_loss((v_hat[:, idx1] - v_hat[:, idx0]) / 0.5, (v_true[:, idx1] - v_true[:, idx0]) / 0.5,
                          reduction="none", delta=1.0)
        if "w_speed" in batch:
            ws = torch.cat([torch.ones_like(zeros), batch["w_speed"]], 1)
            lv = lv * torch.maximum(ws[:, idx0], ws[:, idx1])
        total = total + (lh * ok_h).sum() / ok_h.sum().clamp(min=1) + (lv * ok_v).sum() / ok_v.sum().clamp(min=1)
    return total / len(js)


def compute_losses(out, batch, res_std, weights=None, class_weights=None):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    mask, lmask = batch["mask"], batch["long_mask"]
    u_hat = out["u_hat"]
    traj = physics.integrate(u_hat[..., 0], u_hat[..., 1], batch["v0"], out["p_stop"])
    traj["speed_raw"] = torch.relu(batch["v0"][:, None] + torch.cumsum(u_hat[..., 1], 1) * physics.DT)

    sd = res_std.to(u_hat.device)
    l_res = (F.huber_loss(u_hat / sd, batch["u_true"] / sd, reduction="none", delta=1.0).mean(-1) * mask).sum() \
        / mask.sum().clamp(min=1)
    l_inc = increment_loss(traj, batch)

    v_true_vec = true_velocity(batch)
    aw = angle_weights(batch["v_true"], batch["hvalid"], lmask)
    l_angle = ((1 - cos_angle(traj["vel"], v_true_vec)) * aw).sum() / aw.sum().clamp(min=1)
    wmag = lmask * batch["w_speed"] if "w_speed" in batch else lmask
    l_mag = (F.huber_loss(traj["speed"], batch["v_true"], reduction="none", delta=1.0) * wmag).sum() \
        / wmag.sum().clamp(min=1)
    # relative distance error over non-overlapping 10 s and 30 s windows (the quantity along-track drift integrates)
    l_win = u_hat.new_zeros(())
    if w["win"] > 0:
        terms = []
        for j in (100, 300):
            H_ = mask.shape[1] // j * j
            sh = (traj["speed"][:, :H_] * mask[:, :H_]).reshape(len(mask), -1, j).sum(-1) * physics.DT
            st = (batch["v_true"][:, :H_] * mask[:, :H_]).reshape(len(mask), -1, j).sum(-1) * physics.DT
            full = mask[:, :H_].reshape(len(mask), -1, j).min(-1).values
            rel = (sh - st).abs() / st.clamp(min=10.0)
            terms.append((rel * full).sum() / full.sum().clamp(min=1))
        l_win = sum(terms) / len(terms)
    rel_err = (traj["pos"] - batch["pos_rel"]).norm(dim=-1) / batch["dist"].clamp(min=10.0)
    l_pos = (rel_err * lmask).sum() / lmask.sum().clamp(min=1)
    stop_target = (batch["v_true"] < 0.1).float()
    l_stop = (F.binary_cross_entropy_with_logits(out["stop_logit"], stop_target, reduction="none") * mask).sum() \
        / mask.sum().clamp(min=1)
    e = out["e_hat"] / out["e_hat"].new_tensor([0.05, 0.5, 0.5])
    l_smooth = ((e[:, 1:] - e[:, :-1]) ** 2).mean()

    l_cls = u_hat.new_zeros(())
    if "state" in batch and "state_logits" in out and (batch["state"] >= 0).any():
        cw = None if class_weights is None else class_weights.to(u_hat.device)
        l_cls = F.cross_entropy(out["state_logits"].reshape(-1, out["state_logits"].shape[-1]),
                                batch["state"].reshape(-1), weight=cw, ignore_index=-1)
    parts = {"res": l_res, "inc": l_inc, "angle": l_angle, "mag": l_mag, "pos": l_pos, "stop": l_stop,
             "smooth": l_smooth, "cls": l_cls, "win": l_win}
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}, traj


if __name__ == "__main__":
    # angle term: parallel 0, perpendicular 1, opposite 2; rotation-invariant
    a = torch.tensor([[1.0, 0.0]])
    for b, want in [([2.0, 0.0], 0.0), ([0.0, 3.0], 1.0), ([-1.0, 0.0], 2.0)]:
        got = float(1 - cos_angle(a, torch.tensor([b])))
        assert abs(got - want) < 1e-5, (b, got)
    rot = lambda v, t: torch.stack([v[..., 0] * math.cos(t) - v[..., 1] * math.sin(t),
                                    v[..., 0] * math.sin(t) + v[..., 1] * math.cos(t)], -1)
    x, y = torch.randn(10, 2), torch.randn(10, 2)
    assert torch.allclose(cos_angle(x, y), cos_angle(rot(x, 0.7), rot(y, 0.7)), atol=1e-5)
    wts = angle_weights(torch.tensor([0.0, 1.0, 5.0]), torch.ones(3), torch.ones(3))
    assert wts.tolist() == [0.0, 0.5, 1.0]
    print("losses OK: angle 0/1/2, rotation-invariant, zero weight when stationary")
