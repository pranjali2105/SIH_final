"""Non-autoregressive residual (sensor-error) model: dilated-CNN stem + compact Temporal Fusion Transformer.

    u_hat_t = C @ u_raw_t - (e_hat_t + b_hat)            u = [w_yaw, a_fwd, a_lat]

C (3x3, init I) is a learned static calibration (Brossard et al. 2020); e_hat is a per-step correction and b_hat a
per-sample bias, both zero-initialised so an untrained model reproduces the physics baseline exactly. All outage
steps are produced in one causal forward pass from sensor inputs + static context; no prediction is fed back.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from idr.dataset import CTX, HORIZON, N_SEQ, N_STATIC
from idr.route_features import N_PTS

# output scales: residual heads emit O(1) numbers, these map them to physical units
E_SCALE = torch.tensor([0.05, 0.5, 0.5])     # rad/s, m/s^2, m/s^2
N_STATES = 5


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network (TFT) with optional context."""

    def __init__(self, d_in, d_hidden, d_out, d_ctx=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.fc2(F.elu(h))
        return self.norm(self.skip(x) + self.glu(self.drop(h)))


class VSN(nn.Module):
    """Variable selection network over n_vars scalar inputs; returns (embedding, weights)."""

    def __init__(self, n_vars, d, d_ctx):
        super().__init__()
        self.emb_w = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.emb_b = nn.Parameter(torch.zeros(n_vars, d))
        self.select = GRN(n_vars * d, d, n_vars, d_ctx)
        self.var_w = nn.Parameter(torch.randn(n_vars, d, d) * (1.0 / d ** 0.5))

    def forward(self, x, c):
        e = x[..., None] * self.emb_w + self.emb_b                      # (..., V, d)
        w = torch.softmax(self.select(e.flatten(-2), c), dim=-1)        # (..., V)
        # per-variable linear map as one bmm over variables (same maths as einsum "...vd,vde->...ve"; bmm keeps memory
        # at O(B*L*V*d) and exports to a plain 3-D MatMul)
        B_, L_, V_, d_ = e.shape
        h = torch.bmm(e.permute(2, 0, 1, 3).reshape(V_, B_ * L_, d_), self.var_w)
        h = F.elu(h.reshape(V_, B_, L_, -1).permute(1, 2, 0, 3))
        return (w[..., None] * h).sum(-2), w


class CausalDilatedStem(nn.Module):
    """Brossard-style causal dilated CNN (kernel 5, dilations 1/3/9/27 -> ~13 s receptive field at 10 Hz)."""

    def __init__(self, d, dilations=(1, 3, 9, 27), k=5):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(d, d, k, dilation=dl) for dl in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in dilations)   # per-timestep: keeps the stem causal
        self.pads = [(k - 1) * dl for dl in dilations]

    def forward(self, x):                      # (B, L, d)
        h = x.transpose(1, 2)
        for conv, norm, pad in zip(self.convs, self.norms, self.pads):
            y = conv(F.pad(h, (pad, 0)))
            h = h + F.gelu(norm(y.transpose(1, 2)).transpose(1, 2))
        return h.transpose(1, 2)


class RouteEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 3, padding=1))
        self.pos = nn.Parameter(torch.randn(N_PTS, d) * 0.02)
        self.pool_q = nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, route):                  # (B, N_PTS, 4) -> per-point (B,N,d), pooled (B,d), mask (B,N)
        h = self.conv(route.transpose(1, 2)).transpose(1, 2) + self.pos
        valid = route[..., 3] > 0.5
        # never fully masked: force the first point valid (no in-place write: it is mis-traced by the ONNX exporter)
        valid = torch.cat([torch.ones_like(valid[:, :1]), valid[:, 1:]], dim=1)
        att = (h @ self.pool_q).masked_fill(~valid, -1e4)
        pooled = (torch.softmax(att, -1)[..., None] * h).sum(1)
        return h, pooled, valid


class ResidualTFT(nn.Module):
    def __init__(self, d=64, heads=4, dropout=0.1, use_tft=True):
        super().__init__()
        self.d, self.use_tft = d, use_tft
        self.C = nn.Parameter(torch.eye(3))
        self.route = RouteEncoder(d)
        self.static_grn = GRN(N_STATIC + d, d, d)
        self.ctx = nn.ModuleList(GRN(d, d, d) for _ in range(4))    # selection, enrichment, h0, c0
        self.vsn = VSN(N_SEQ, d, d)
        self.stem = CausalDilatedStem(d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = GLU(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, d_ctx=d, dropout=dropout)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.post_attn = GLU(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.route_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.post_route = GLU(d, d)
        self.norm3 = nn.LayerNorm(d)
        self.ff = GRN(d, d, d, dropout=dropout)
        self.head = nn.Linear(d, 4)            # 3 residual channels + stop logit
        self.cls_head = nn.Linear(d, N_STATES)  # roundabout state: straight/turn/entry/circulating/exit
        self.bias_head = nn.Linear(2 * d, 3)
        for lin in (self.head, self.bias_head):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.head.bias[3] = -4.0           # p_stop ~ 0 at init
        self.register_buffer("e_scale", E_SCALE.clone())
        L = CTX + HORIZON
        # additive float causal mask (0 / -inf); a boolean attn_mask is mis-exported to ONNX
        self.register_buffer("causal", torch.triu(torch.full((L, L), float("-inf")), 1), persistent=False)

    def forward(self, seq, static, route, u_raw):
        """seq (B,L,N_SEQ), static (B,N_STATIC), route (B,N_PTS,4), u_raw (B,H,3) -> dict of (B,H,...)"""
        B, L, _ = seq.shape
        r_pts, r_pool, r_valid = self.route(route)
        has_route = static[:, -1:]                                   # route_available flag
        s = self.static_grn(torch.cat([static, r_pool * has_route], -1))
        c_sel, c_enr, c_h, c_c = (g(s) for g in self.ctx)

        x, var_w = self.vsn(seq, c_sel[:, None, :].expand(B, L, -1))
        x = self.stem(x)
        if self.use_tft:
            h, _ = self.lstm(x, (c_h[None].contiguous(), c_c[None].contiguous()))
            h = self.norm1(x + self.post_lstm(h))
            e = self.enrich(h, c_enr[:, None, :].expand(B, L, -1))
            a, attn_w = self.attn(e, e, e, attn_mask=self.causal[:L, :L], need_weights=True)
            h = self.norm2(e + self.post_attn(a))
            ra, _ = self.route_attn(h, r_pts, r_pts, key_padding_mask=~r_valid)
            h = self.norm3(h + self.post_route(ra) * has_route[:, :, None])
            h = self.ff(h)
        else:
            h, attn_w = x, None

        out = self.head(h[:, CTX:])                                  # outage steps only
        state_logits = self.cls_head(h[:, CTX:])
        e_hat = out[..., :3] * self.e_scale
        stop_logit = out[..., 3]
        ctx_summary = h[:, :CTX].mean(1)
        b_hat = self.bias_head(torch.cat([ctx_summary, s], -1)) * self.e_scale
        u_cal = u_raw @ self.C.T
        u_hat = u_cal - e_hat - b_hat[:, None, :]
        return {"u_hat": u_hat, "e_hat": e_hat, "b_hat": b_hat, "stop_logit": stop_logit, "state_logits": state_logits,
                "p_stop": torch.sigmoid(stop_logit), "var_w": var_w, "attn_w": attn_w}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from idr import physics
    torch.manual_seed(0)
    m = ResidualTFT().eval()
    B = 2
    seq = torch.randn(B, CTX + HORIZON, N_SEQ)
    static = torch.randn(B, N_STATIC)
    static[:, -1] = 0.0
    route = torch.randn(B, N_PTS, 4)
    u_raw = torch.randn(B, HORIZON, 3)
    o = m(seq, static, route, u_raw)
    print("params:", count_params(m), " u_hat:", tuple(o["u_hat"].shape))
    # zero-init: corrected signal == raw signal -> integrated trajectory == physics baseline
    assert torch.allclose(o["u_hat"], u_raw, atol=1e-6)
    v0 = torch.tensor([10.0, 5.0])
    a = physics.integrate(o["u_hat"][..., 0], o["u_hat"][..., 1], v0)
    b = physics.integrate(u_raw[..., 0], u_raw[..., 1], v0)
    assert torch.allclose(a["pos"], b["pos"], atol=1e-4)
    # causality: perturb inputs after step k, outputs up to k unchanged (needs non-zero head)
    with torch.no_grad():
        nn.init.normal_(m.head.weight, std=0.1)
        nn.init.normal_(m.bias_head.weight, std=0.1)
    o1 = m(seq, static, route, u_raw)["e_hat"]
    seq2 = seq.clone()
    k = 300
    seq2[:, CTX + k + 1:] += torch.randn_like(seq2[:, CTX + k + 1:])
    o2 = m(seq2, static, route, u_raw)["e_hat"]
    assert torch.allclose(o1[:, :k + 1], o2[:, :k + 1], atol=1e-5), (o1[:, :k + 1] - o2[:, :k + 1]).abs().max()
    assert not torch.allclose(o1[:, k + 1:], o2[:, k + 1:])
    # no route -> output independent of the route tensor
    o3 = m(seq, static, torch.randn(B, N_PTS, 4), u_raw)["e_hat"]
    assert torch.allclose(o1, o3, atol=1e-5)
    print("model OK: zero-init == physics, causal, route ignored when unavailable")
