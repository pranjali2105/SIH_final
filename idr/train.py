"""Train the residual TFT. `python -m idr.train --epochs 40` (smoke test: `--epochs 1 --samples 640 --val 256`)."""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from idr import physics
from idr.dataset import OutageDataset, attach_labels, load_drives, residual_std, split_names
from idr.losses import angle_weights, compute_losses, cos_angle, true_velocity
from idr.model import ResidualTFT, count_params
from idr.normalise import ARTIFACTS


def pick_device() -> torch.device:
    import os
    if os.environ.get("IDR_DEVICE"):
        return torch.device(os.environ["IDR_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def rollout_metrics(model, loader, dev, baseline=False) -> dict:
    """Per-sample drift at the end of each outage (and at 30 s), mean angle error. baseline=True: raw physics."""
    if model is not None:
        model.eval()
    finals, drifts, d30, ang = [], [], [], []
    for batch in loader:
        b = to_dev(batch, dev)
        if baseline:
            u, p_stop = b["u_raw"], None
        else:
            out = model(b["seq"], b["static"], b["route"], b["u_raw"])
            u, p_stop = out["u_hat"], out["p_stop"]
        traj = physics.integrate(u[..., 0], u[..., 1], b["v0"], p_stop)
        err = (traj["pos"] - b["pos_rel"]).norm(dim=-1)
        last = b["mask"].sum(1).long() - 1
        ar = torch.arange(len(last), device=dev)
        fe, dist = err[ar, last], b["dist"][ar, last]
        finals.append(fe.cpu())
        drifts.append(torch.where(dist > 50, fe / dist.clamp(min=1e-3) * 100, torch.nan).cpu())
        i30 = torch.clamp(last, max=299)
        e30, di30 = err[ar, i30], b["dist"][ar, i30]
        d30.append(torch.where((last >= 299) & (di30 > 50), e30 / di30.clamp(min=1e-3) * 100, torch.nan).cpu())
        aw = angle_weights(b["v_true"], b["hvalid"], b["mask"])
        c = cos_angle(traj["vel"], true_velocity(b)).clamp(-1, 1)
        deg = torch.rad2deg(torch.acos(c))
        ang.append(((deg * aw).sum(1) / aw.sum(1).clamp(min=1e-3)).cpu())
    drifts, d30 = torch.cat(drifts).numpy(), torch.cat(d30).numpy()
    return {
        "median_drift_pct": float(np.nanmedian(drifts)), "p90_drift_pct": float(np.nanpercentile(drifts, 90)),
        "median_drift30_pct": float(np.nanmedian(d30)),
        "median_final_err_m": float(torch.cat(finals).median()), "mean_angle_deg": float(torch.cat(ang).mean()),
        "n": int(np.isfinite(drifts).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--no_tft", action="store_true", help="Brossard-only ablation: CNN stem + residual head")
    ap.add_argument("--tag", default="tft")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val_drives", nargs="*", default=None, help="override validation drives (checkpoint selection)")
    ap.add_argument("--loss_weights", default="{}", help='JSON overrides of losses.DEFAULT_WEIGHTS, e.g. {"win": 1.0}')
    ap.add_argument("--decel_weight", type=float, default=1.0, help="x weight on braking/post-turn/roundabout speed loss")
    ap.add_argument("--rb_frac", type=float, default=0.3, help="share of training outages drawn around roundabouts")
    ap.add_argument("--init", default=None, help="warm-start from artifacts/<init>.pt (e.g. after an interrupted run)")
    args = ap.parse_args()

    dev = pick_device()
    torch.manual_seed(0)
    drives, stats = load_drives()
    sp = split_names(list(drives))
    if args.val_drives:
        sp["train"] = [n for n in sp["train"] + sp["val"] if n not in args.val_drives]
        sp["val"] = list(args.val_drives)
    loss_w = json.loads(args.loss_weights)
    rs = residual_std(drives, sp["train"])
    from idr.roundabout_labels import STATES, all_labels
    labels = all_labels(drives, list(drives))
    attach_labels(drives, labels)
    cnt = np.zeros(len(STATES))
    for n in sp["train"]:
        if drives[n].states is not None:
            st = drives[n].states[drives[n].valid]
            cnt += np.bincount(st, minlength=len(STATES))
    class_w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(STATES), dtype=torch.float32).clamp(max=50.0)
    print("state label counts (train):", dict(zip(STATES, cnt.astype(int).tolist())),
          " drives with roundabout starts:", sum(1 for n in sp["train"] if drives[n].rb_starts is not None
                                                   and len(drives[n].rb_starts)))
    res_std = torch.tensor(rs, dtype=torch.float32)
    train_ds = OutageDataset(drives, sp["train"], args.samples, mode="train", residual_std=rs)
    train_ds._stats = stats
    train_ds.rb_frac = args.rb_frac
    train_ds.decel_weight = args.decel_weight
    val_ds = OutageDataset(drives, sp["val"], args.val, mode="val", seed=123, residual_std=rs)
    dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers, drop_last=True)
    vdl = DataLoader(val_ds, batch_size=64, num_workers=0)

    model = ResidualTFT(d=args.d, use_tft=not args.no_tft).to(dev)
    if args.init:
        init = torch.load(ARTIFACTS / f"{args.init}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"], strict=False)
        print(f"warm start from {args.init}.pt (val median drift {init['val']['median_drift_pct']:.2f}%)")
    print(f"device={dev} params={count_params(model)} train_drives={len(sp['train'])} val_drives={sp['val']}")
    base = rollout_metrics(None, vdl, dev, baseline=True)
    print("physics baseline (val):", json.dumps({k: round(v, 3) for k, v in base.items()}))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    best, bad, log = math.inf, 0, []
    for ep in range(args.epochs):
        model.train()
        t0, agg, nb = time.time(), {}, 0
        for batch in dl:
            b = to_dev(batch, dev)
            out = model(b["seq"], b["static"], b["route"], b["u_raw"])
            loss, parts, _ = compute_losses(out, b, res_std, weights=loss_w, class_weights=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        vm = rollout_metrics(model, vdl, dev)
        rec = {"epoch": ep + 1, "time_s": round(time.time() - t0, 1), **{k: round(v / nb, 4) for k, v in agg.items()},
               **{f"val_{k}": round(v, 3) for k, v in vm.items()}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vm["median_drift_pct"] < best:
            best, bad = vm["median_drift_pct"], 0
            torch.save({"model": model.state_dict(), "args": vars(args), "res_std": rs.tolist(), "val": vm,
                        "baseline_val": base}, ARTIFACTS / f"{args.tag}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    (ARTIFACTS / f"{args.tag}_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val median drift {best:.2f}% (baseline {base['median_drift_pct']:.2f}%)")


if __name__ == "__main__":
    main()
