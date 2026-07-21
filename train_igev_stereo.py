"""Fine-tune IGEV-Stereo on rendered_datasets/ with the same recipe used
for SymRAFTStereo (rehearsal + per-Δ weighting).

E2 cross-architecture validation: demonstrate the dataset + recipe transfer
to a categorically different stereo architecture (3D cost volume + iterative
geometry encoding, vs. RAFT's 2D correlation + GRU).
"""
import os
import argparse, sys, time, types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

IGEV_ROOT = ROOT / "third_party/IGEV/IGEV-Stereo"
sys.path.insert(0, str(IGEV_ROOT))
sys.path.insert(0, str(IGEV_ROOT / "core"))

from data.zdp_dataset import ZdpStereoDataset, make_val_split
from core.igev_stereo import IGEVStereo
from models.signed_igev_stereo import SignedIGEVStereo


def model_args():
    a = types.SimpleNamespace()
    a.hidden_dims = [128, 128, 128]
    a.n_downsample = 2
    a.n_gru_layers = 3
    a.corr_levels = 2
    a.corr_radius = 4
    a.max_disp = 192
    a.mixed_precision = True
    a.precision_dtype = "float16"
    return a


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid,
                  sample_weights=None, loss_gamma=0.9, max_disp=700):
    """IGEV-Stereo's multi-scale loss adapted to support:
       - negative disparities (max_disp on |disp|, default 700 px)
       - per-sample weights for easy-disp upweighting

    disp_gt expected shape: [B, 1, H, W] (or [B, H, W], we unsqueeze).
    disp_init_pred / disp_preds[i]: [B, 1, H, W].
    """
    if disp_gt.dim() == 3:
        disp_gt = disp_gt.unsqueeze(1)
    n = len(disp_preds)
    mag = disp_gt.abs()
    valid_b = ((valid >= 0.5) & (mag < max_disp).squeeze(1)).unsqueeze(1) if valid.dim() == 3 else \
              ((valid >= 0.5) & (mag < max_disp))
    if sample_weights is not None:
        # broadcast [B] -> [B, 1, H, W]
        w = sample_weights.view(-1, 1, 1, 1).to(disp_gt.dtype)
    else:
        w = 1.0

    # Initial prediction term
    init_err = (disp_init_pred - disp_gt).abs() * w
    loss = (init_err * valid_b.float()).sum() / (valid_b.float().sum() + 1e-6)
    # Sequence multi-scale
    adj_gamma = loss_gamma ** (15 / max(1, n - 1))
    for i, dp in enumerate(disp_preds):
        wt = adj_gamma ** (n - i - 1)
        err = (dp - disp_gt).abs() * w
        loss = loss + wt * (err * valid_b.float()).sum() / (valid_b.float().sum() + 1e-6)
    return loss


def _sample_weights_from_meta(metas, easy_w):
    w = torch.ones(len(metas))
    for i, m in enumerate(metas):
        d = int(m.get("delta_px", 0))
        if m.get("source") == "sceneflow" or d in (-16, 0):
            w[i] = easy_w
    return w


def _collate(samples):
    Ls, Rs, ds, metas = zip(*samples)
    return torch.stack(Ls), torch.stack(Rs), torch.stack(ds), list(metas)


def _run_val(model, loader, iters, step, out):
    model.eval()
    epes = []
    with torch.no_grad():
        for batch in loader:
            L, R, disp_gt = (b.cuda(non_blocking=True) for b in batch[:3])
            with torch.amp.autocast("cuda", dtype=torch.float16):
                disp = model(L, R, iters=iters, test_mode=True)
            disp = disp.squeeze(1) if disp.dim() == 4 else disp
            valid = disp_gt.abs() < 1000
            e = (disp - disp_gt).abs()[valid]
            if e.numel() > 0:
                epes.append(e.mean().item())
    model.train()
    epe = sum(epes) / max(1, len(epes))
    with open(out / "val.log", "a") as f:
        f.write(f"{step}\t{epe:.4f}\n")
    print(f"  val step={step} EPE={epe:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", required=True)
    p.add_argument("--ckpt-init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--crop", type=int, nargs=2, default=[320, 480])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--train-iters", type=int, default=12)
    p.add_argument("--ckpt-every", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--rehearsal-root", default=None)
    p.add_argument("--easy-disp-weight", type=float, default=1.5)
    p.add_argument("--signed-volume", action="store_true",
                   help="Use signed GWC cost volume (Path B; default False = Path A)")
    p.add_argument("--d-neg", type=int, default=64,
                   help="Negative-disparity extent in full-resolution pixels (signed volume only)")
    p.add_argument("--d-pos", type=int, default=192,
                   help="Positive-disparity extent in full-resolution pixels (signed volume only)")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    full = ZdpStereoDataset(args.train_root, crop=tuple(args.crop), augment=True)
    n = len(full)
    val_ids = make_val_split(n, frac=0.05, seed=42)  # smaller val for speed
    train_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                                 augment=True, val_indices=val_ids, mode="train")
    val_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                               augment=False, val_indices=val_ids, mode="val")
    print(f"train={len(train_set)} val={len(val_set)} full={n}")

    sf_loader = None
    half = args.batch
    if args.rehearsal_root:
        from data.sceneflow_dataset import SceneflowStereoDataset
        sf_set = SceneflowStereoDataset(args.rehearsal_root, subset="all",
                                        split="train",
                                        crop=tuple(args.crop), augment=True,
                                        pass_kind="finalpass")
        print(f"sceneflow rehearsal: {len(sf_set)} pairs")
        half = max(1, args.batch // 2)
        sf_loader = DataLoader(sf_set, batch_size=half, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, collate_fn=_collate)
    train_loader = DataLoader(train_set, batch_size=half, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=half, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=_collate)

    # ---- model ----
    if args.signed_volume:
        print(f"using SignedIGEVStereo: d_neg={args.d_neg}, d_pos={args.d_pos}")
        base_model = SignedIGEVStereo(model_args(), d_neg=args.d_neg, d_pos=args.d_pos)
    else:
        base_model = IGEVStereo(model_args())
    model = torch.nn.DataParallel(base_model)
    sd = torch.load(args.ckpt_init, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # Normalise "module." prefix between checkpoint and current model.
    has_module = next(iter(sd.keys())).startswith("module.")
    if not has_module:
        sd = {f"module.{k}": v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded: missing={len(miss)} unexpected={len(unexp)}")
    model = model.cuda().train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters,
        pct_start=0.05, anneal_strategy="linear",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def _next_or_restart(it, loader):
        try: return next(it), it
        except StopIteration:
            new_it = iter(loader); return next(new_it), new_it

    step = 0; t0 = time.time()
    zdp_iter = iter(train_loader)
    sf_iter = iter(sf_loader) if sf_loader is not None else None
    while step < args.iters:
        (zL, zR, zD, zM), zdp_iter = _next_or_restart(zdp_iter, train_loader)
        if sf_iter is not None:
            (sL, sR, sD, sM), sf_iter = _next_or_restart(sf_iter, sf_loader)
            L = torch.cat([zL, sL], dim=0); R = torch.cat([zR, sR], dim=0)
            disp_gt = torch.cat([zD, sD], dim=0); metas = list(zM) + list(sM)
        else:
            L, R, disp_gt, metas = zL, zR, zD, list(zM)
        L = L.cuda(non_blocking=True); R = R.cuda(non_blocking=True)
        disp_gt = disp_gt.cuda(non_blocking=True)
        valid = disp_gt.abs() < 1000
        sample_w = _sample_weights_from_meta(metas, args.easy_disp_weight).cuda()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            disp_init, disp_preds = model(L, R, iters=args.train_iters)
            loss = sequence_loss(disp_preds, disp_init, disp_gt, valid,
                                 sample_weights=sample_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.iters} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={dt:.0f}s", flush=True)

        if (step % args.ckpt_every == 0 and step > 0) or step == args.iters - 1:
            ck = out / f"ckpt_{step:06d}.pth"
            torch.save(model.state_dict(), ck)
            torch.save(model.state_dict(), out / "latest.pth")
            print(f"  saved {ck}")

        if step % args.val_every == 0 and step > 0:
            _run_val(model, val_loader, args.train_iters, step, out)

        step += 1
    print("training done")


if __name__ == "__main__":
    main()
