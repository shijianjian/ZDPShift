"""Fine-tune SymRAFTStereo on multi_dataset_open_movie/."""
import argparse, sys, time, types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/RAFT-Stereo"))
sys.path.insert(0, str(ROOT / "third_party/RAFT-Stereo/core"))

from data.zdp_dataset import ZdpStereoDataset, make_val_split
from raft_stereo_sym import SymRAFTStereo


def model_args():
    a = types.SimpleNamespace()
    a.hidden_dims = [128, 128, 128]
    a.corr_implementation = "reg"
    a.shared_backbone = False
    a.corr_levels = 4
    a.corr_radius = 4
    a.n_downsample = 2
    a.context_norm = "batch"
    a.slow_fast_gru = False
    a.n_gru_layers = 3
    a.mixed_precision = True
    return a


def sequence_loss(disp_preds, disp_gt, valid, sample_weights=None,
                  gamma=0.8, max_disp=700):
    """Standard RAFT multi-scale L1 disparity loss with optional per-sample weights.

    RAFT-Stereo emits flow[:, 0] = -disp, so loss compares (flow[:, 0] + disp_gt).
    `valid` must be a bool tensor (we combine with another bool mask via &).
    `disp_preds[i]` has shape [B, 1, H, W]; we take channel 0.
    `sample_weights` (optional): shape [B] float; multiplies the per-sample mask.
    """
    n = len(disp_preds)
    loss = 0.0
    valid_combined = valid & (disp_gt.abs() < max_disp)
    valid_f = valid_combined.float()
    if sample_weights is not None:
        # Broadcast [B] -> [B, H, W]
        w = sample_weights.view(-1, 1, 1).to(valid_f)
        valid_f = valid_f * w
    for i, dp in enumerate(disp_preds):
        wt = gamma ** (n - i - 1)
        loss = loss + wt * (valid_f * (dp[:, 0] + disp_gt).abs()).mean()
    return loss


def _sample_weights_from_meta(metas, easy_w):
    """Up-weight Δ ∈ {-16, 0} and sceneflow samples (positive-disp rehearsal)."""
    w = torch.ones(len(metas))
    for i, m in enumerate(metas):
        d = int(m.get("delta_px", 0))
        if m.get("source") == "sceneflow" or d in (-16, 0):
            w[i] = easy_w
    return w


def _collate(samples):
    Ls, Rs, ds, metas = zip(*samples)
    L = torch.stack(Ls)
    R = torch.stack(Rs)
    disp = torch.stack(ds)
    return L, R, disp, list(metas)


def _run_val(model, loader, gru_iters, step, out):
    model.eval()
    epes = []
    with torch.no_grad():
        for batch in loader:
            L, R, disp_gt = (b.cuda(non_blocking=True) for b in batch[:3])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(L, R, iters=gru_iters, test_mode=True)
            # test_mode returns (coords1-coords0, flow_up); take flow_up = preds[1]
            disp_pred = -preds[1][:, 0]
            valid = disp_gt.abs() < 1000
            e = (disp_pred - disp_gt).abs()[valid]
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
    p.add_argument("--ckpt-init", required=True,
                   help="Sceneflow .pth to initialize from")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, nargs=2, default=[384, 512])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--gru-iters", type=int, default=12)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--rehearsal-root", default=None,
                   help="Optional Sceneflow root (e.g. datasets/sceneflow/). "
                        "When set, training batches are 50/50 mixed from the "
                        "multi-ZDP train set and Sceneflow finalpass, to retain "
                        "the model's positive-disparity competence.")
    p.add_argument("--easy-disp-weight", type=float, default=1.5,
                   help="Per-sample loss multiplier for Δ in {-16, 0} and for "
                        "Sceneflow samples (positive disp only). Default 1.5 "
                        "biases gradient slightly toward the easy positive case "
                        "so fine-tuning doesn't forget it.")
    p.add_argument("--freeze-fnet", action="store_true",
                   help="Freeze the feature encoder (correlation features) so "
                        "pretrained matching is preserved; only context+GRU adapt.")
    p.add_argument("--keep-deltas", type=str, default=None,
                   help="Ablation −R2: comma-separated list of Δ values to keep "
                        "from the ZDPShift train split, e.g. '0' to drop ZDP-shift "
                        "augmentation. Default: keep all 5 shifts.")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Sample-efficiency study: fraction of scenes to use for "
                        "training (e.g., 0.25, 0.50, 0.75, 1.00).")
    args = p.parse_args()

    keep = None
    if args.keep_deltas:
        keep = {int(s.strip()) for s in args.keep_deltas.split(',')}
        print(f"keep_deltas filter: {sorted(keep)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build dataset, split val
    full = ZdpStereoDataset(args.train_root, crop=tuple(args.crop), augment=True,
                             keep_deltas=keep, train_fraction=args.train_fraction)
    n = len(full)
    val_ids = make_val_split(n, frac=0.1, seed=42)
    train_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                                 augment=True, val_indices=val_ids, mode="train",
                                 keep_deltas=keep, train_fraction=args.train_fraction)
    val_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                               augment=False, val_indices=val_ids, mode="val",
                               keep_deltas=keep, train_fraction=args.train_fraction)
    print(f"train={len(train_set)} val={len(val_set)} full={n} (frac 10%)")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=_collate)

    sf_loader = None
    if args.rehearsal_root:
        from data.sceneflow_dataset import SceneflowStereoDataset
        sf_set = SceneflowStereoDataset(args.rehearsal_root, subset="all",
                                        split="train",
                                        crop=tuple(args.crop), augment=True,
                                        pass_kind="finalpass")
        print(f"sceneflow rehearsal: {len(sf_set)} pairs")
        half = max(1, args.batch // 2)
        train_loader = DataLoader(train_set, batch_size=half, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  collate_fn=_collate)
        sf_loader = DataLoader(sf_set, batch_size=half, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True,
                               collate_fn=_collate)

    # Build model + load ckpt
    model = torch.nn.DataParallel(SymRAFTStereo(model_args()))
    sd = torch.load(args.ckpt_init, map_location="cpu")
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("loaded:", "missing", len(miss), "unexpected", len(unexp))
    model = model.cuda().train()

    if getattr(args, "freeze_fnet", False):
        n_frozen = 0
        for name, p in model.named_parameters():
            if ".fnet." in name or name.startswith("module.fnet"):
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"froze fnet: {n_frozen/1e6:.2f}M params frozen")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters,
        pct_start=0.05, anneal_strategy="linear",
    )

    def _next_or_restart(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            it = iter(loader)
            return next(it), it

    step = 0
    t0 = time.time()
    zdp_iter = iter(train_loader)
    sf_iter = iter(sf_loader) if sf_loader is not None else None
    while step < args.iters:
        (zL, zR, zD, zM), zdp_iter = _next_or_restart(zdp_iter, train_loader)
        if sf_iter is not None:
            (sL, sR, sD, sM), sf_iter = _next_or_restart(sf_iter, sf_loader)
            L = torch.cat([zL, sL], dim=0)
            R = torch.cat([zR, sR], dim=0)
            disp_gt = torch.cat([zD, sD], dim=0)
            metas = list(zM) + list(sM)
        else:
            L, R, disp_gt, metas = zL, zR, zD, list(zM)
        L = L.cuda(non_blocking=True)
        R = R.cuda(non_blocking=True)
        disp_gt = disp_gt.cuda(non_blocking=True)
        # Pass bool valid; sequence_loss combines it with another bool via &.
        valid = disp_gt.abs() < 1000
        sample_w = _sample_weights_from_meta(metas, args.easy_disp_weight).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(L, R, iters=args.gru_iters)
            loss = sequence_loss(preds, disp_gt, valid, sample_weights=sample_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.iters} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={dt:.0f}s",
                  flush=True)

        if (step % args.ckpt_every == 0 and step > 0) or step == args.iters - 1:
            ck = out / f"ckpt_{step:06d}.pth"
            torch.save(model.state_dict(), ck)
            torch.save(model.state_dict(), out / "latest.pth")
            print(f"  saved {ck}")

        if step % args.val_every == 0 and step > 0:
            _run_val(model, val_loader, args.gru_iters, step, out)

        step += 1

    print("training done")


if __name__ == "__main__":
    main()
