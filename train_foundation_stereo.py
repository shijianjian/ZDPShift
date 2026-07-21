"""Fine-tune FoundationStereo on rendered_datasets/ with the same recipe
used for SymRAFTStereo and IGEV-Stereo (multi-Δ + SceneFlow rehearsal +
per-Δ loss weighting).

DINOv2 ViT backbone is **frozen** (upstream `Feature.__init__` calls
`freeze_model(self.dino)`); only the CNN encoder, cost-volume stack,
hourglass aggregator, classifier, context net, GRU update, and upsampler
are trained.

Use `--signed-volume` to switch to the SignedFoundationStereo variant
(Path B in the paper); without it, vanilla FoundationStereo is fine-tuned
(Path A).
"""
import os
import argparse, os, sys, time
from pathlib import Path

# pandas must be imported before torch — FS's Utils.py imports pandas via
# `from Utils import *`, and importing it after torch triggers a libstdc++
# GLIBCXX_3.4.29 mismatch in this conda env.
import pandas  # noqa: F401

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FS_DIR = ROOT / "third_party/FoundationStereo"
sys.path.insert(0, str(FS_DIR))

from data.zdp_dataset import ZdpStereoDataset, make_val_split
from core.foundation_stereo import FoundationStereo
from models.signed_foundation_stereo import SignedFoundationStereo


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid,
                  sample_weights=None, loss_gamma=0.9, max_disp=700):
    """Sequence loss adapted for signed disparities + per-sample weights.

    Mirrors train_igev_stereo.py:sequence_loss for cross-comparison.
    """
    if disp_gt.dim() == 3:
        disp_gt = disp_gt.unsqueeze(1)
    n = len(disp_preds)
    mag = disp_gt.abs()
    valid_b = ((valid >= 0.5) & (mag < max_disp).squeeze(1)).unsqueeze(1) if valid.dim() == 3 else \
              ((valid >= 0.5) & (mag < max_disp))
    if sample_weights is not None:
        w = sample_weights.view(-1, 1, 1, 1).to(disp_gt.dtype)
    else:
        w = 1.0

    # FS returns init_disp at 1/4 resolution; downsample gt/valid/w to match
    # for the init-disp term. disp_preds[i] are already full resolution.
    if disp_init_pred.shape[-1] != disp_gt.shape[-1]:
        s = disp_gt.shape[-1] // disp_init_pred.shape[-1]
        gt_init = F.avg_pool2d(disp_gt.float(), s, stride=s) / s
        valid_init = F.max_pool2d(valid_b.float(), s, stride=s) >= 0.5
        w_init = w if isinstance(w, float) else w  # same per-sample weight
    else:
        gt_init, valid_init, w_init = disp_gt, valid_b, w
    init_err = (disp_init_pred - gt_init).abs() * w_init
    loss = (init_err * valid_init.float()).sum() / (valid_init.float().sum() + 1e-6)
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
    print(f"  val step={step} EPE={epe:.4f}", flush=True)


def load_model(args, cfg):
    if args.signed_volume:
        print(f"using SignedFoundationStereo: d_neg={args.d_neg}, d_pos={args.d_pos}",
              flush=True)
        return SignedFoundationStereo(cfg, d_neg=args.d_neg, d_pos=args.d_pos)
    return FoundationStereo(cfg)


def trainable_params(model):
    """Return only params with requires_grad=True; DINOv2 is already frozen
    by upstream `freeze_model(self.dino)`."""
    return [p for p in model.parameters() if p.requires_grad]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", required=True)
    p.add_argument("--ckpt-init", required=True,
                   help="Pretrained FS checkpoint .pth (cfg.yaml read from sibling dir)")
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
                   help="Use SignedFoundationStereo (Path B); default off = Path A")
    p.add_argument("--d-neg", type=int, default=64)
    p.add_argument("--d-pos", type=int, default=192)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- cfg ----
    cfg_path = Path(args.ckpt_init).parent / "cfg.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.setdefault("vit_size", "vitl")
    cfg.setdefault("hiera", 0)
    cfg.setdefault("low_memory", 0)
    cfg.setdefault("valid_iters", 32)
    cfg.mixed_precision = True
    print(f"cfg: vit_size={cfg.vit_size} max_disp={cfg.max_disp} "
          f"corr_levels={cfg.corr_levels} corr_radius={cfg.corr_radius}", flush=True)

    # ---- datasets ----
    full = ZdpStereoDataset(args.train_root, crop=tuple(args.crop), augment=True)
    n = len(full)
    val_ids = make_val_split(n, frac=0.05, seed=42)
    train_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                                 augment=True, val_indices=val_ids, mode="train")
    val_set = ZdpStereoDataset(args.train_root, crop=tuple(args.crop),
                               augment=False, val_indices=val_ids, mode="val")
    print(f"train={len(train_set)} val={len(val_set)} full={n}", flush=True)

    sf_loader = None
    half = args.batch
    if args.rehearsal_root:
        from data.sceneflow_dataset import SceneflowStereoDataset
        sf_set = SceneflowStereoDataset(args.rehearsal_root, subset="all",
                                        split="train", crop=tuple(args.crop),
                                        augment=True, pass_kind="finalpass")
        print(f"sceneflow rehearsal: {len(sf_set)} pairs", flush=True)
        half = max(1, args.batch // 2)
        sf_loader = DataLoader(sf_set, batch_size=half, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, collate_fn=_collate)
    train_loader = DataLoader(train_set, batch_size=half, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=half, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=_collate)

    # ---- model ----
    base_model = load_model(args, cfg)
    sd = torch.load(args.ckpt_init, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # Strip 'module.' prefix if present (FS ckpts are not DataParallel)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    miss, unexp = base_model.load_state_dict(sd, strict=False)
    print(f"loaded: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if miss[:3]:
        print(f"  first 3 missing: {miss[:3]}", flush=True)
    if unexp[:3]:
        print(f"  first 3 unexpected: {unexp[:3]}", flush=True)

    base_model = base_model.cuda()
    base_model.train()
    # Keep DINOv2 in eval mode (BN/dropout); upstream Feature.forward also
    # asserts this every call, but be explicit.
    base_model.feature.dino.eval()

    n_total = sum(p.numel() for p in base_model.parameters())
    n_train = sum(p.numel() for p in trainable_params(base_model))
    print(f"params: total={n_total/1e6:.1f}M trainable={n_train/1e6:.1f}M "
          f"(frozen={(n_total-n_train)/1e6:.1f}M)", flush=True)

    opt = torch.optim.AdamW(trainable_params(base_model), lr=args.lr, weight_decay=args.wd)
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
            disp_init, disp_preds = base_model(L, R, iters=args.train_iters)
            loss = sequence_loss(disp_preds, disp_init, disp_gt, valid,
                                 sample_weights=sample_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params(base_model), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.iters} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={dt:.0f}s", flush=True)

        if (step % args.ckpt_every == 0 and step > 0) or step == args.iters - 1:
            ck = out / f"ckpt_{step:06d}.pth"
            torch.save(base_model.state_dict(), ck)
            torch.save(base_model.state_dict(), out / "latest.pth")
            print(f"  saved {ck}", flush=True)

        if step % args.val_every == 0 and step > 0:
            _run_val(base_model, val_loader, args.train_iters, step, out)

        step += 1
    print("training done", flush=True)


if __name__ == "__main__":
    main()
