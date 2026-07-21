"""Evaluate a fine-tuned FoundationStereo (Path A) or SignedFoundationStereo
(Path B) across the multi-Delta test set.

Mirrors eval_igev_signed.py output format (per_pair.csv + summary.csv) so the
result slots into the Table 4 aggregation.
"""
from __future__ import annotations
import os
import argparse, csv, sys, time
from collections import defaultdict
from pathlib import Path

import pandas  # noqa: F401  (import before torch; see train_foundation_stereo.py)
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FS_DIR = ROOT / "third_party/FoundationStereo"
sys.path.insert(0, str(FS_DIR))

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from models.signed_foundation_stereo import SignedFoundationStereo


def load_model(ckpt, cfg_path, signed, d_neg, d_pos, device="cuda"):
    cfg = OmegaConf.load(cfg_path)
    cfg.setdefault("vit_size", "vitl")
    cfg.setdefault("hiera", 0)
    cfg.setdefault("low_memory", 0)
    cfg.setdefault("valid_iters", 32)
    if signed:
        model = SignedFoundationStereo(cfg, d_neg=d_neg, d_pos=d_pos)
    else:
        model = FoundationStereo(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt}: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def infer(model, L_np, R_np, iters, device="cuda"):
    L = torch.from_numpy(L_np).permute(2, 0, 1).float()[None].to(device)
    R = torch.from_numpy(R_np).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(L.shape, divis_by=32)
    L, R = padder.pad(L, R)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        disp = model(L, R, iters=iters, test_mode=True)
    disp = padder.unpad(disp).float().cpu().numpy()
    return disp[0, 0] if disp.ndim == 4 else disp[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default=str(FS_DIR / "pretrained_models/11-33-40/cfg.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--signed-volume", action="store_true")
    ap.add_argument("--d-neg", type=int, default=64)
    ap.add_argument("--d-pos", type=int, default=192)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, args.cfg, args.signed_volume, args.d_neg, args.d_pos)

    rows = []; n = 0; t0 = time.time()
    for shift_dir in sorted(Path(args.dataset).rglob("shift_*")):
        if not shift_dir.is_dir() or not (shift_dir / "left.png").exists():
            continue
        delta = int(shift_dir.name.split("_")[-1])
        frame = shift_dir.parent.name
        scene = shift_dir.parent.parent.name
        L = np.asarray(Image.open(shift_dir / "left.png").convert("RGB")).astype(np.float32)
        R = np.asarray(Image.open(shift_dir / "right.png").convert("RGB")).astype(np.float32)
        gt = np.load(shift_dir / "disparity.npy")
        try:
            pred = infer(model, L, R, iters=args.iters)
        except Exception as e:
            print(f"  SKIP {shift_dir}: {e}", flush=True)
            continue
        fin = np.isfinite(gt) & np.isfinite(pred)
        if not fin.any():
            continue
        e = np.abs(pred - gt)[fin]
        rows.append(dict(scene=scene, frame=frame, delta=delta,
                         epe=float(e.mean()), bad_1=float((e > 1).mean()),
                         bad_3=float((e > 3).mean()), n=int(e.size)))
        n += 1
        if n % 50 == 0:
            print(f"  ... {n} pairs done ({int(time.time()-t0)}s)", flush=True)

    with open(out / "per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    by_d = defaultdict(list)
    for r in rows: by_d[r["delta"]].append(r)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["delta_px", "n_frames", "epe", "bad_1", "bad_3"])
        for d in sorted(by_d):
            ls = by_d[d]
            epe = float(np.mean([r["epe"] for r in ls]))
            b1 = float(np.mean([r["bad_1"] for r in ls]))
            b3 = float(np.mean([r["bad_3"] for r in ls]))
            w.writerow([d, len(ls), epe, b1, b3])
            print(f"  Δ={d:+3d}: EPE={epe:7.3f} bad_1={b1*100:5.1f}% bad_3={b3*100:5.1f}%", flush=True)
    print(f"\nwrote {out}/per_pair.csv, {out}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
