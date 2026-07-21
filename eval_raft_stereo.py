"""Evaluate a RAFT-Stereo checkpoint (vanilla or symmetric) on a multi-ZDP dataset.

Outputs:
  <out>/per_pair.csv   one row per (scene, frame, shift) with epe/bad_1/bad_3
  <out>/summary.csv    one row per shift Δ with aggregated metrics
"""
import argparse, csv, json, sys, types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/RAFT-Stereo"))
sys.path.insert(0, str(ROOT / "third_party/RAFT-Stereo/core"))

# Reuse helpers from evaluate.py
sys.path.insert(0, str(ROOT))
from evaluate import find_shift_dirs, load_sample


def build_args():
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
    a.mixed_precision = False
    return a


def load_model(ckpt: str, symmetric: bool, device: str = "cuda"):
    if symmetric:
        from raft_stereo_sym import SymRAFTStereo as ModelCls
    else:
        from core.raft_stereo import RAFTStereo as ModelCls
    model = torch.nn.DataParallel(ModelCls(build_args()))
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)  # strict=False tolerates a renamed block in sym
    return model.module.to(device).eval()


def infer(model, L_np, R_np, iters=32, device: str = "cuda"):
    from core.utils.utils import InputPadder
    L = torch.from_numpy(L_np).permute(2, 0, 1).float()[None].to(device)
    R = torch.from_numpy(R_np).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(L.shape, divis_by=32)
    L, R = padder.pad(L, R)
    with torch.no_grad():
        _, flow_up = model(L, R, iters=iters, test_mode=True)
    # InputPadder.unpad requires 4-D input; take channel after unpad.
    disp = padder.unpad(-flow_up)[:, 0].cpu().numpy()[0]
    return disp


def metrics(pred, gt):
    finite = np.isfinite(gt) & np.isfinite(pred)
    e = np.abs(pred - gt)[finite]
    if e.size == 0:
        return dict(epe=float("nan"), bad_1=float("nan"), bad_3=float("nan"), n=0)
    return dict(
        epe=float(e.mean()),
        bad_1=float((e > 1).mean()),
        bad_3=float((e > 3).mean()),
        n=int(e.size),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--symmetric", action="store_true",
                   help="use SymRAFTStereo from raft_stereo_sym.py")
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--device", default="cuda",
                   help="cuda | cpu (fall back to cpu when driver mismatched)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.symmetric, device=args.device)
    per_pair_path = out / "per_pair.csv"
    per_pair = open(per_pair_path, "w", newline="")
    pw = csv.writer(per_pair)
    pw.writerow(["scene", "frame", "delta_px", "epe", "bad_1", "bad_3", "n_valid"])

    agg = {}  # delta -> list of metric dicts
    n_pairs = 0
    for scene, frame, delta, sd in find_shift_dirs(Path(args.dataset)):
        try:
            L, R, gt, meta = load_sample(sd)
        except Exception as e:
            print(f"  SKIP {sd}: {e}", file=sys.stderr)
            continue
        pred = infer(model, L, R, iters=args.iters, device=args.device)
        m = metrics(pred, gt)
        pw.writerow([scene, frame, delta, m["epe"], m["bad_1"], m["bad_3"], m["n"]])
        agg.setdefault(delta, []).append(m)
        n_pairs += 1
        if n_pairs % 10 == 0:
            print(f"  ... {n_pairs} pairs done")
    per_pair.close()

    with open(out / "summary.csv", "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["delta_px", "n_frames", "epe", "bad_1", "bad_3"])
        for delta in sorted(agg):
            ms = agg[delta]
            mean = lambda k: float(np.mean([x[k] for x in ms if np.isfinite(x[k])]))
            sw.writerow([delta, len(ms), f"{mean('epe'):.3f}",
                         f"{mean('bad_1'):.4f}", f"{mean('bad_3'):.4f}"])
    print(f"wrote {per_pair_path} and {out/'summary.csv'} ({n_pairs} pairs)")


if __name__ == "__main__":
    main()
