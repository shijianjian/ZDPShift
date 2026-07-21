"""No-regression check on a standard positive-disparity benchmark (KITTI 2015).

Evaluates RAFT-Stereo before (SceneFlow-pretrained) and after our ZDPShift
fine-tuning on KITTI-2015 training pairs (real, all-positive, independent of
our training data). If post ~ pre, extending to negative disparity did not
cost positive-regime accuracy.
"""
import os
import sys, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
from eval_raft_stereo import load_model, infer   # reuse loader/infer

KITTI = Path(os.path.join(os.environ.get("SEARAFT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SEA-RAFT")), "datasets/KITTI/training"))
N = 200   # full KITTI-2015 training set

CKPTS = {
    "pre  (SceneFlow)":  (str(ROOT / "third_party/RAFT-Stereo/models/raftstereo-sceneflow.pth"), False),
    "post (ours, v3)":   (str(ROOT / "weights/raft_zdpshift.pth"), True),
}


def kitti_pairs(n):
    lefts = sorted(glob.glob(str(KITTI / "image_2" / "*_10.png")))[:n]
    trips = []
    for lp in lefts:
        stem = Path(lp).name
        rp = KITTI / "image_3" / stem
        gp = KITTI / "disp_occ_0" / stem
        if rp.exists() and gp.exists():
            trips.append((lp, str(rp), str(gp)))
    return trips


def main():
    pairs = kitti_pairs(N)
    print(f"KITTI pairs: {len(pairs)}")
    for label, (ckpt, sym) in CKPTS.items():
        model = load_model(ckpt, symmetric=sym)
        epes, bad3 = [], []
        for lp, rp, gp in pairs:
            L = np.asarray(Image.open(lp).convert("RGB")).astype(np.float32)
            R = np.asarray(Image.open(rp).convert("RGB")).astype(np.float32)
            gt = np.asarray(Image.open(gp)).astype(np.float32) / 256.0   # KITTI uint16/256; 0=invalid
            pred = infer(model, L, R, iters=32)
            valid = gt > 0
            if not valid.any():
                continue
            e = np.abs(pred - gt)[valid]
            epes.append(float(e.mean()))
            bad3.append(float((e > 3).mean()))
        print(f"{label:20s}  EPE={np.mean(epes):.3f}  bad-3={100*np.mean(bad3):.1f}%  (n={len(epes)})")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
