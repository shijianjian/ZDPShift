"""KITTI-2015 no-regression eval for FoundationStereo: pre (SceneFlow/native)
vs our fine-tuned (data-only and signed-volume).
"""
import os
import sys, glob
from pathlib import Path
import pandas  # noqa: F401 (before torch; FS Utils quirk)
import numpy as np
from PIL import Image
import torch

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
from eval_foundation_stereo import load_model, infer   # reuse FS loader/infer

KITTI = Path(os.path.join(os.environ.get("SEARAFT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SEA-RAFT")), "datasets/KITTI/training"))
CFG = os.path.join(os.environ.get("FOUNDATIONSTEREO_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/FoundationStereo")), "pretrained_models/11-33-40/cfg.yaml")
N = 200

CKPTS = {
    "pre  (native)":  (os.path.join(os.environ.get("FOUNDATIONSTEREO_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/FoundationStereo")), "pretrained_models/11-33-40/model_best_bp2.pth"), False),
    "post signed":    (str(ROOT/"weights/fs_zdpshift_signed.pth"), True),
}

def main():
    lefts = sorted(glob.glob(str(KITTI/"image_2"/"*_10.png")))[:N]
    pairs = [(lp, str(KITTI/"image_3"/Path(lp).name), str(KITTI/"disp_occ_0"/Path(lp).name))
             for lp in lefts if (KITTI/"disp_occ_0"/Path(lp).name).exists()]
    print(f"KITTI pairs: {len(pairs)}", flush=True)
    for label,(ckpt,signed) in CKPTS.items():
        m = load_model(ckpt, CFG, signed, 64, 192)
        epes=[]; bad3=[]
        for lp,rp,gp in pairs:
            L=np.asarray(Image.open(lp).convert("RGB")).astype(np.float32)
            R=np.asarray(Image.open(rp).convert("RGB")).astype(np.float32)
            gt=np.asarray(Image.open(gp)).astype(np.float32)/256.0
            pred=infer(m,L,R,iters=32); v=gt>0
            if not v.any(): continue
            e=np.abs(pred-gt)[v]; epes.append(float(e.mean())); bad3.append(float((e>3).mean()))
        print(f"{label:18s}  EPE={np.mean(epes):.3f}  bad-3={100*np.mean(bad3):.1f}%  (n={len(epes)})", flush=True)
        del m; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
