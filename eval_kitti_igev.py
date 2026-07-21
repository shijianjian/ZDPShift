"""KITTI-2015 no-regression eval for IGEV-Stereo: pre (SceneFlow) vs our
fine-tuned (data-only and signed-volume). All-positive, disjoint from our data.
"""
import os
import sys, glob, types
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
IGEV = Path(os.environ.get("IGEV_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/IGEV/IGEV-Stereo")))
sys.path.insert(0, str(IGEV)); sys.path.insert(0, str(IGEV / "core"))

from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from models.signed_igev_stereo import SignedIGEVStereo

KITTI = Path(os.path.join(os.environ.get("SEARAFT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SEA-RAFT")), "datasets/KITTI/training"))
N = 200

def margs():
    a = types.SimpleNamespace(hidden_dims=[128,128,128], n_downsample=2, n_gru_layers=3,
                              corr_levels=2, corr_radius=4, max_disp=192,
                              mixed_precision=True, precision_dtype="float16")
    return a

def load(ckpt, signed):
    m = SignedIGEVStereo(margs(), d_neg=64, d_pos=192) if signed else IGEVStereo(margs())
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.cuda().eval()

@torch.no_grad()
def infer(m, L, R):
    Lt = torch.from_numpy(L).permute(2,0,1).float()[None].cuda()
    Rt = torch.from_numpy(R).permute(2,0,1).float()[None].cuda()
    pad = InputPadder(Lt.shape, divis_by=32); Lt, Rt = pad.pad(Lt, Rt)
    d = m(Lt, Rt, iters=32, test_mode=True)
    d = pad.unpad(d).cpu().numpy()
    return d[0,0] if d.ndim==4 else d[0]

CKPTS = {
    "pre  (SceneFlow)": (str(IGEV/"pretrained_models/IGEV/sceneflow/sceneflow.pth"), False),
    "post signed":      (str(ROOT/"weights/igev_zdpshift_signed.pth"), True),
}

def main():
    lefts = sorted(glob.glob(str(KITTI/"image_2"/"*_10.png")))[:N]
    pairs = [(lp, str(KITTI/"image_3"/Path(lp).name), str(KITTI/"disp_occ_0"/Path(lp).name))
             for lp in lefts if (KITTI/"disp_occ_0"/Path(lp).name).exists()]
    print(f"KITTI pairs: {len(pairs)}")
    for label,(ckpt,signed) in CKPTS.items():
        m = load(ckpt, signed); epes=[]; bad3=[]
        for lp,rp,gp in pairs:
            L=np.asarray(Image.open(lp).convert("RGB")).astype(np.float32)
            R=np.asarray(Image.open(rp).convert("RGB")).astype(np.float32)
            gt=np.asarray(Image.open(gp)).astype(np.float32)/256.0
            pred=infer(m,L,R); v=gt>0
            if not v.any(): continue
            e=np.abs(pred-gt)[v]; epes.append(float(e.mean())); bad3.append(float((e>3).mean()))
        print(f"{label:18s}  EPE={np.mean(epes):.3f}  bad-3={100*np.mean(bad3):.1f}%  (n={len(epes)})", flush=True)
        del m; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
