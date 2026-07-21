"""Measure signed disparity in a real SBS 3D movie clip, independently of our
stereo model, using sign-agnostic optical flow (SEA-RAFT).

For a delivered stereoscopic pair the zero-disparity plane is the screen: a
pixel with d = x_L - x_R > 0 sits in front of the screen (crossed / pop-out),
d < 0 sits behind it (uncrossed / into the scene). Optical flow from the left
to the right view gives u_x with x_R = x_L + u_x, so d = -u_x -- and because
flow is unconstrained in sign, this reads out behind-screen content directly,
with no positive-disparity assumption.

Usage: python analyze_real_3d.py <video.mp4> <out_prefix> [--squeeze]
"""
import os
import sys, os, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEARAFT = Path(os.environ.get("SEARAFT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/SEA-RAFT")))
sys.path.insert(0, str(SEARAFT))
sys.path.insert(0, str(SEARAFT / "core"))
CFG = str(SEARAFT / "config/eval/kitti-M.json")
CKPT = str(SEARAFT / "models/Tartan-C-T-TSKH-kitti432x960-M.pth")

from config.parser import json_to_args
from raft import RAFT
from utils.utils import load_ckpt


def build_model():
    args = json_to_args(CFG)
    args.device = "cuda"
    model = RAFT(args)
    load_ckpt(model, CKPT)
    return model.cuda().eval(), args


def to_tensor(img):  # HxWx3 uint8 RGB -> 1x3xHxW float
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].cuda()


@torch.no_grad()
def flow_lr(model, args, L, R):
    """Return horizontal flow u_x (L->R), same HxW as inputs."""
    H, W = L.shape[:2]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    Lt, Rt = to_tensor(L), to_tensor(R)
    Lt = F.pad(Lt, (0, pw, 0, ph), mode="replicate")
    Rt = F.pad(Rt, (0, pw, 0, ph), mode="replicate")
    out = model(Lt, Rt, iters=args.iters, test_mode=True)
    flow = out["flow"][-1][0].cpu().numpy()          # 2xHxW
    return flow[0, :H, :W], flow[1, :H, :W]           # u_x, u_y


def split_sbs(frame, squeeze_fix, target_w=1024):
    """frame HxWx3 -> left, right eyes, optionally un-squeezed to ~16:9."""
    h, w = frame.shape[:2]
    half = w // 2
    L, R = frame[:, :half], frame[:, half:2 * half]
    if squeeze_fix:                                   # half-SBS: stretch width x2
        newh = int(round(target_w * (h) / (2 * half)))
        newh -= newh % 8
        L = cv2.resize(L, (target_w, newh), interpolation=cv2.INTER_AREA)
        R = cv2.resize(R, (target_w, newh), interpolation=cv2.INTER_AREA)
    return L, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--squeeze", action="store_true",
                    help="input is half-SBS (each eye horizontally squeezed); un-squeeze")
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--deadband", type=float, default=0.5, help="|d|<=deadband counted as on-screen")
    args_cli = ap.parse_args()

    model, args = build_model()
    cap = cv2.VideoCapture(args_cli.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # sample frames, skipping first/last 8%
    idxs = np.linspace(int(total * 0.08), int(total * 0.92), args_cli.nframes).astype(int)

    all_d, per_frame, vis_frames = [], [], []
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L, R = split_sbs(rgb, args_cli.squeeze)
        ux, uy = flow_lr(model, args, L, R)
        d = -ux                                        # signed disparity (px, in this resolution)
        # robust validity: drop extreme flow (occlusion / failure) and high vertical flow
        valid = (np.abs(uy) < 3.0) & (np.abs(d) < np.percentile(np.abs(d), 99.5))
        dv = d[valid]
        db = args_cli.deadband
        pct_neg = 100 * np.mean(dv < -db)
        pct_pos = 100 * np.mean(dv > db)
        per_frame.append((int(fi), pct_neg, pct_pos, float(np.median(dv)),
                          float(np.median(np.abs(uy)))))
        all_d.append(dv)
        if k in (args_cli.nframes // 4, args_cli.nframes // 2, 3 * args_cli.nframes // 4):
            vis_frames.append((L, d, valid, pct_neg))
        print(f"frame {fi:5d}: neg(behind)={pct_neg:5.1f}%  pos(front)={pct_pos:5.1f}%  "
              f"median d={np.median(dv):+5.2f}  |v|med={np.median(np.abs(uy)):.2f}", flush=True)
    cap.release()

    allv = np.concatenate(all_d)
    db = args_cli.deadband
    frac_neg = 100 * np.mean(allv < -db)
    frac_pos = 100 * np.mean(allv > db)
    print("\n==== SUMMARY ====")
    print(f"frames analysed: {len(per_frame)}")
    print(f"behind-screen (d<0): {frac_neg:.1f}%   in-front (d>0): {frac_pos:.1f}%   "
          f"on-screen: {100-frac_neg-frac_pos:.1f}%")
    print(f"per-frame behind-screen %: min {min(p[1] for p in per_frame):.0f} "
          f"max {max(p[1] for p in per_frame):.0f} "
          f"mean {np.mean([p[1] for p in per_frame]):.0f}")

    # ---- visualization ----
    vmax = float(np.percentile(np.abs(allv), 95))
    n = len(vis_frames)
    fig = plt.figure(figsize=(13, 3.2 * n + 2.4))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[3] * n + [2.3], hspace=0.28, wspace=0.08)
    for i, (L, d, valid, pn) in enumerate(vis_frames):
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(L); ax.axis("off")
        if i == 0: ax.set_title("left view", fontsize=11)
        dd = np.where(valid, d, np.nan)
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(dd, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.axis("off")
        if i == 0: ax.set_title("signed disparity  (red = in front, blue = behind screen)", fontsize=11)
        ax = fig.add_subplot(gs[i, 2])
        beh = np.zeros((*d.shape, 4)); beh[..., 2] = 1.0
        beh[..., 3] = np.where(valid & (d < -db), 0.55, 0.0)
        ax.imshow(L); ax.imshow(beh); ax.axis("off")
        ax.text(0.02, 0.06, f"{pn:.0f}% behind screen", transform=ax.transAxes,
                color="w", fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#2c5aa0", ec="none"))
        if i == 0: ax.set_title("behind-screen mask (negative disparity)", fontsize=11)
    # histogram
    ax = fig.add_subplot(gs[n, :])
    ax.hist(allv[np.abs(allv) < vmax * 2.5], bins=140, color="0.6")
    ax.axvline(0, color="k", lw=1.2)
    ax.axvspan(allv.min(), 0, color="#2c5aa0", alpha=0.10)
    ax.set_xlim(-vmax * 2.2, vmax * 2.2)
    ax.set_title(f"signed-disparity distribution across {len(per_frame)} frames "
                 f"— {frac_neg:.0f}% of pixels behind the screen (negative)", fontsize=11)
    ax.set_xlabel("disparity  d = x_L - x_R  (px)"); ax.set_yticks([])
    fig.suptitle(Path(args_cli.video).stem[:60], fontsize=10, y=0.995, color="0.4")
    out = args_cli.out + ".png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
