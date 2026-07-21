"""Determine L/R eye order of an SBS clip, ZDP-independently.

Near objects always have LARGER disparity than far objects (d = fB/Z - Delta,
Delta constant), regardless of the ZDP. So if measured disparity correlates
POSITIVELY with monocular proximity, the left-half is the left eye (order OK);
if NEGATIVE, the halves are swapped and every sign is inverted.
"""
import os
import sys, argparse
import numpy as np
import cv2, torch
from analyze_real_3d import build_model, flow_lr, split_sbs
from depth_anything_wrapper import DepthAnything

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--squeeze", action="store_true")
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()

model, args = build_model()
da = DepthAnything(size="small")
cap = cv2.VideoCapture(a.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(int(total*0.1), int(total*0.9), a.n).astype(int)

corrs = []
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, bgr = cap.read()
    if not ok: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    L, R = split_sbs(rgb, a.squeeze)
    ux, uy = flow_lr(model, args, L, R)
    d = -ux                                  # signed disparity for as-is order
    prox = da.predict(torch.from_numpy(L).permute(2,0,1).float()/255.0)[0].cpu().numpy()  # larger=closer
    prox = cv2.resize(prox, (d.shape[1], d.shape[0]))
    m = (np.abs(uy) < 3) & (np.abs(d) < np.percentile(np.abs(d), 99))
    c = np.corrcoef(d[m], prox[m])[0, 1]
    corrs.append(c)
    print(f"frame {fi:5d}: corr(disparity, proximity) = {c:+.3f}   median d = {np.median(d[m]):+.2f}")
cap.release()
mc = float(np.mean(corrs))
print(f"\nmean corr = {mc:+.3f}  ->  {'ORDER OK (left-half = left eye)' if mc>0 else 'SWAPPED (left-half = right eye); flip sign'}")
