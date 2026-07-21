"""Quick behind-screen ranker over several SBS clips. For each video, sample a
few frames with the same gating as scan_negative_frames.py (reject dark / global
-shift / low-depth-corr) and report median + peak behind-screen (negative) %.
"""
import os
import sys, argparse, glob
from pathlib import Path
import numpy as np
import cv2, torch
from analyze_real_3d import build_model, flow_lr, split_sbs
from depth_anything_wrapper import DepthAnything

DIR = os.environ.get("SBS_VIDEO_DIR", os.path.expanduser("~/3d_movies"))


def score_video(path, model, args, da, n=12, squeeze=True, deadband=0.5):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(int(total*0.08), int(total*0.92), n).astype(int)
    negs, best = [], (-1, None)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, bgr = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L, R = split_sbs(rgb, squeeze)
        if L.mean() < 22: continue
        ux, uy = flow_lr(model, args, L, R); d = -ux
        valid = (np.abs(uy) < 3) & (np.abs(d) < np.percentile(np.abs(d), 99.5))
        dv = d[valid]
        if dv.size < 1000 or abs(np.median(dv)) > 25: continue
        prox = da.predict(torch.from_numpy(L).permute(2,0,1).float()/255.)[0].cpu().numpy()
        prox = cv2.resize(prox, (d.shape[1], d.shape[0]))
        corr = float(np.corrcoef(d[valid], prox[valid])[0,1])
        if not np.isfinite(corr) or corr < 0.30: continue
        pn = 100*np.mean(dv < -deadband)
        negs.append(pn)
        if pn > best[0]: best = (pn, int(fi))
    cap.release()
    if not negs:
        return dict(median=0.0, peak=0.0, n=0, best_frame=None)
    return dict(median=float(np.median(negs)), peak=float(np.max(negs)),
                n=len(negs), best_frame=best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="youtube id substrings")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    model, args = build_model(); da = DepthAnything(size="small")
    rows = []
    for vid in a.ids:
        hits = glob.glob(f"{DIR}/*{vid}*.mp4")
        if not hits:
            print(f"[skip] no file for {vid}"); continue
        path = hits[0]
        w = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_WIDTH))
        squeeze = w < 3000               # 3840-wide = full-SBS (no squeeze)
        r = score_video(path, model, args, da, n=a.n, squeeze=squeeze)
        r["id"] = vid; r["name"] = Path(path).name[:46]; rows.append(r)
        print(f"{vid:14s} median={r['median']:5.1f}%  peak={r['peak']:5.1f}%  "
              f"(kept {r['n']}, best f{r['best_frame']})  {r['name']}", flush=True)
    rows.sort(key=lambda r: r["peak"], reverse=True)
    print("\n==== ranked by peak behind-screen % ====")
    for r in rows:
        print(f"  {r['peak']:5.1f}%  median {r['median']:4.1f}%  {r['id']}  {r['name']}")


if __name__ == "__main__":
    main()
