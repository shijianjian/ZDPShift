"""Scan an SBS 3D clip densely and rank frames by behind-screen (negative)
disparity content. Reports how many frames contain meaningful negative
disparity and dumps the top-K max-negative frames (L/R halves + score) for
the demo comparison.
"""
import os
import sys, argparse, json
from pathlib import Path
import numpy as np
import cv2, torch
from analyze_real_3d import build_model, flow_lr, split_sbs
from depth_anything_wrapper import DepthAnything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("outdir")
    ap.add_argument("--squeeze", action="store_true")
    ap.add_argument("--stride", type=int, default=4, help="sample every Nth frame")
    ap.add_argument("--deadband", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-neg-pct", type=float, default=2.0,
                    help="frame 'contains negative' if behind-screen%% exceeds this")
    # sanity gates against flow-failure on dark / textureless / scene-cut frames
    ap.add_argument("--min-lum", type=float, default=22.0, help="reject frames darker than this mean luminance")
    ap.add_argument("--max-abs-median", type=float, default=25.0,
                    help="reject frames whose median|d| exceeds this (global-shift artifact)")
    ap.add_argument("--min-depth-corr", type=float, default=0.30,
                    help="require corr(disparity, monocular proximity) above this (valid stereo)")
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    model, args = build_model()
    da = DepthAnything(size="small")
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(range(int(total*0.04), int(total*0.96), a.stride))

    rows = []
    n_dark = n_shift = n_lowcorr = 0
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, bgr = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L, R = split_sbs(rgb, a.squeeze)
        if L.mean() < a.min_lum:                # dark / fade frame
            n_dark += 1; continue
        ux, uy = flow_lr(model, args, L, R)
        d = -ux
        valid = (np.abs(uy) < 3.0) & (np.abs(d) < np.percentile(np.abs(d), 99.5))
        dv = d[valid]
        if dv.size < 1000:
            continue
        if abs(np.median(dv)) > a.max_abs_median:   # whole frame shifted = flow failure
            n_shift += 1; continue
        # validate: real stereo -> disparity correlates with monocular proximity
        prox = da.predict(torch.from_numpy(L).permute(2,0,1).float()/255.0)[0].cpu().numpy()
        prox = cv2.resize(prox, (d.shape[1], d.shape[0]))
        corr = float(np.corrcoef(d[valid], prox[valid])[0, 1])
        if not np.isfinite(corr) or corr < a.min_depth_corr:
            n_lowcorr += 1; continue
        pct_neg = 100 * np.mean(dv < -a.deadband)
        rows.append(dict(frame=int(fi), pct_neg=float(pct_neg),
                         min_d=float(np.percentile(dv, 0.5)),
                         median_d=float(np.median(dv)),
                         corr=corr, lum=float(L.mean())))
        if len(rows) % 25 == 0:
            print(f"  kept {len(rows)} (dark {n_dark}, shift {n_shift}, lowcorr {n_lowcorr}) ...", flush=True)

    print(f"gates: rejected {n_dark} dark, {n_shift} global-shift, {n_lowcorr} low-depth-corr")
    n_contain = sum(1 for r in rows if r["pct_neg"] >= a.min_neg_pct)
    rows_sorted = sorted(rows, key=lambda r: r["pct_neg"], reverse=True)
    print(f"\n==== {Path(a.video).stem[:50]} ====")
    print(f"frames scanned: {len(rows)}  (stride {a.stride})")
    print(f"frames with >= {a.min_neg_pct:.0f}% behind-screen: {n_contain} "
          f"({100*n_contain/max(1,len(rows)):.0f}%)")
    print(f"peak behind-screen %: {rows_sorted[0]['pct_neg']:.1f}  "
          f"(frame {rows_sorted[0]['frame']})")
    print("top frames by behind-screen %:")
    for r in rows_sorted[:a.topk]:
        print(f"  frame {r['frame']:6d}: {r['pct_neg']:5.1f}%  "
              f"min_d={r['min_d']:+6.1f}  median_d={r['median_d']:+5.1f}  corr={r['corr']:+.2f}")

    # dump top-K L/R halves for the demo
    for rank, r in enumerate(rows_sorted[:a.topk]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"]); ok, bgr = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L, R = split_sbs(rgb, a.squeeze)
        cv2.imwrite(str(outdir/f"top{rank:02d}_f{r['frame']}_L.png"), cv2.cvtColor(L, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outdir/f"top{rank:02d}_f{r['frame']}_R.png"), cv2.cvtColor(R, cv2.COLOR_RGB2BGR))
    cap.release()
    (outdir/"scan.json").write_text(json.dumps(
        dict(video=a.video, n_scanned=len(rows), n_contain=n_contain,
             min_neg_pct=a.min_neg_pct, top=rows_sorted[:a.topk]), indent=2))
    print("dumped top-K L/R halves to", outdir)


if __name__ == "__main__":
    main()
