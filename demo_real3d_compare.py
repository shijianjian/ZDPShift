"""Demo on real 3D-movie frames: FoundationStereo (zero-shot, positive-only)
vs. our signed FoundationStereo. On the max-negative frames, the zero-shot
model can only report positive disparity for behind-screen content, while ours
recovers the negative regime.

Input: a directory of top{rank}_f{frame}_L.png / _R.png dumped by
scan_negative_frames.py. Output: a comparison panel per frame.
"""
import os
import sys, argparse, glob, re
from pathlib import Path
import numpy as np
import cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
from eval_foundation_stereo import load_model, infer

CFG = os.path.join(os.environ.get("FOUNDATIONSTEREO_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/FoundationStereo")), "pretrained_models/11-33-40/cfg.yaml")
ZS = os.path.join(os.environ.get("FOUNDATIONSTEREO_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/FoundationStereo")), "pretrained_models/11-33-40/model_best_bp2.pth")
SIGNED = str(ROOT / "weights/fs_zdpshift_signed.pth")
DNEG, DPOS = 64, 192


def anaglyph(L, R):
    """Red-cyan anaglyph: left eye -> red channel, right eye -> green+blue."""
    Lg = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
    Rg = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    out = np.zeros_like(L)
    out[..., 0] = Lg          # R channel from left eye
    out[..., 1] = Rg          # G channel from right eye
    out[..., 2] = Rg          # B channel from right eye
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("out")
    ap.add_argument("--ranks", type=int, default=8, help="how many top frames to show")
    ap.add_argument("--frames", type=str, default=None,
                    help="comma-separated frame numbers to use (overrides --ranks)")
    ap.add_argument("--iters", type=int, default=32)
    a = ap.parse_args()

    # Any "<name>_L.png" paired with "<name>_R.png"; label is <name>.
    all_lp = sorted(glob.glob(f"{a.indir}/*_L.png"))
    pairs = []
    if a.frames:
        # legacy: select scan_negative_frames dumps by frame number (top{r}_f{frame})
        want = [int(x) for x in a.frames.split(",")]
        by_frame = {}
        for lp in all_lp:
            m = re.search(r"top(\d+)_f(\d+)", lp)
            if m:
                by_frame[int(m.group(2))] = (int(m.group(1)), lp)
        for fr in want:
            if fr in by_frame:
                rank, lp = by_frame[fr]
                pairs.append((rank, fr, lp, lp.replace("_L.png", "_R.png")))
            else:
                print(f"warning: frame {fr} not found in {a.indir}")
    else:
        for i, lp in enumerate(all_lp[:a.ranks]):
            label = os.path.basename(lp)[:-6]           # strip "_L.png"
            pairs.append((i, label, lp, lp.replace("_L.png", "_R.png")))
    if not pairs:
        print("no frames selected in", a.indir); return

    print("loading zero-shot FoundationStereo ...")
    m_zs = load_model(ZS, CFG, False, DNEG, DPOS)
    print("loading signed FoundationStereo ...")
    m_sg = load_model(SIGNED, CFG, True, DNEG, DPOS)

    n = len(pairs)
    fig = plt.figure(figsize=(16.5, 2.9 * n))
    gs = fig.add_gridspec(n, 5, wspace=0.03, hspace=0.10)
    titles = ["left", "right", "anaglyph (red-cyan)",
              "FoundationStereo (zero-shot)\npositive-only volume",
              "FoundationStereo + ours\nsigned cost volume"]

    for i, (rank, frame, lp, rp) in enumerate(pairs):
        L = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        Lf, Rf = L.astype(np.float32), R.astype(np.float32)
        d_zs = infer(m_zs, Lf, Rf, a.iters)
        d_sg = infer(m_sg, Lf, Rf, a.iters)
        pct_neg_zs = 100 * np.mean(d_zs < -0.5)
        pct_neg_sg = 100 * np.mean(d_sg < -0.5)
        vmax = float(np.percentile(np.abs(d_sg), 96))

        def panel(col, img, cmap=None, vmax_=None):
            ax = fig.add_subplot(gs[i, col])
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=-vmax_, vmax=vmax_)
            else:
                ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[col], fontsize=11)
            return ax

        panel(0, L)
        panel(1, R)
        panel(2, anaglyph(L, R))
        ax = panel(3, d_zs, "RdBu_r", vmax)
        ax.text(0.03, 0.07, f"{pct_neg_zs:.0f}% behind screen", transform=ax.transAxes,
                color="w", fontsize=9.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#888", ec="none"))
        ax = panel(4, d_sg, "RdBu_r", vmax)
        ax.text(0.03, 0.07, f"{pct_neg_sg:.0f}% behind screen", transform=ax.transAxes,
                color="w", fontsize=9.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#2c5aa0", ec="none"))

    fig.text(0.5, 0.085 / n, "disparity colormap:  "
             "red = positive (in front of screen)      blue = negative (behind screen)",
             ha="center", fontsize=10, color="0.25")
    out = a.out + ".png"
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
