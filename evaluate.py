"""
Evaluate a stereo model across ZDP shifts and produce robustness curves.

Expects a dataset in the layout produced by blender_render.py:
  dataset_root/
    scene_XXX/
      frame_XXXX/
        shift_-16/  shift_+0/  shift_+8/  ...
          left.png  right.png  disparity.npy  meta.json

Supported model backends (--model):
  foundationstereo   FoundationStereo (requires FoundationStereo repo on PYTHONPATH)
  dummy              Returns zero disparity (baseline sanity check)

Usage:
  python evaluate.py \\
      --dataset  dataset/ \\
      --model    foundationstereo \\
      --ckpt     /path/to/model.pth \\
      --out      results/

Outputs:
  results/per_shift.csv      — EPE / D1 / neg_pct per Δ (aggregated over all frames)
  results/robustness.png     — ZDP robustness curve
  results/per_frame.csv      — per-frame breakdown
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# Dataset loader
# =============================================================================

def find_shift_dirs(root: Path):
    """
    Yield (scene, frame, delta, shift_dir) for every shift directory found.
    Handles layouts:
      root/shift_*/          (single frame)
      root/frame/shift_*/
      root/scene/frame/shift_*/
    """
    def _shift_dirs_in(d: Path):
        return sorted(
            [x for x in d.iterdir() if x.is_dir() and x.name.startswith("shift_")],
            key=lambda x: _parse_delta(x.name),
        )

    def _parse_delta(name: str) -> int:
        return int(name.replace("shift_", ""))

    # Detect depth
    direct = _shift_dirs_in(root)
    if direct:
        for sd in direct:
            yield (".", ".", _parse_delta(sd.name), sd)
        return

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sds = _shift_dirs_in(child)
        if sds:
            for sd in sds:
                yield (".", child.name, _parse_delta(sd.name), sd)
            continue
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir():
                continue
            sds = _shift_dirs_in(grandchild)
            for sd in sds:
                yield (child.name, grandchild.name, _parse_delta(sd.name), sd)


def load_sample(shift_dir: Path):
    left  = np.array(Image.open(shift_dir / "left.png"))
    right = np.array(Image.open(shift_dir / "right.png"))
    disp  = np.load(shift_dir / "disparity.npy")
    meta  = {}
    if (shift_dir / "meta.json").exists():
        with open(shift_dir / "meta.json") as f:
            meta = json.load(f)
    return left, right, disp, meta


# =============================================================================
# Metrics
# =============================================================================

def epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """End-point error (mean absolute disparity error) over valid pixels."""
    err = np.abs(pred - gt)
    if mask is not None:
        err = err[mask]
    return float(err.mean()) if err.size > 0 else float("nan")


def d1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, threshold: float = 3.0) -> float:
    """D1: % of pixels with |pred - gt| > max(threshold, 0.05·|gt|)."""
    abs_err = np.abs(pred - gt)
    thr     = np.maximum(threshold, 0.05 * np.abs(gt))
    bad     = abs_err > thr
    if mask is not None:
        bad = bad[mask]
    return float(bad.mean() * 100) if bad.size > 0 else float("nan")


def valid_mask(gt: np.ndarray) -> np.ndarray:
    """Mask for pixels with valid (non-zero, finite) GT disparity."""
    return (gt != 0) & np.isfinite(gt)


# =============================================================================
# Model backends
# =============================================================================

class DummyModel:
    """Returns zero disparity — useful as a baseline."""

    def predict(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        H, W = left.shape[:2]
        return np.zeros((H, W), dtype=np.float32)


class FoundationStereoModel:
    """Wrapper that delegates to load_foundationstereo (which lives next to this file)."""

    def __init__(self, ckpt: str, device: str = "cuda", iters: int = 32):
        from load_foundationstereo import load_model
        self.model = load_model(ckpt, device)
        self.device = device
        self.iters = iters

    def predict(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        from load_foundationstereo import predict_disparity
        return predict_disparity(self.model, left, right, iters=self.iters, device=self.device)


def make_model(name: str, ckpt: str | None, device: str, iters: int = 32):
    if name == "dummy":
        return DummyModel()
    if name == "foundationstereo":
        if not ckpt:
            sys.exit("--ckpt is required for the foundationstereo backend")
        return FoundationStereoModel(ckpt, device, iters=iters)
    sys.exit(f"Unknown model: {name}. Choices: foundationstereo, dummy")


# =============================================================================
# Evaluation loop
# =============================================================================

def evaluate(dataset_root: str, model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    shift_buckets  = {}   # delta → list of (epe, d1)

    entries = list(find_shift_dirs(Path(dataset_root)))
    if not entries:
        sys.exit(f"No shift directories found under {dataset_root}")

    for scene, frame, delta, shift_dir in entries:
        try:
            left, right, gt_disp, meta = load_sample(shift_dir)
        except Exception as e:
            print(f"  SKIP {shift_dir}: {e}", file=sys.stderr)
            continue

        pred = model.predict(left, right)

        # Align spatial size if model resizes internally
        if pred.shape != gt_disp.shape:
            from PIL import Image as _PIL
            pred = np.array(
                _PIL.fromarray(pred).resize(
                    (gt_disp.shape[1], gt_disp.shape[0]), _PIL.BILINEAR
                )
            )

        # Save the prediction next to its shift directory for later inspection
        np.save(shift_dir / "pred_disparity.npy", pred.astype(np.float32))

        mask = valid_mask(gt_disp)
        e    = epe(pred, gt_disp, mask)
        d    = d1(pred, gt_disp, mask)

        per_frame_rows.append(dict(
            scene=scene, frame=frame, delta=delta,
            epe=e, d1=d,
            neg_pct=meta.get("pct_negative", float("nan")),
        ))
        shift_buckets.setdefault(delta, []).append((e, d))
        print(f"  {scene}/{frame}  Δ={delta:+d}  EPE={e:.3f}  D1={d:.2f}%  "
              f"pred=[{pred.min():.2f},{pred.max():.2f}]  "
              f"gt=[{gt_disp.min():.2f},{gt_disp.max():.2f}]")

    # Aggregate per shift
    deltas     = sorted(shift_buckets)
    agg_epe    = [np.nanmean([v[0] for v in shift_buckets[k]]) for k in deltas]
    agg_d1     = [np.nanmean([v[1] for v in shift_buckets[k]]) for k in deltas]

    per_shift_rows = [
        dict(delta=d, epe=e, d1=d1v)
        for d, e, d1v in zip(deltas, agg_epe, agg_d1)
    ]

    # Write CSVs
    _write_csv(out_dir / "per_frame.csv", per_frame_rows,
               ["scene", "frame", "delta", "epe", "d1", "neg_pct"])
    _write_csv(out_dir / "per_shift.csv", per_shift_rows,
               ["delta", "epe", "d1"])

    print(f"\nResults written to {out_dir}")
    print(f"\n{'Δ':>6}  {'EPE':>8}  {'D1%':>8}")
    print("-" * 28)
    for r in per_shift_rows:
        print(f"{r['delta']:>+6d}  {r['epe']:8.3f}  {r['d1']:8.2f}")

    if HAS_MPL:
        _plot_robustness(deltas, agg_epe, agg_d1, out_dir / "robustness.png")
    else:
        print("\n[WARN] matplotlib not installed — skipping plot")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plot_robustness(deltas, epe_vals, d1_vals, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(deltas, epe_vals, marker="o", color="steelblue")
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("ZDP shift Δ (px)")
    ax1.set_ylabel("EPE (px)")
    ax1.set_title("End-Point Error vs ZDP Shift")
    ax1.grid(True, alpha=0.3)

    ax2.plot(deltas, d1_vals, marker="s", color="tomato")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("ZDP shift Δ (px)")
    ax2.set_ylabel("D1 (%)")
    ax2.set_title("D1 Error vs ZDP Shift")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate stereo model across ZDP shifts.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory")
    parser.add_argument("--model",   required=True,
                        choices=["foundationstereo", "dummy"],
                        help="Model backend")
    parser.add_argument("--ckpt",    default=None,
                        help="Checkpoint path (required for foundationstereo)")
    parser.add_argument("--out",     default="results/",
                        help="Output directory for CSVs and plots")
    parser.add_argument("--device",  default="cuda",
                        help="Torch device (cuda / cpu)")
    parser.add_argument("--iters",   type=int, default=32,
                        help="GRU update iterations (foundationstereo only)")
    args = parser.parse_args()

    model = make_model(args.model, args.ckpt, args.device, iters=args.iters)
    evaluate(args.dataset, model, Path(args.out))


if __name__ == "__main__":
    main()
