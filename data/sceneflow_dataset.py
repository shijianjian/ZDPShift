"""Sceneflow stereo dataset (FlyingThings3D / Driving / Monkaa) for RAFT-Stereo
fine-tune rehearsal.

Mirrors the interface of `data/zdp_dataset.py`:
    __getitem__ -> (L_tensor, R_tensor, disp_tensor, meta_dict)
    - L/R: float32 [3, H, W] in 0..255
    - disp: float32 [H, W], positive (Sceneflow GT is left-camera disparity in px)
    - meta: dict with at least focal_px, baseline_m=1.0, delta_px=0,
            source="sceneflow"

The dataset pairs samples by walking the disparity .pfm tree and deriving the
RGB paths by substituting "disparity" -> "frames_<pass_kind>" and ".pfm" ->
".png" (with .webp fallback). This is robust to partially-extracted downloads:
a sample is only enumerated when the LEFT disparity .pfm exists.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# PFM reader (standard implementation, handles Sceneflow disparity .pfm files)
# ---------------------------------------------------------------------------
def read_pfm(path) -> np.ndarray:
    """Read a .pfm file. Returns float32 array, shape (H, W) or (H, W, 3)."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dims = f.readline().decode("latin-1")
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1")
        w, h = map(int, dims.split())

        scale = float(f.readline().decode("latin-1").rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (h, w, 3) if color else (h, w)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM is bottom-up
        if scale != 1.0:
            data = data * scale
        return np.ascontiguousarray(data)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------
_SUBSETS = ("FlyingThings3D", "Driving", "Monkaa")


def _infer_focal_px(rgb_path: Path, default: float = 1050.0) -> float:
    """Heuristic for Sceneflow focal length (in pixels at native 960x540).

    Driving has both 15mm and 35mm variants in the path; FlyingThings3D and
    Monkaa use 35mm-equivalent. Used only as metadata for per-delta loss
    weighting downstream — not as a calibration parameter (baseline_m=1.0
    means GT disparity in px is what consumers actually use).
    """
    p = str(rgb_path)
    if "15mm_focallength" in p:
        return 450.0
    return default


def _derive_rgb_pair(disp_pfm: Path, subset_root: Path, pass_kind: str):
    """Given a left-disparity .pfm path, return (left_rgb, right_rgb) Paths
    or None if neither .png nor .webp exists for both sides."""
    rel = disp_pfm.relative_to(subset_root)  # e.g. disparity/TRAIN/A/0000/left/0006.pfm
    parts = list(rel.parts)
    # Replace top-level "disparity" with frames_<pass_kind>
    if parts[0] != "disparity":
        return None
    parts[0] = f"frames_{pass_kind}"
    rgb_base = subset_root.joinpath(*parts).with_suffix("")  # drop .pfm

    left_rgb = _resolve_image(rgb_base)
    if left_rgb is None:
        return None

    # /left/ -> /right/ (last occurrence, since 'left' is a fixed subdir name)
    right_parts = list(parts)
    # walk from the end and flip the first "left" to "right"
    flipped = False
    for i in range(len(right_parts) - 1, -1, -1):
        if right_parts[i] == "left":
            right_parts[i] = "right"
            flipped = True
            break
    if not flipped:
        return None
    right_base = subset_root.joinpath(*right_parts).with_suffix("")
    right_rgb = _resolve_image(right_base)
    if right_rgb is None:
        return None
    return left_rgb, right_rgb


def _resolve_image(stem_path: Path) -> Optional[Path]:
    for ext in (".png", ".webp", ".jpg", ".jpeg"):
        cand = stem_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _scan_subset(subset_root: Path, split: str, pass_kind: str) -> list[tuple]:
    """Walk a subset's disparity tree and return list of (L_path, R_path,
    disp_path, focal_px) tuples for available samples.

    `split` is "train" or "test"; only FlyingThings3D distinguishes these
    via TRAIN/TEST directories. For other subsets, "test" returns [].
    """
    disp_root = subset_root / "disparity"
    if not disp_root.exists():
        return []

    name = subset_root.name
    if name == "FlyingThings3D":
        wanted = "TRAIN" if split == "train" else "TEST"
        search_root = disp_root / wanted
        if not search_root.exists():
            return []
    else:
        # Driving and Monkaa have no test split
        if split != "train":
            return []
        search_root = disp_root

    out: list[tuple] = []
    for pfm in search_root.rglob("*.pfm"):
        # Sceneflow stores left disparity under .../left/*.pfm
        if pfm.parent.name != "left":
            continue
        pair = _derive_rgb_pair(pfm, subset_root, pass_kind)
        if pair is None:
            continue
        L, R = pair
        out.append((L, R, pfm, _infer_focal_px(L)))
    out.sort(key=lambda t: str(t[2]))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SceneflowStereoDataset(Dataset):
    """PyTorch Dataset for Sceneflow (FlyingThings3D + Driving + Monkaa).

    Args:
        root: path to datasets/sceneflow/
        subset: "all" | "FlyingThings3D" | "Driving" | "Monkaa"
        split: "train" | "test"; only FlyingThings3D has TEST, others are
               train-only (handled gracefully — empty contribution).
        crop: (H, W) random crop for training; None for full-res
        augment: mild brightness/saturation jitter applied IDENTICALLY to L
                 and R (no horizontal flip — would swap L/R semantics)
        pass_kind: "finalpass" or "cleanpass"
    """

    def __init__(
        self,
        root,
        subset: str = "all",
        split: str = "train",
        crop: Optional[tuple[int, int]] = (384, 512),
        augment: bool = False,
        pass_kind: str = "finalpass",
    ):
        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.crop = crop
        self.augment = augment
        self.pass_kind = pass_kind

        if subset == "all":
            subset_names = _SUBSETS
        else:
            if subset not in _SUBSETS:
                raise ValueError(
                    f"subset must be one of {('all',) + _SUBSETS}, got {subset!r}"
                )
            subset_names = (subset,)

        samples: list[tuple] = []
        for name in subset_names:
            sub_root = self.root / name
            if not sub_root.exists():
                continue
            samples.extend(_scan_subset(sub_root, split, pass_kind))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        L_path, R_path, disp_path, focal_px = self.samples[idx]

        L = np.asarray(Image.open(L_path).convert("RGB"))
        R = np.asarray(Image.open(R_path).convert("RGB"))
        disp = read_pfm(disp_path)
        if disp.ndim == 3:
            # defensive: collapse stray color dim
            disp = disp[..., 0]
        # Sceneflow PFM can contain inf for invalid pixels (none documented,
        # but cheap to scrub). Replace inf -> 0 so downstream loss masks see
        # finite values; the typical disparity range is ~0..400.
        if not np.isfinite(disp).all():
            disp = np.where(np.isfinite(disp), disp, 0.0)

        if self.crop is not None:
            ch, cw = self.crop
            H, W = L.shape[:2]
            if H > ch and W > cw:
                y = random.randint(0, H - ch)
                x = random.randint(0, W - cw)
                L = L[y : y + ch, x : x + cw]
                R = R[y : y + ch, x : x + cw]
                disp = disp[y : y + ch, x : x + cw]

        if self.augment:
            # Identical jitter on both eyes to preserve stereo photometric
            # consistency. Mild ranges: brightness in [0.8,1.2], saturation
            # in [0.8,1.2]. Applied as a single multiplicative gain on RGB
            # plus a per-pixel saturation interpolation toward grayscale.
            b = 0.8 + 0.4 * random.random()
            s = 0.8 + 0.4 * random.random()
            L = _photometric(L, b, s)
            R = _photometric(R, b, s)

        L_t = torch.from_numpy(np.ascontiguousarray(L).astype(np.float32)).permute(2, 0, 1)
        R_t = torch.from_numpy(np.ascontiguousarray(R).astype(np.float32)).permute(2, 0, 1)
        disp_t = torch.from_numpy(np.ascontiguousarray(disp).astype(np.float32))

        meta = {
            "focal_px": float(focal_px),
            "baseline_m": 1.0,  # Sceneflow virtual baseline = 1 Blender unit
            "delta_px": 0,
            "source": "sceneflow",
            "subset": disp_path.parents[-1].name if False else _which_subset(disp_path, self.root),
            "pass_kind": self.pass_kind,
            "left_path": str(L_path),
            "right_path": str(R_path),
            "disp_path": str(disp_path),
        }
        return L_t, R_t, disp_t, meta


def _which_subset(disp_path: Path, root: Path) -> str:
    try:
        rel = disp_path.relative_to(root)
        return rel.parts[0]
    except ValueError:
        return ""


def _photometric(img: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
    """Identical brightness + saturation jitter for one stereo pair side.

    img: uint8 HxWx3. Returns uint8 HxWx3.
    """
    x = img.astype(np.float32)
    # brightness: multiplicative gain
    x = x * brightness
    # saturation: lerp toward grayscale luminance
    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = gray + saturation * (x - gray)
    return np.clip(x, 0, 255).astype(np.uint8)
