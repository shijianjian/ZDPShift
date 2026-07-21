"""Multi-ZDP stereo dataset. Walks `shift_*/` dirs and yields (L, R, disp, meta).

Reuses the path-walking pattern from evaluate.py:find_shift_dirs.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _scan(root: Path, keep_deltas: Optional[set[int]] = None) -> list[Path]:
    """Return every shift_*/ dir under root (any depth: scene/frame/shift, frame/shift, shift).

    keep_deltas: if not None, only keep dirs whose Δ value (parsed from shift_NN name)
                  is in the set. Used for the ablation −R2 (no ZDP-shift augmentation).
    """
    out: list[Path] = []
    for shift_dir in root.rglob("shift_*"):
        if not shift_dir.is_dir():
            continue
        if not (shift_dir / "left.png").exists():
            continue
        if not (shift_dir / "right.png").exists():
            continue
        if not (shift_dir / "disparity.npy").exists():
            continue
        if keep_deltas is not None:
            try:
                d = int(shift_dir.name.replace("shift_", ""))
            except ValueError:
                continue
            if d not in keep_deltas:
                continue
        out.append(shift_dir)
    return sorted(out)


class ZdpStereoDataset(Dataset):
    def __init__(self, root, crop: Optional[tuple[int, int]] = (384, 512),
                 augment: bool = False, val_indices: Optional[set[int]] = None,
                 mode: str = "all", keep_deltas: Optional[set[int]] = None,
                 train_fraction: float = 1.0, seed: int = 42):
        """
        mode: 'all' | 'train' | 'val'
        val_indices: set of dataset indices to treat as val (used with mode='train' or 'val').
        keep_deltas: ablation filter; e.g. {0} keeps only Δ=0 shifts (−R2 ablation).
        train_fraction: sample-efficiency study; e.g. 0.5 keeps a random 50% of all scenes
                        (applied at scene level, not at shift level, so all 5 Δs of a kept
                        scene are kept). Only meaningful when mode='train' or 'all'.
        """
        self.root = Path(root)
        self.crop = crop
        self.augment = augment
        all_dirs = _scan(self.root, keep_deltas=keep_deltas)
        if train_fraction < 1.0 and mode in ("all", "train"):
            # Pick a fraction of unique scenes (parent.parent of shift_*/) deterministically
            scenes = sorted({sd.parent.parent for sd in all_dirs})
            rng = random.Random(seed)
            rng.shuffle(scenes)
            keep_scenes = set(scenes[: int(len(scenes) * train_fraction)])
            all_dirs = [sd for sd in all_dirs if sd.parent.parent in keep_scenes]
            print(f"train_fraction={train_fraction}: kept {len(keep_scenes)}/{len(scenes)} scenes, {len(all_dirs)} pairs")
        if val_indices is None:
            self.dirs = all_dirs
        else:
            if mode == "train":
                self.dirs = [p for i, p in enumerate(all_dirs) if i not in val_indices]
            elif mode == "val":
                self.dirs = [p for i, p in enumerate(all_dirs) if i in val_indices]
            else:
                self.dirs = all_dirs

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        sd = self.dirs[idx]
        L = np.asarray(Image.open(sd / "left.png").convert("RGB"))
        R = np.asarray(Image.open(sd / "right.png").convert("RGB"))
        disp = np.load(sd / "disparity.npy")
        with open(sd / "meta.json") as f:
            meta = json.load(f)

        if self.crop is not None:
            ch, cw = self.crop
            H, W = L.shape[:2]
            if H > ch and W > cw:
                y = random.randint(0, H - ch)
                x = random.randint(0, W - cw)
                L = L[y:y + ch, x:x + cw]
                R = R[y:y + ch, x:x + cw]
                disp = disp[y:y + ch, x:x + cw]

        if self.augment:
            # Mild brightness jitter ONLY; no spatial / sign-changing aug
            jitter = 0.8 + 0.4 * random.random()
            L = np.clip(L.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
            R = np.clip(R.astype(np.float32) * jitter, 0, 255).astype(np.uint8)

        L_t = torch.from_numpy(L.astype(np.float32)).permute(2, 0, 1)
        R_t = torch.from_numpy(R.astype(np.float32)).permute(2, 0, 1)
        disp_t = torch.from_numpy(disp.astype(np.float32))
        return L_t, R_t, disp_t, meta


def make_val_split(n: int, frac: float = 0.1, seed: int = 42) -> set[int]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return set(indices[: int(n * frac)])
