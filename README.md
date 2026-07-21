<div align="center">

# ZDPShift: Beyond the Zero-Disparity Plane in Stereo

**Signed stereo matching for content on both sides of the screen.**

[Paper](#) · [arXiv](#) · [Project page](#) · [Dataset](#) · [Weights (HF)](#)

<img src="assets/teaser.png" width="90%">

</div>

Modern stereo matchers fail by **19–62×** when disparity becomes negative — yet
stereoscopic content, from cinema 3D to VR, lives in exactly that regime. This
repository releases:

1. **ZDPShift**, a calibrated open-movie stereo dataset rendering 4,405 frames at
   five zero-disparity-plane shifts each, with analytical *signed* ground truth.
2. A **signed cost volume** — a parameter-free generalization of the one-sided
   volume every cost-volume matcher relies on — that lets IGEV-Stereo and
   FoundationStereo express negative disparity. Pretrained checkpoints load verbatim.
3. Fine-tuning recipes and evaluation for RAFT-Stereo, IGEV-Stereo, and
   FoundationStereo across the full signed-disparity regime.

## Install

The three backbones and SEA-RAFT (for the in-the-wild verifier) are git
submodules under `third_party/` — clone recursively:

```bash
git clone --recursive <this-repo-url> zdpshift
cd zdpshift
# (or, if already cloned:  git submodule update --init --depth 1)

python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

| Submodule | Path | Upstream |
|-----------|------|----------|
| FoundationStereo | `third_party/FoundationStereo`     | NVlabs/FoundationStereo |
| IGEV-Stereo      | `third_party/IGEV/IGEV-Stereo`     | gangweiX/IGEV |
| RAFT-Stereo      | `third_party/RAFT-Stereo`          | princeton-vl/RAFT-Stereo |
| SEA-RAFT         | `third_party/SEA-RAFT`             | princeton-vl/SEA-RAFT |

Paths resolve automatically; override via `env.example` only if you relocate them.
Each backbone's **SceneFlow init checkpoint** (for training) and FoundationStereo's
**pretrained weights** (`third_party/FoundationStereo/pretrained_models/…`, for the
zero-shot baseline) are downloaded from the respective upstream repos.

## Weights

Fine-tuned ZDPShift checkpoints go in `weights/` (hosted on Hugging Face; see
[`weights/README.md`](weights/README.md)):

| File | Backbone | Route | mean EPE |
|------|----------|-------|----------|
| `weights/raft_zdpshift.pth`        | RAFT-Stereo      | data recipe          | 0.97 px |
| `weights/igev_zdpshift_signed.pth` | IGEV-Stereo      | + signed cost volume | 0.92 px |
| `weights/fs_zdpshift_signed.pth`   | FoundationStereo | + signed cost volume | **0.76 px** |

## Quick start (shipped samples)

The repo ships a tiny sample set so you can run without the full dataset:

```bash
# (1) Evaluate on 5 sample frames (one scene across all five ZDP shifts)
python eval_foundation_stereo.py --dataset dataset/eval \
    --ckpt weights/fs_zdpshift_signed.pth --signed-volume --d-neg 64 --d-pos 192 --out eval_fs

# (2) In-the-wild inference: 2 real 3D-movie frames (zero-shot FS vs ours)
python demo_real3d_compare.py dataset/inference demo   # writes demo.png
```

```
dataset/eval/…/frame_1031_00/shift_{-16,+0,+16,+24,+32}/   left.png right.png disparity.npy meta.json
dataset/inference/                                          {jurassic_world,finding_dory}_{L,R}.png
```

## Dataset (full)

Generate ZDPShift from Blender open movies, or download the released renders:

```bash
python download_scenes.py                 # fetch open-movie source .blend scenes
python build_config.py                    # write per-scene render configs
python blender_render.py --config <cfg>   # Cycles render at Δ ∈ {-16,0,+16,+24,+32}
```

Each pair carries left/right RGB and analytical signed disparity `d(Z,Δ)=fB/Z−Δ`.
Layout: `<root>/<split>/<scene>/frame_xxxxx/shift_±N/{left.png,right.png,disparity.npy,meta.json}`.

## Training

```bash
# RAFT-Stereo — data recipe is the whole method (its correlation is already symmetric)
python train_raft_stereo.py --train-root <zdpshift>/train --ckpt-init <raft_sceneflow.pth> \
    --out out_raft --rehearsal-root <sceneflow>

# IGEV-Stereo / FoundationStereo — add the signed cost volume
python train_igev_stereo.py --train-root <zdpshift>/train --ckpt-init <igev_sceneflow.pth> \
    --out out_igev --signed-volume --d-neg 64 --d-pos 192 --rehearsal-root <sceneflow>
python train_foundation_stereo.py --train-root <zdpshift>/train --ckpt-init <fs.pth> \
    --out out_fs --signed-volume --d-neg 64 --d-pos 192 --rehearsal-root <sceneflow>
```

Single RTX 3090 (24 GB), fp16, 30–50k iters (3–10 h).

## Evaluation

```bash
# Full ZDPShift test split
python eval_foundation_stereo.py --dataset <zdpshift>/test --ckpt weights/fs_zdpshift_signed.pth \
    --signed-volume --d-neg 64 --d-pos 192 --out eval_fs
python eval_igev_signed.py       --dataset <zdpshift>/test --ckpt weights/igev_zdpshift_signed.pth \
    --d-neg 64 --d-pos 192 --out eval_igev
python eval_raft_stereo.py       --dataset <zdpshift>/test --ckpt weights/raft_zdpshift.pth --out eval_raft

# KITTI-2015 no-regression check (positive regime, disjoint from our data)
python eval_kitti_fs.py          # KITTI at $SEARAFT_ROOT/datasets/KITTI/training
```

## In the wild: real 3D movies

```bash
# rank side-by-side clips by behind-screen content, then compare zero-shot vs ours
export SBS_VIDEO_DIR=<dir of SBS .mp4>
python rank_videos.py <clip_id> ...
python scan_negative_frames.py <clip.mp4> out/ --squeeze
python real3d_order_check.py  <clip.mp4> --squeeze          # verify L/R eye order vs monocular depth
python demo_real3d_compare.py out/ demo --frames F1,F2
```

The L/R order check uses Depth-Anything V2 (vendored `depth_anything_wrapper.py`, via `transformers`).

## Citation

```bibtex
@inproceedings{zdpshift2027,
  title     = {ZDPShift: Beyond the Zero-Disparity Plane in Stereo},
  author    = {Anonymous Authors},
  booktitle = {Under review at ICLR},
  year      = {2027}
}
```

## License

Code released under the [MIT License](LICENSE). ZDPShift renders derive from
Blender Studio open movies (CC-BY); film frames in figures/samples are shown for
research illustration only.
