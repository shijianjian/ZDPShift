"""
Emit multi_dataset/config.json — the minimal recipe to re-render the
dataset from its source .blend files.

Captures:
  - The CLI invocation
  - render/stereo/DOF rules
  - SHA-256 of every script involved
  - For each rendered scene: source .blend identity (sha256, size, mtime)
    + camera names + source URL (where the blend came from on download.blender.org)

Read-only with respect to per-frame data; relies on each scene's
reproduce.json for camera state.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from write_reproduce import sha256_of, blend_identity, script_versions


# Manual mapping of source URLs we used for the splash files.
# Local-only scenes (the user already had them in scenes/) have no URL.
SOURCE_URLS = {
    "blender-3.4-splash":  "https://download.blender.org/demo/splash/blender-3.4-splash.zip",
    "blender-3.5-splash":  "https://download.blender.org/demo/splash/blender-3.5-splash.blend",
    "blender-4.0-splash":  "https://download.blender.org/demo/splash/blender-4.0-splash.blend",
    "blender-4.1-splash":  "https://download.blender.org/demo/splash/blender-4.1-splash.blend",
    "blender-4.5-splash":  "https://download.blender.org/demo/splash/blender-4.5-splash.blend",
    "blender-5.1-splash":  "https://download.blender.org/demo/splash/blender-5.1-splash.blend",
    "splash-pokedstudio":  "https://download.blender.org/demo/test/splash-pokedstudio.blend.zip",
    "splash_fishy_cat":    "https://download.blender.org/demo/test/splash_fishy_cat_2.zip",
    "classroom":           "https://download.blender.org/demo/test/classroom.zip",
    "sky-texture-demo":    "https://download.blender.org/demo/test/sky-texture-demo.zip",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="multi_dataset")
    ap.add_argument("--scenes-dir", default="scenes")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--shifts", type=int, nargs="+", default=[-16, 0, 16, 24, 32])
    args = ap.parse_args()

    root = Path(args.root)
    scenes_dir = Path(args.scenes_dir)

    scene_records = []
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        repro_path = sd / "reproduce.json"
        if not repro_path.exists():
            continue
        repro = json.loads(repro_path.read_text())

        blend_rel = repro["source_blend"].get("path")
        blend_path = (Path(blend_rel) if blend_rel else None)
        if blend_path is not None and not blend_path.exists():
            blend_path = scenes_dir / f"{sd.name}.blend"

        timeline_frames = sorted({fr.get("timeline_frame") for fr in repro["frames"]
                                  if fr.get("timeline_frame") is not None})
        cameras = sorted({fr["camera"]["name"] for fr in repro["frames"]})
        scene_records.append(dict(
            tag=sd.name,
            source_blend=blend_identity(blend_path),
            source_url=SOURCE_URLS.get(sd.name),
            n_frames=len(repro["frames"]),
            timeline_frames=timeline_frames,
            cameras=cameras,
            reproduce_json=str((sd / "reproduce.json").relative_to(root)),
        ))

    cmd = (
        f"python3 multi_pose_render.py "
        f"--scenes-dir {args.scenes_dir} --out {args.root} "
        f"--width {args.width} --height {args.height} --samples {args.samples} "
        f"--shifts {' '.join(str(s) for s in args.shifts)}"
    )

    config = dict(
        schema_version=1,
        generated_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        command=cmd,
        render_config=dict(
            width=args.width, height=args.height, samples=args.samples,
            shifts=list(args.shifts), engine="BLENDER_EEVEE_NEXT",
        ),
        stereo_config=dict(
            baseline_rule="max(0.02, min(2.0, 0.05 * subject_distance_m))",
            subject_distance_rule=(
                "active_cam.data.dof.focus_distance, clamped to (0.5, 5000); "
                "default 10.0 m if outside"
            ),
            stereo_offset_rule=(
                "cam_L.shift_x = +delta/(2W);  cam_R.shift_x = -delta/(2W)  "
                "(off-axis sensor shift, parallel cameras, no convergence)"
            ),
            dof=dict(
                enforced_use_dof=True,
                aperture_fstop=64.0,
                aperture_blades=0,
                rationale="sub-pixel CoC -> effective pinhole at f/64",
            ),
            ortho_handling="ORTHO active cameras forced to PERSP in build_stereo_pair",
        ),
        scripts=script_versions(),
        scenes=scene_records,
    )

    out = root / "config.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {out}  ({len(scene_records)} scenes, "
          f"{sum(s['n_frames'] for s in scene_records)} frames)")


if __name__ == "__main__":
    main()
