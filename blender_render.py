"""
Multi-ZDP stereo dataset renderer (bpy-module driven, no Blender GUI).

For each desired pixel disparity shift Δ:
  1. Build a parallel stereo camera pair (cam_L, cam_R) from a base camera.
  2. Apply a symmetric off-axis sensor shift on each camera so that
        d'(x) = d(x) - Δ          (matches project.md §2)
        cam_L.shift_x = +Δ/(2·W)   cam_R.shift_x = -Δ/(2·W)
  3. Render left.png and right.png separately.
  4. Render a depth pass from cam_L via the compositor.
  5. Compute disparity   D'(x) = f·B / Z(x) - Δ   and save it.

Run:
  python3 blender_render.py --scene scenes/blender-3.5-splash.blend \\
                            --output demo_dataset/

Layout produced (matches project.md §5):
  output/
    shift_-16/  shift_-8/  shift_+0/  shift_+8/  shift_+16/
      left.png  right.png  disparity.npy  meta.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    sys.exit("This script needs the bpy Python module (Blender as a library).")


# =============================================================================
# Scene + camera setup
# =============================================================================

_GPU_DEVICES_INIT_DONE = False


def _init_cycles_gpu(prefer: str = "OPTIX"):
    """Configure Cycles to use the GPU (OptiX preferred, CUDA fallback).
    Idempotent — safe to call before each scene load.

    GPU is required: we refuse to silently fall back to CPU because a
    misconfigured environment (no driver, no device) would render at ~50x
    the cost without warning.
    """
    global _GPU_DEVICES_INIT_DONE
    if _GPU_DEVICES_INIT_DONE:
        return
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError) as e:
        raise RuntimeError(
            "Cycles addon preferences not available — cannot enforce GPU. "
            f"Underlying error: {e}"
        )
    chosen = None
    last_err = None
    for t in (prefer, "CUDA"):
        try:
            prefs.compute_device_type = t
            chosen = t
            break
        except (TypeError, RuntimeError) as e:
            last_err = e
            continue
    if chosen is None:
        raise RuntimeError(
            f"No GPU compute_device_type accepted by Cycles "
            f"(tried {prefer}, CUDA). Last error: {last_err}"
        )
    try:
        prefs.refresh_devices()
    except AttributeError:
        pass
    n_gpu = 0
    for d in prefs.devices:
        # Enable matching GPU devices, disable CPU (CPU+GPU mixed is slower
        # for our small frames since CPU starves the GPU).
        if d.type == chosen:
            d.use = True
            n_gpu += 1
        elif d.type == "CPU":
            d.use = False
    if n_gpu == 0:
        raise RuntimeError(
            f"No {chosen} GPU devices found. Available device types: "
            f"{sorted({d.type for d in prefs.devices})}. "
            "Install/update the GPU driver or run on a GPU host."
        )
    print(f"[cycles] compute_device_type={chosen}  enabled GPUs={n_gpu}")
    _GPU_DEVICES_INIT_DONE = True


def load_scene(blend_path: str):
    # Cycles GPU init happens lazily in configure_render — only Cycles scenes
    # need it, and we want EEVEE-only scenes to load fine on GPU-less hosts.
    # Enable Python driver / script auto-execution. Blender Studio's open-movie
    # files commonly drive node-tree values from the active camera's
    # matrix_world via PyDriver expressions (e.g. "point_to_camera" geo-nodes).
    # Without this, those drivers raise "restricted access" and the geometry
    # they steer stays in its default orientation — producing renders that
    # look like the main subject is missing.
    try:
        bpy.context.preferences.filepaths.use_scripts_auto_execute = True
    except (AttributeError, RuntimeError):
        pass
    bpy.ops.wm.open_mainfile(filepath=blend_path, use_scripts=True)
    scene = bpy.context.scene
    _force_load_images()
    return scene


def _force_load_images():
    """Force-load image data for library-linked textures that Blender 5.0.1
    lazy-loaded as headers-only. TEX_IMAGE nodes that reference these render
    as flat default color until the pixels are loaded. `img.reload()` is
    rejected on library-linked images; `img.update()` succeeds.
    """
    n = 0
    for img in bpy.data.images:
        if img.has_data or not img.filepath:
            continue
        try:
            img.update()
            if img.has_data:
                n += 1
        except Exception:
            pass
    if n:
        print(f"[force-load-images] loaded {n} library-linked textures")


_FX_NAME_PATTERNS = (
    "shard", "particle", "dust", "spark", "bubble", "swarm", "snow",
    "smoke", "fog", "mist", "spray", "splash", "fume", "fire", "flame",
    "ember", "ash", "trail_fx", "ice_belt",
)


def strip_render_particles(scene, verbose: bool = True):
    """Disable particles / volumetric / mesh-FX that corrupt the depth pass with
    per-instance scatterer depths. After this runs, the L/R images and the
    disparity GT are mutually consistent — both show the underlying solid
    geometry without the ambient FX overlay.

    Scope:
      - Particle systems: count → 0; instancer disabled for render.
      - Hair / fur systems (also a particle modifier): same.
      - Objects whose mesh type is VOLUME (Blender's Volume datablock — smoke,
        fog domains): hide_render = True.
      - Objects whose material chain contains Volume Scatter / Absorption
        nodes (procedural volumetrics in a regular mesh): hide_render = True.
      - Objects whose name matches a common FX naming convention (ice_shard,
        particle, dust, spark, bubble, swarm, snow, smoke, mist, etc.) — this
        catches mesh-based "particles" emitted by Geometry Nodes instancers,
        which don't register as classical particle systems. Singularity uses
        thousands of `GEO-ice_belt_outer_edge-ice_shard_*` meshes that are
        only reachable via name pattern.
    """
    n_particles, n_volumes, n_vol_mats, n_name_fx = 0, 0, 0, 0
    for obj in bpy.data.objects:
        # Volume datablocks (smoke / fluid / fog domains)
        if obj.type == "VOLUME":
            if not obj.hide_render:
                obj.hide_render = True
                n_volumes += 1
            continue
        # Particle / hair systems
        for ps in obj.particle_systems:
            try:
                ps.settings.count = 0
                n_particles += 1
            except (AttributeError, RuntimeError):
                pass
        if obj.particle_systems:
            try:
                obj.show_instancer_for_render = False
            except (AttributeError, RuntimeError):
                pass
        # Material-based volumetrics (Volume Scatter / Absorption nodes)
        hide_for_vol = False
        for slot in (obj.material_slots or []):
            m = slot.material
            if not m or not m.use_nodes or not m.node_tree:
                continue
            if any(n.type in {"VOLUME_SCATTER", "VOLUME_ABSORPTION", "PRINCIPLED_VOLUME"}
                   for n in m.node_tree.nodes):
                hide_for_vol = True
                break
        if hide_for_vol:
            if not obj.hide_render:
                obj.hide_render = True
                n_vol_mats += 1
            continue
        # FX naming convention: hide meshes whose name carries an FX tag
        lname = obj.name.lower()
        if any(pat in lname for pat in _FX_NAME_PATTERNS):
            if not obj.hide_render:
                obj.hide_render = True
                n_name_fx += 1
    if verbose:
        print(f"[strip-particles] disabled: particle-systems={n_particles} "
              f"volume-objects={n_volumes} volume-material-objs={n_vol_mats} "
              f"name-pattern-fx={n_name_fx}")


def configure_render(scene, width: int, height: int, samples: int,
                     cycles_samples: int = 32, cpu: bool = False):
    """Set up scene.render for our pipeline. Respects the artist's authored
    engine (Cycles or EEVEE). For Cycles, enables OIDN denoising — the
    artist's compositor (which we disable) often had its own denoiser, so
    we replace it with the native one to keep noise out of the final image.

    If `cpu=True`, Cycles is bound to CPU instead of GPU. Use this only when
    intentionally avoiding GPU contention (e.g. an audit pass running
    alongside an existing GPU render); CPU is ~50× slower than OPTIX.
    """
    r = scene.render
    authored = (r.engine or "").upper()
    if "CYCLES" in authored:
        r.engine = "CYCLES"
    elif "EEVEE" in authored:
        try:
            r.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            r.engine = "BLENDER_EEVEE"
    else:
        try:
            r.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            r.engine = "BLENDER_EEVEE"

    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    if hasattr(r.image_settings, "media_type"):
        try:
            r.image_settings.media_type = "IMAGE"
        except TypeError:
            pass
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    # Some older .blend files (e.g. blender-2.91 splash) ship with
    # use_file_extension=False so Blender writes "left" instead of "left.png"
    # and "_depth_" instead of "_depth_0001.exr"; force-on to fix that.
    r.use_file_extension = True
    r.use_sequencer = False
    r.use_compositing = False
    r.use_multiview = False

    if r.engine == "CYCLES":
        if not hasattr(scene, "cycles"):
            raise RuntimeError(
                "scene.cycles missing on a CYCLES-engine scene; cannot configure device."
            )
        if not cpu:
            # Enforce GPU. Raises if no OPTIX/CUDA device is available.
            _init_cycles_gpu()
        scene.cycles.samples = cycles_samples
        scene.cycles.use_denoising = True
        # OPTIX denoiser is ~2-5x faster than OIDN on RTX cards.
        try:
            scene.cycles.denoiser = "OPTIX"
        except TypeError:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        try:
            scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        except (AttributeError, TypeError):
            pass
        # Re-use BVH + scene data between frames in the same animation.
        # Big win for stride-rendered timelines (5-10x on geometry-heavy
        # scenes like Sprite Fright).
        try:
            scene.cycles.use_persistent_data = True
        except AttributeError:
            pass
        # Adaptive sampling (default-on in modern Blender, but be explicit).
        try:
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.02
            scene.cycles.adaptive_min_samples = 0  # auto
        except AttributeError:
            pass
        # Camera frustum culling: skip BVH traversal for geometry outside the
        # camera's view (with a small margin). Exact for visible pixels —
        # off-frustum geometry can't influence visible pixels in a primary
        # ray, and adaptive_threshold/denoise tolerate the small change in
        # indirect-light contribution from culled occluders. Big win on
        # Sprite Fright forest shots where most foliage is off-camera.
        # Per-object opt-in is also required (linked-lib objects default
        # to off); we walk the scene and set the flag where writable.
        try:
            scene.cycles.use_camera_cull = True
            scene.cycles.camera_cull_margin = 0.1
            n_culled = 0
            for obj in scene.objects:
                try:
                    obj.cycles.use_camera_cull = True
                    n_culled += 1
                except (AttributeError, RuntimeError):
                    # RuntimeError: linked from library and not overridable
                    pass
            print(f"[configure_render] camera_cull enabled on {n_culled} objects")
        except AttributeError:
            pass
        # Bind this scene's render to the chosen device.
        target = "CPU" if cpu else "GPU"
        scene.cycles.device = target
        if scene.cycles.device != target:
            raise RuntimeError(
                f"Failed to set scene.cycles.device={target!r} "
                f"(got {scene.cycles.device!r})."
            )
    elif hasattr(scene, "eevee"):
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass

    print(f"[configure_render] authored={authored}  -> using {r.engine}  "
          f"samples={cycles_samples if r.engine == 'CYCLES' else samples}")


def build_stereo_pair(base_cam_obj, baseline: float):
    """Duplicate base camera into cam_L / cam_R, separated along the base
    camera's world-space +X axis.

    Critical: open-movie shot cameras commonly carry a COPY_TRANSFORMS
    constraint (or a parent) that locks their world transform to a linked
    rig. If we leave those constraints on the duplicates, our local-space
    offset is silently overridden and we get *identical* L/R views.
    Strip parent + constraints on the duplicates and set their world
    transform directly so the baseline offset survives.

    Forces PERSPECTIVE projection. An orthographic camera produces a
    constant, depth-independent pixel disparity (B / pixel_width), making
    `D = f·B/Z` invalid.
    """
    scene = bpy.context.scene
    coll = scene.collection

    # Capture the actual world transform of the base camera *before* any
    # changes (the constraint stack is what matters; matrix_world reflects it).
    bpy.context.view_layer.update()
    base_world = base_cam_obj.matrix_world.copy()
    base_loc = base_world.translation.copy()
    base_right = (base_world.to_3x3() @ Vector((1.0, 0.0, 0.0)))
    base_right.normalize()

    def _dup(name):
        obj = base_cam_obj.copy()
        obj.data = base_cam_obj.data.copy()
        obj.name = name
        obj.data.name = name + "-data"
        coll.objects.link(obj)
        # Strip parent + constraints so our world transform takes effect
        obj.parent = None
        while obj.constraints:
            obj.constraints.remove(obj.constraints[0])
        # Clear animation_data — without this, a keyframed location action on
        # the base camera (Caminandes Llamigos uses this pattern) keeps
        # re-driving the duplicates' location every depsgraph evaluation,
        # overriding our manual baseline offset and silently producing
        # identical L/R renders. Spring/Settlers/Sprite_Fright cameras don't
        # have this bug because they're driven via parent + constraints
        # (already stripped above) rather than direct fcurve animation.
        if obj.animation_data is not None:
            obj.animation_data_clear()
        if obj.data.animation_data is not None:
            obj.data.animation_data_clear()
        return obj

    cam_L = _dup("STEREO-L")
    cam_R = _dup("STEREO-R")

    if base_cam_obj.data.type != "PERSP":
        print(f"[build_stereo_pair] WARNING: base camera '{base_cam_obj.name}' "
              f"is {base_cam_obj.data.type}; forcing PERSP for f·B/Z disparity.")
        for c in (cam_L, cam_R):
            c.data.type = "PERSP"
            if c.data.lens <= 0.0:
                c.data.lens = 50.0
            if c.data.sensor_width <= 0.0:
                c.data.sensor_width = 36.0

    base_focus = base_cam_obj.data.dof.focus_distance if base_cam_obj.data.dof.focus_distance > 0 else 10.0
    for c in (cam_L, cam_R):
        c.data.stereo.interocular_distance = 0.0
        c.data.stereo.convergence_distance = 1e9
        c.data.stereo.convergence_mode = "PARALLEL"
        c.data.shift_x = 0.0
        c.data.shift_y = 0.0
        c.data.dof.use_dof = True
        c.data.dof.aperture_fstop = 64.0
        c.data.dof.aperture_blades = 0
        c.data.dof.focus_object = None
        c.data.dof.focus_distance = base_focus

    # Build the desired world transform: same orientation as base, but
    # translated ±baseline/2 along base's world +X.
    L_world = base_world.copy()
    L_world.translation = base_loc - base_right * (baseline / 2.0)
    R_world = base_world.copy()
    R_world.translation = base_loc + base_right * (baseline / 2.0)
    cam_L.matrix_world = L_world
    cam_R.matrix_world = R_world

    bpy.context.view_layer.update()
    return cam_L, cam_R


def focal_px(cam_data, render_width: int) -> float:
    return cam_data.lens * render_width / cam_data.sensor_width


def get_scene_bbox():
    """World-space AABB across all visible mesh objects (8-corner enumeration)."""
    mins = [+math.inf] * 3
    maxs = [-math.inf] * 3
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        mw = obj.matrix_world
        for v in obj.bound_box:
            wv = mw @ Vector(v)
            for i in range(3):
                if wv[i] < mins[i]:
                    mins[i] = wv[i]
                if wv[i] > maxs[i]:
                    maxs[i] = wv[i]
    return mins, maxs


def collect_scene_meta(cam_L, cam_R, baseline, f_px, width, height):
    """Snapshot enough info for downstream visualisation + reconstruction.

    Records full pinhole intrinsics for the unshifted (Δ=0) view:
      K = [[f_px, 0, cx], [0, f_py, cy], [0, 0, 1]]
    where cx=W/2, cy=H/2 (no sensor shift). Per-shift intrinsics with the Δ
    offset baked into cx are emitted by render_shift into each
    shift's meta.json (see `K_L`, `K_R`).
    """
    bbox_min, bbox_max = get_scene_bbox()
    sensor_w = cam_L.data.sensor_width
    sensor_h = cam_L.data.sensor_height
    sensor_fit = str(cam_L.data.sensor_fit)
    lens = cam_L.data.lens
    fov_h = 2.0 * math.degrees(math.atan(sensor_w / (2.0 * lens)))
    fov_v = 2.0 * math.degrees(math.atan(sensor_w * (height / width) / (2.0 * lens)))

    # Square-pixel intrinsics: f_py = f_px because pixel_aspect_ratio=1
    # and the Blender camera defaults to sensor_fit=HORIZONTAL.
    K_unshifted = [
        [float(f_px), 0.0,         width / 2.0],
        [0.0,         float(f_px), height / 2.0],
        [0.0,         0.0,         1.0],
    ]

    def mat_to_list(M):
        return [list(row) for row in M]

    return dict(
        cam_L=dict(
            name=cam_L.name,
            location=list(cam_L.location),
            matrix_world=mat_to_list(cam_L.matrix_world),
        ),
        cam_R=dict(
            name=cam_R.name,
            location=list(cam_R.location),
            matrix_world=mat_to_list(cam_R.matrix_world),
        ),
        intrinsics=dict(
            K=K_unshifted,
            f_px=float(f_px),
            f_py=float(f_px),
            cx=width / 2.0,
            cy=height / 2.0,
            focal_mm=float(lens),
            sensor_width_mm=float(sensor_w),
            sensor_height_mm=float(sensor_h),
            sensor_fit=sensor_fit,
            resolution_x=int(width),
            resolution_y=int(height),
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            note=("K is the Δ=0 intrinsic. Per-Δ K_L/K_R live in each "
                  "shift_<Δ>/meta.json with the sensor-shift principal-"
                  "point offset baked into cx."),
        ),
        baseline_m=baseline,
        focal_px=f_px,
        focal_mm=lens,
        sensor_width_mm=sensor_w,
        fov_h_deg=fov_h,
        fov_v_deg=fov_v,
        render_w=width,
        render_h=height,
        scene_bbox_min=list(bbox_min),
        scene_bbox_max=list(bbox_max),
    )


# =============================================================================
# Compositor for depth output
# =============================================================================

_COMPOSITOR_GROUP_NAME = "_zdp_depth_compositor"


def setup_depth_compositor(scene, exr_basename: str, output_dir: Path):
    """Wire compositor so that rendering writes Z-depth to <output_dir>/<exr_basename>####.exr."""
    scene.view_layers[0].use_pass_z = True

    # In Blender 5.x the compositor lives in a CompositorNodeTree assigned to
    # scene.compositing_node_group. Reuse / recreate our private group.
    existing = bpy.data.node_groups.get(_COMPOSITOR_GROUP_NAME)
    if existing is not None:
        bpy.data.node_groups.remove(existing)
    tree = bpy.data.node_groups.new(_COMPOSITOR_GROUP_NAME, "CompositorNodeTree")
    scene.compositing_node_group = tree
    scene.render.use_compositing = True   # has to be on for the OutputFile to write

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = scene.view_layers[0].name

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(output_dir)
    fo.file_name = exr_basename
    # Node-level format only supports OPEN_EXR_MULTILAYER in Blender 5.x;
    # individual single-channel EXRs come from per-item formats below.
    fo.format.file_format = "OPEN_EXR_MULTILAYER"

    # In Blender 5.x the file output node uses file_output_items (typed sockets).
    # We add a single FLOAT item which exposes a NodeSocketFloat input.
    fo.file_output_items.clear()
    item = fo.file_output_items.new("FLOAT", "depth")
    item.override_node_format = True
    item.format.file_format = "OPEN_EXR"
    item.format.color_depth = "32"

    # Z output is named "Depth" (Blender 5.x); fall back to "Z" if not present
    out = rl.outputs.get("Depth") or rl.outputs.get("Z")
    if out is None:
        raise RuntimeError("Render Layers node has no Depth/Z output.")
    # The first non-virtual input on the OutputFile node corresponds to our item
    target_input = next(i for i in fo.inputs if i.bl_idname != "NodeSocketVirtual")
    tree.links.new(out, target_input)

    return fo  # caller may inspect / clean up


def disable_compositor(scene):
    scene.compositing_node_group = None
    scene.render.use_compositing = False
    g = bpy.data.node_groups.get(_COMPOSITOR_GROUP_NAME)
    if g is not None:
        bpy.data.node_groups.remove(g)


# =============================================================================
# Render helpers
# =============================================================================

def render_view(scene, cam_obj, output_path: Path):
    """Render a single view from cam_obj to output_path (PNG).

    Disabling multi-view is not sufficient when the scene's render config has
    views_format == STEREO_3D (e.g. Caminandes Llamigos): Blender's stereoscopy
    pipeline overrides the camera world position with the camera's own
    ``data.stereo.interocular_distance`` offset. Since we set interocular = 0
    on the duplicates, both cam_L and cam_R render at the same world point and
    L == R. Switching views_format to INDIVIDUAL pulls the render onto the
    camera's actual matrix_world.
    """
    scene.camera = cam_obj
    scene.render.use_multiview = False
    # Caminandes Llamigos scenes save with views_format == STEREO_3D and a
    # populated scene.render.views collection (two entries with camera_suffix
    # "_L"/"_R") that overrides the rendered viewpoint with the camera's stereo
    # offset. Even with use_multiview = False this leaks through. Switch
    # to MULTIVIEW and clear the per-view camera_suffix so Cycles uses
    # scene.camera directly (and respects its actual matrix_world).
    scene.render.views_format = "MULTIVIEW"
    for v in scene.render.views:
        v.camera_suffix = ""
    scene.render.filepath = str(output_path.with_suffix(""))   # Blender appends ext
    bpy.ops.render.render(write_still=True)


def render_with_depth(scene, cam_obj, image_path: Path, depth_dir: Path, depth_stem: str):
    """Render cam_obj to image_path AND write a depth EXR via the compositor."""
    setup_depth_compositor(scene, depth_stem, depth_dir)
    render_view(scene, cam_obj, image_path)
    disable_compositor(scene)


def find_compositor_exr(depth_dir: Path, depth_stem: str) -> Path:
    """Compositor output prepends the slot name and frame number — find the actual file."""
    matches = sorted(depth_dir.glob(f"{depth_stem}*.exr"))
    if not matches:
        raise FileNotFoundError(f"No EXR matching {depth_stem}*.exr in {depth_dir}")
    return matches[-1]


# =============================================================================
# Depth EXR → NumPy
# =============================================================================

def load_depth_exr(path: Path) -> np.ndarray:
    """Load single-channel float32 depth from an EXR file."""
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        # The compositor writes a single-channel image; channel name varies
        chans = f.header()["channels"].keys()
        ch = next((c for c in ("V", "Z", "R", "Y") if c in chans), next(iter(chans)))
        raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
    except ImportError:
        pass

    # Fallback via imageio (OpenEXR plugin)
    import imageio.v3 as iio
    arr = iio.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


# =============================================================================
# Per-shift loop
# =============================================================================

class InvalidDepthError(RuntimeError):
    """Signals that the depth pass for a shift contained depth ≤ 0 / NaN.

    Typically from shadow-catcher / holdout / matte shaders that contribute
    visible color but write 0 to the depth pass. The render outputs
    (left.png, right.png, disparity.npy with whatever the math produced) are
    preserved on disk so the user can inspect them; this exception only
    informs the caller that the shift's GT is not trustworthy. The user
    decides whether to keep or reject the scene.
    """


def render_shift(
    scene,
    cam_L,
    cam_R,
    delta: int,
    out_dir: Path,
    f_px: float,
    baseline: float,
    width: int,
    *,
    height: int,
):
    """Render one Δ for one (frame, camera): cam_L color+depth + cam_R color,
    write disparity (`f·B/Z − Δ`) and per-Δ pinhole intrinsics with the
    sensor-shift principal point baked in.

    `cam.shift_x = ±Δ/(2W)` is applied per-render so each Δ is an independent
    Blender render. Do NOT replace this with a single wide render + crop:
    Cycles is Monte Carlo and color values would diverge from a true per-shift
    render even at matching seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Off-axis sensor shift (sensor-width units)
    s = (delta / 2.0) / width
    cam_L.data.shift_x = +s
    cam_R.data.shift_x = -s

    left_png = out_dir / "left.png"
    right_png = out_dir / "right.png"
    depth_dir = out_dir
    depth_stem = "_depth_"

    # Left view + depth (single render, depth captured via compositor)
    render_with_depth(scene, cam_L, left_png, depth_dir, depth_stem)

    # Right view (no depth needed)
    render_view(scene, cam_R, right_png)

    # Read depth and compute disparity
    exr_path = find_compositor_exr(depth_dir, depth_stem)
    depth = load_depth_exr(exr_path)

    # Depth validity check (informational only; we never delete renders).
    # depth <= 0 / NaN typically come from shadow-catcher / holdout shaders.
    # We always save left, right, disparity.npy so the user can inspect.
    invalid_mask = ~np.isfinite(depth) | (depth <= 0)
    n_invalid = int(invalid_mask.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        disp_base = (f_px * baseline / depth).astype(np.float32)
    disp = (disp_base - float(delta)).astype(np.float32)
    # For pixels with non-finite / non-positive depth the disparity is
    # undefined — clamp to 0 so downstream consumers see a single sentinel
    # rather than NaN/inf landmines.
    disp[invalid_mask] = 0.0
    finite_disp = disp[np.isfinite(disp) & (depth > 0) & (depth < 1e6)]

    np.save(out_dir / "disparity.npy", disp)

    # Per-Δ pinhole intrinsics. cam_L.shift_x = +Δ/(2W) shifts the
    # principal-point column LEFT by Δ/2 pixels in the rendered image
    # (empirically verified); cam_R is the mirror.
    cx_L = width / 2.0 - delta / 2.0
    cx_R = width / 2.0 + delta / 2.0
    cy = height / 2.0
    K_L = [[float(f_px), 0.0, cx_L], [0.0, float(f_px), cy], [0.0, 0.0, 1.0]]
    K_R = [[float(f_px), 0.0, cx_R], [0.0, float(f_px), cy], [0.0, 0.0, 1.0]]

    meta = dict(
        delta_px      = delta,
        focal_px      = f_px,
        baseline_m    = baseline,
        zdp_m         = (f_px * baseline / delta) if delta != 0 else None,
        disp_min      = float(finite_disp.min()) if finite_disp.size else 0.0,
        disp_max      = float(finite_disp.max()) if finite_disp.size else 0.0,
        pct_negative  = float((finite_disp < 0).mean() * 100) if finite_disp.size else 0.0,
        pct_positive  = float((finite_disp > 0).mean() * 100) if finite_disp.size else 0.0,
        K_L           = K_L,
        K_R           = K_R,
        resolution_x  = int(width),
        resolution_y  = int(height),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Clean up intermediate depth EXR
    try:
        exr_path.unlink()
    except FileNotFoundError:
        pass

    zdp_str = "inf" if delta == 0 else f"{meta['zdp_m']:.2f}m"
    print(
        f"  Δ={delta:+d}  "
        f"ZDP={zdp_str}  "
        f"disp=[{meta['disp_min']:6.2f}, {meta['disp_max']:6.2f}]  "
        f"neg={meta['pct_negative']:5.1f}%  pos={meta['pct_positive']:5.1f}%"
    )

    # All output is on disk before we surface the invalid-depth signal —
    # the caller may log the warning, but never deletes anything.
    if n_invalid > 0:
        pct = 100.0 * n_invalid / int(invalid_mask.size)
        raise InvalidDepthError(
            f"depth pass at Δ={delta:+d} has {n_invalid}/{int(invalid_mask.size)} "
            f"({pct:.2f}%) invalid pixels (depth ≤ 0 or NaN); "
            f"shadow-catcher / holdout shader detected — output preserved"
        )

    return meta


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="Path to .blend file")
    p.add_argument("--output", required=True, help="Output dataset directory")
    p.add_argument("--camera", default=None, help="Name of base camera (defaults to scene's active camera)")
    p.add_argument("--baseline", type=float, default=0.1, help="Stereo baseline in metres (default: 0.1)")
    p.add_argument("--width", type=int, default=480, help="Render width (default: 480)")
    p.add_argument("--height", type=int, default=320, help="Render height (default: 320)")
    p.add_argument("--samples", type=int, default=16, help="EEVEE TAA samples (default: 16)")
    p.add_argument("--shifts", type=int, nargs="+", default=[-16, -8, 0, 8, 16],
                   help="Δ pixel shifts (default: -16 -8 0 8 16)")
    args = p.parse_args()

    t0 = time.time()

    scene = load_scene(args.scene)
    configure_render(scene, args.width, args.height, args.samples)

    base_cam = bpy.data.objects[args.camera] if args.camera else scene.camera
    if base_cam is None or base_cam.type != "CAMERA":
        sys.exit("No active camera found and --camera not provided.")

    cam_L, cam_R = build_stereo_pair(base_cam, args.baseline)
    f_px = focal_px(cam_L.data, args.width)

    print(f"Scene        : {args.scene}")
    print(f"Base camera  : {base_cam.name}  ({tuple(round(v, 2) for v in base_cam.location)})")
    print(f"Render       : {args.width}×{args.height}  EEVEE  samples={args.samples}")
    print(f"Baseline     : {args.baseline} m")
    print(f"Focal        : {f_px:.1f} px")
    print(f"Shifts       : {args.shifts}")
    print(f"Output       : {args.output}\n")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_meta = collect_scene_meta(cam_L, cam_R, args.baseline, f_px, args.width, args.height)
    scene_meta["shifts"] = list(args.shifts)
    with open(out_root / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)
    bb_min = scene_meta["scene_bbox_min"]
    bb_max = scene_meta["scene_bbox_max"]
    print(f"Scene bbox   : x[{bb_min[0]:.1f},{bb_max[0]:.1f}] "
          f"y[{bb_min[1]:.1f},{bb_max[1]:.1f}] z[{bb_min[2]:.1f},{bb_max[2]:.1f}]\n")

    for delta in args.shifts:
        shift_dir = out_root / f"shift_{delta:+d}"
        render_shift(scene, cam_L, cam_R, delta, shift_dir,
                     f_px, args.baseline, args.width, height=args.height)

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
