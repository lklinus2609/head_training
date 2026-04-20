"""Export an mp4 video of an inference sequence — prediction + GT side-by-side + audio.

Takes the three artifacts written by viewer/generate_sequence.py (pred .npy,
_gt.npy, .wav) and renders them as FLAME meshes with the source audio muxed in.
This is explicitly NOT run automatically after inference — invoke it only when
you want a video artifact from a specific run.

Two rendering backends are available:

  * `--backend pyrender` — GPU offscreen via pyrender. Needs OpenGL access:
    `PYOPENGL_PLATFORM=egl` on NVIDIA headless with DRI permissions, or `=osmesa`
    with OSMesa libraries installed.
  * `--backend cpu` — pure software rasterizer (matplotlib Agg + PolyCollection
    with painter's algorithm + backface cull). No OpenGL required. Slower but
    reliable on TACC compute nodes that block `/dev/dri/*`.
  * `--backend auto` (default) — try pyrender first, fall back to cpu on failure.

Typical invocation on a headless compute node:

    python viewer/export_video.py \
        --sequence viewer/static/sequences/test_sequence.npy \
        --output test_sequence.mp4
"""

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.flame_decoder import FLAMEDecoder
from viewer.export_flame_mesh import load_flame_model

DEFAULT_FLAME_PATH = os.path.expandvars("$WORK/models/flame/generic_model.pkl")


def _camera_params(template_verts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (centroid, cam_dist) used by both backends so their framing matches."""
    centroid = template_verts.mean(axis=0).astype(np.float32)
    extent = float((template_verts.max(axis=0) - template_verts.min(axis=0)).max())
    cam_dist = 2.5 * extent
    return centroid, cam_dist


def _build_scene(template_verts: np.ndarray, resolution: int):
    """Create a pyrender scene framed on the FLAME template bounding box."""
    import pyrender

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])
    centroid, cam_dist = _camera_params(template_verts)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)
    camera_pose = np.eye(4)
    camera_pose[0, 3] = centroid[0]
    camera_pose[1, 3] = centroid[1]
    camera_pose[2, 3] = centroid[2] + cam_dist
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(key_light, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[0, 3] = centroid[0] - cam_dist
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    return scene, renderer


def _render_frames_pyrender(expr: np.ndarray, decoder: FLAMEDecoder, faces: np.ndarray,
                            scene, renderer) -> np.ndarray:
    """Render a sequence of FLAME expression params to an RGB frame stack (GPU path)."""
    import pyrender
    import torch
    import trimesh

    T = expr.shape[0]
    frames = np.empty((T, renderer.viewport_height, renderer.viewport_width, 3),
                      dtype=np.uint8)
    with torch.no_grad():
        for i in range(T):
            params = torch.from_numpy(expr[i:i + 1]).float()
            verts = decoder.expression_to_vertices(params)[0].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            color, _ = renderer.render(scene)
            frames[i] = color
            scene.remove_node(node)
    return frames


def _render_frames_cpu(expr: np.ndarray, decoder: FLAMEDecoder, faces: np.ndarray,
                      template: np.ndarray, resolution: int) -> np.ndarray:
    """CPU rasterizer — matplotlib Agg + PolyCollection with painter's algorithm.

    No OpenGL required. Appropriate for headless compute nodes without DRI access.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import torch

    centroid, cam_dist = _camera_params(template)
    eye = centroid.copy()
    eye[2] += cam_dist
    fov_y = np.pi / 6.0
    focal = 1.0 / np.tan(fov_y / 2)
    W = H = resolution

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    T = expr.shape[0]
    frames = np.empty((T, H, W, 3), dtype=np.uint8)

    with torch.no_grad():
        for i in range(T):
            params = torch.from_numpy(expr[i:i + 1]).float()
            verts = decoder.expression_to_vertices(params)[0].cpu().numpy()

            # Perspective project: camera at (centroid + z*cam_dist) looking -Z
            v_cam = verts - eye
            z_safe = np.where(np.abs(v_cam[:, 2]) < 1e-6, -1e-6, v_cam[:, 2])
            x_pix = ((v_cam[:, 0] * focal / (-z_safe)) + 1.0) * 0.5 * W
            y_pix = (1.0 - ((v_cam[:, 1] * focal / (-z_safe)) + 1.0) * 0.5) * H
            z_eye = -v_cam[:, 2]

            # Face normals & Lambertian shade (light colocated with camera at +Z).
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            n_unit = n / n_norm
            visible = n_unit[:, 2] > 0
            shade = np.clip(n_unit[:, 2], 0.0, 1.0)

            # Painter's algorithm — draw far triangles first.
            z_centroid = (z_eye[faces[:, 0]] + z_eye[faces[:, 1]] + z_eye[faces[:, 2]]) / 3
            vis_idx = np.where(visible)[0]
            order = vis_idx[np.argsort(-z_centroid[vis_idx])]

            triangles = np.stack([
                np.stack([x_pix[faces[:, 0]], y_pix[faces[:, 0]]], axis=1),
                np.stack([x_pix[faces[:, 1]], y_pix[faces[:, 1]]], axis=1),
                np.stack([x_pix[faces[:, 2]], y_pix[faces[:, 2]]], axis=1),
            ], axis=1)[order]
            s = shade[order] * 0.75 + 0.2
            colors = np.stack([s, s, s], axis=1)

            for coll in list(ax.collections):
                coll.remove()
            pc = PolyCollection(triangles, facecolors=colors, edgecolors="none",
                                linewidths=0)
            ax.add_collection(pc)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames[i] = buf.copy()

    plt.close(fig)
    return frames


def _compose_and_label(pred_frames: np.ndarray, gt_frames: np.ndarray | None,
                       layout: str) -> np.ndarray:
    """Concatenate pred/GT panels along the chosen axis and overlay labels."""
    from PIL import Image, ImageDraw, ImageFont

    if layout == "pred_only" or gt_frames is None:
        composed = pred_frames
        labels: list[tuple[str, int]] = [("Prediction", 0)]
        stacked = False
    elif layout == "stacked":
        composed = np.concatenate([pred_frames, gt_frames], axis=1)
        labels = [("Prediction", 0), ("Ground Truth", pred_frames.shape[1])]
        stacked = True
    else:
        composed = np.concatenate([pred_frames, gt_frames], axis=2)
        labels = [("Prediction", 0), ("Ground Truth", pred_frames.shape[2])]
        stacked = False

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    out = np.empty_like(composed)
    for i in range(composed.shape[0]):
        img = Image.fromarray(composed[i])
        draw = ImageDraw.Draw(img)
        for text, offset in labels:
            xy = (8, offset + 8) if stacked else (offset + 8, 8)
            draw.text(xy, text, fill=(20, 20, 20), font=font,
                      stroke_width=2, stroke_fill=(255, 255, 255))
        out[i] = np.asarray(img)
    return out


def _write_video(frames: np.ndarray, path: Path, fps: int):
    import imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _mux_audio(silent_video: Path, audio: Path, output: Path):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", str(silent_video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Render an inference sequence as mp4 with GT panel and audio."
    )
    parser.add_argument("--sequence", required=True, type=str,
                        help="Path to predicted .npy (e.g. viewer/static/sequences/foo.npy). "
                             "Sibling {stem}_gt.npy and {stem}.wav are auto-discovered.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mp4 path (default: sibling {stem}.mp4).")
    parser.add_argument("--flame_path", type=str, default=DEFAULT_FLAME_PATH,
                        help="Path to FLAME generic_model.pkl.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Per-panel square resolution in pixels.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output framerate (default 30, matches dataset).")
    parser.add_argument("--layout", type=str, default="side_by_side",
                        choices=["side_by_side", "stacked", "pred_only"])
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "pyrender", "cpu"],
                        help="Render backend. 'auto' tries pyrender then falls back to cpu.")
    parser.add_argument("--no_audio", action="store_true",
                        help="Skip the audio mux even if a .wav is present.")
    args = parser.parse_args()

    seq_path = Path(args.sequence)
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_path}")

    stem = seq_path.with_suffix("").name
    gt_path = seq_path.with_name(f"{stem}_gt.npy")
    wav_path = seq_path.with_name(f"{stem}.wav")
    output_path = Path(args.output) if args.output else seq_path.with_suffix(".mp4")

    pred = np.load(str(seq_path)).astype(np.float32)
    gt = None
    layout = args.layout
    if layout != "pred_only":
        if gt_path.exists():
            gt = np.load(str(gt_path)).astype(np.float32)
            min_len = min(pred.shape[0], gt.shape[0])
            pred = pred[:min_len]
            gt = gt[:min_len]
        else:
            warnings.warn(f"No GT at {gt_path}; falling back to pred_only.", stacklevel=1)
            layout = "pred_only"

    print(f"Sequence: {seq_path}  shape={pred.shape}")
    if gt is not None:
        print(f"GT:       {gt_path}  shape={gt.shape}")
    print(f"Layout:   {layout}")
    print(f"Output:   {output_path}")

    flame_model = load_flame_model(args.flame_path)
    faces = np.array(flame_model["f"], dtype=np.int32)
    template = np.array(flame_model["v_template"], dtype=np.float32)

    decoder = FLAMEDecoder(args.flame_path, device="cpu")

    scene = None
    renderer = None
    backend = args.backend
    if backend in ("auto", "pyrender"):
        if backend == "auto" and os.environ.get("PYOPENGL_PLATFORM") is None:
            print("PYOPENGL_PLATFORM unset; skipping pyrender and going straight to cpu.")
            backend = "cpu"
        else:
            try:
                scene, renderer = _build_scene(template, args.resolution)
                backend = "pyrender"
            except Exception as e:
                if args.backend == "pyrender":
                    raise
                warnings.warn(
                    f"pyrender init failed ({type(e).__name__}: {e}); "
                    "falling back to cpu renderer.",
                    stacklevel=1,
                )
                backend = "cpu"

    print(f"Backend:  {backend}")

    try:
        print("Rendering prediction...")
        if backend == "pyrender":
            pred_frames = _render_frames_pyrender(pred, decoder, faces, scene, renderer)
        else:
            pred_frames = _render_frames_cpu(pred, decoder, faces, template, args.resolution)

        gt_frames = None
        if gt is not None:
            print("Rendering ground truth...")
            if backend == "pyrender":
                gt_frames = _render_frames_pyrender(gt, decoder, faces, scene, renderer)
            else:
                gt_frames = _render_frames_cpu(gt, decoder, faces, template, args.resolution)
    finally:
        if renderer is not None:
            renderer.delete()

    print("Compositing frames...")
    frames = _compose_and_label(pred_frames, gt_frames, layout)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_audio = (not args.no_audio) and wav_path.exists()
    if include_audio:
        silent_path = output_path.with_suffix(".silent.mp4")
        print(f"Writing silent video to {silent_path}...")
        _write_video(frames, silent_path, args.fps)
        print(f"Muxing audio from {wav_path}...")
        _mux_audio(silent_path, wav_path, output_path)
        silent_path.unlink()
    else:
        if not args.no_audio and not wav_path.exists():
            warnings.warn(f"No audio at {wav_path}; writing silent video.", stacklevel=1)
        print(f"Writing video to {output_path}...")
        _write_video(frames, output_path, args.fps)

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
