"""Export an mp4 video of an inference sequence — prediction + GT side-by-side + audio.

Takes the three artifacts written by viewer/generate_sequence.py (pred .npy,
_gt.npy, .wav) and renders them as FLAME meshes with the source audio muxed in.
This is explicitly NOT run automatically after inference — invoke it only when
you want a video artifact from a specific run.

Typical invocation (Linux / Jetson — headless OpenGL):

    PYOPENGL_PLATFORM=egl python viewer/export_video.py \
        --sequence viewer/static/sequences/test_sequence.npy \
        --output test_sequence.mp4

If EGL is unavailable, try `PYOPENGL_PLATFORM=osmesa` (requires OSMesa install).
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


def _build_scene(template_verts: np.ndarray, resolution: int):
    """Create a pyrender scene framed on the FLAME template bounding box."""
    import pyrender

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])

    centroid = template_verts.mean(axis=0)
    extent = float((template_verts.max(axis=0) - template_verts.min(axis=0)).max())
    cam_dist = 2.5 * extent

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


def _render_frames(expr: np.ndarray, decoder: FLAMEDecoder, faces: np.ndarray,
                   scene, renderer) -> np.ndarray:
    """Render a sequence of FLAME expression params to an RGB frame stack."""
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
    parser.add_argument("--no_audio", action="store_true",
                        help="Skip the audio mux even if a .wav is present.")
    args = parser.parse_args()

    if os.environ.get("PYOPENGL_PLATFORM") is None:
        warnings.warn(
            "PYOPENGL_PLATFORM is unset. Offscreen rendering on Jetson/cluster usually "
            "needs PYOPENGL_PLATFORM=egl (or =osmesa as a fallback).",
            stacklevel=1,
        )

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

    scene, renderer = _build_scene(template, args.resolution)
    try:
        print("Rendering prediction...")
        pred_frames = _render_frames(pred, decoder, faces, scene, renderer)
        gt_frames = None
        if gt is not None:
            print("Rendering ground truth...")
            gt_frames = _render_frames(gt, decoder, faces, scene, renderer)
    finally:
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
