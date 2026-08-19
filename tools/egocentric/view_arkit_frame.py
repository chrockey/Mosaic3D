"""viser viewer for one egocentric training sample — the visual correctness check.

Shows, for a single (scene, annotated frame):

  full scene   grey, the complete reconstruction the encoder used to see
  visible      what ``ARKitScenesFrameDataset`` actually returns for this frame
  masks        each caption's points in its own colour
  camera       the matched pose's frustum, drawn from the real intrinsics

If the geometry is right the coloured subset is exactly the part of the room in
front of the drawn frustum, with nothing from behind a wall, and each mask sits
on the object its caption names.

Run in the CPU venv:
    /home/jovyan/cholab/users/chunghyun/mosaic3d-scannet-eval/.venv-data/bin/python \
        tools/egocentric/view_arkit_frame.py --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.egocentric.arkit_frame_dataset import ARKitScenesFrameDataset  # noqa: E402
from src.data.egocentric.frustum import visible_indices  # noqa: E402

DATA = "/home/jovyan/cholab/datasets/mosaic3d/data/arkitscenes"
RAW = "/home/jovyan/cholab/datasets/mosaic3d/arkit_raw"

# Distinct hues; masks are re-coloured by index so neighbours never share one.
PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--point-size", type=float, default=0.015)
    args = ap.parse_args()

    import viser

    ds = ARKitScenesFrameDataset(
        data_dir=DATA, raw_dir=RAW, split="train", transforms=None,
        num_masks=None, min_score=args.min_score,
    )
    scenes = sorted({s for s, _ in ds.frames})
    print(f"{len(ds.frames)} frames over {len(scenes)} scenes")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    with server.gui.add_folder("sample"):
        g_scene = server.gui.add_dropdown("scene", options=scenes, initial_value=scenes[0])
        g_frame = server.gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
        g_info = server.gui.add_markdown("")
    with server.gui.add_folder("layers"):
        g_full = server.gui.add_checkbox("full scene", True)
        g_vis = server.gui.add_checkbox("visible (the sample)", True)
        g_masks = server.gui.add_checkbox("caption masks", True)
        g_cam = server.gui.add_checkbox("camera frustum", True)
        g_size = server.gui.add_slider("point size", min=0.002, max=0.05, step=0.002,
                                       initial_value=args.point_size)
    with server.gui.add_folder("captions"):
        g_caps = server.gui.add_markdown("")

    state = {"handles": []}

    def frames_of(scene):
        return sorted(g for s, g in ds.frames if s == scene)

    def redraw(_=None):
        for h in state["handles"]:
            h.remove()
        state["handles"] = []

        scene = g_scene.value
        groups = frames_of(scene)
        gi = int(np.clip(g_frame.value, 0, len(groups) - 1))
        group = groups[gi]

        d = os.path.join(DATA, scene)
        coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(d, "color.npy")).astype(np.uint8)
        z = np.load(os.path.join(d, f"frames.{ds.align_source}.npz"))
        pose = int(z["group_to_pose"][group])
        score = float(z["score"][group])
        tr = ds._track(scene)
        R, t, K, wh = (tr["R"][pose].astype(np.float64), tr["t"][pose].astype(np.float64),
                       tr["K"][pose].astype(np.float64), tr["wh"][pose])

        vis = visible_indices(coord, R, t, K, wh, z_far=ds.z_far)
        masks, caps = ds._group_masks(scene, group, ds.align_source)
        to_local = np.full(len(coord), -1, np.int64)
        to_local[vis] = np.arange(len(vis))

        ps = float(g_size.value)
        if g_full.value:
            state["handles"].append(server.scene.add_point_cloud(
                "/full", points=coord, colors=np.full_like(color, 165), point_size=ps * 0.6))
        if g_vis.value:
            state["handles"].append(server.scene.add_point_cloud(
                "/visible", points=coord[vis], colors=color[vis], point_size=ps))

        lines = [f"**pose** {pose}  |  **coverage** {score:.4f}  |  "
                 f"**visible** {len(vis)} / {len(coord)} pts", ""]
        kept = 0
        for m, (mi, cap) in enumerate(zip(masks, caps)):
            local = to_local[mi.astype(np.int64)]
            local = local[local >= 0]
            if len(local) < ds.min_mask_points:
                lines.append(f"- ~~{cap}~~ (only {len(local)} pts visible)")
                continue
            c = PALETTE[kept % len(PALETTE)]
            kept += 1
            if g_masks.value:
                state["handles"].append(server.scene.add_point_cloud(
                    f"/mask/{m}", points=coord[vis][local],
                    colors=np.tile(c, (len(local), 1)), point_size=ps * 1.6))
            lines.append(f"- <span style='color:rgb({c[0]},{c[1]},{c[2]})'>&#9632;</span> "
                         f"{cap}  *({len(local)} pts)*")

        if g_cam.value:
            # world <- camera, then a frustum with the real FOV and aspect
            R_c2w, t_c2w = R.T, -R.T @ t
            fy, h = float(K[1]), float(wh[1])
            state["handles"].append(server.scene.add_camera_frustum(
                "/cam", fov=2 * np.arctan(0.5 * h / fy),
                aspect=float(wh[0]) / float(wh[1]), scale=0.35, color=(255, 60, 60),
                wxyz=_mat_to_wxyz(R_c2w), position=t_c2w))

        g_info.content = (f"`{scene}` frame **{gi+1}/{len(groups)}** (group {group})  \n"
                          f"{kept}/{len(masks)} masks survive the view")
        g_caps.content = "\n".join(lines)

    def on_scene(_=None):
        g_frame.max = max(len(frames_of(g_scene.value)) - 1, 1)
        g_frame.value = 0
        redraw()

    g_scene.on_update(on_scene)
    for w in (g_frame, g_full, g_vis, g_masks, g_cam, g_size):
        w.on_update(redraw)
    on_scene()

    print(f"viser on http://{args.host}:{args.port}")
    import time
    while True:
        time.sleep(1)


def _mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), the order viser wants."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


if __name__ == "__main__":
    main()
