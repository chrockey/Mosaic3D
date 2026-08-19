"""Check the z-buffer frustum against real ARKitScenes sensor depth.

Training never reads the depth images; this tool downloads them for a single
video so the geometry can be pinned down once.  It answers two questions:

  1. is the pose convention right?  (compare projected depth to sensor depth)
  2. does the splatted z-buffer reproduce the visibility a real depth map gives?

Usage:
    python tools/egocentric/verify_arkit_frustum.py \
        --scene-dir  /home/jovyan/cholab/datasets/mosaic3d/data/arkitscenes/40753679 \
        --video-dir  /home/jovyan/cholab/datasets/mosaic3d/arkit_raw/40753679 \
        --depth-dir  /path/to/lowres_depth
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.egocentric.frustum import (  # noqa: E402
    load_arkit_track,
    project,
    visible_indices,
    visible_indices_from_depth,
)


def _depth_index(depth_dir: str):
    files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    ts = np.array([float(os.path.basename(f).split("_")[1][:-4]) for f in files])
    order = np.argsort(ts)
    return ts[order], [files[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="dir holding coord.npy / color.npy")
    ap.add_argument("--video-dir", required=True, help="dir holding lowres_wide.traj + intrinsics")
    ap.add_argument("--depth-dir", required=True, help="extracted lowres_depth/")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--splat", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--downscale", type=float, nargs="+", default=[1.0, 2.0])
    args = ap.parse_args()

    from PIL import Image

    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float64)
    track = load_arkit_track(args.video_dir)
    dts, dfiles = _depth_index(args.depth_dir)
    print(f"points={len(coord)}  poses={len(track)}  depth_frames={len(dts)}")

    sel = np.linspace(0, len(track) - 1, args.frames).astype(int)

    # --- 1. pose convention: projected depth vs sensor depth -----------------
    errs, fracs = [], []
    for i in sel:
        j = int(np.argmin(np.abs(dts - track.ts[i])))
        depth = np.asarray(Image.open(dfiles[j])).astype(np.float64) / 1000.0
        u, v, z = project(coord, track.R[i], track.t[i], track.K[i])
        w, h = int(track.wh[i][0]), int(track.wh[i][1])
        ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (z > 0.1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        d = depth[vi[ok], ui[ok]]
        m = d > 0.05
        if m.sum() < 100:
            continue
        e = np.abs(z[ok][m] - d[m])
        errs.append(np.median(e))
        fracs.append(float(np.mean(e < 0.15)))
    print(f"\n[pose] median |z_proj - z_sensor| over {len(errs)} frames: "
          f"{np.median(errs):.4f} m   frac<15cm: {np.mean(fracs):.3f}")

    # --- 2. z-buffer visibility vs depth-map visibility ----------------------
    print(f"\n[visibility] reference = real depth, 20% relative tolerance "
          f"(Mosaic3D geometry.py:32)")
    print(f"{'splat':>5} {'down':>5} {'IoU':>7} {'recall':>7} {'prec':>7} {'|V_zbuf|':>9} {'|V_dep|':>8}")
    for splat in args.splat:
        for down in args.downscale:
            ious, recs, precs, nz, nd = [], [], [], [], []
            for i in sel:
                j = int(np.argmin(np.abs(dts - track.ts[i])))
                depth = np.asarray(Image.open(dfiles[j])).astype(np.float64) / 1000.0
                ref = visible_indices_from_depth(
                    coord, track.R[i], track.t[i], track.K[i], depth)
                got = visible_indices(
                    coord, track.R[i], track.t[i], track.K[i], track.wh[i],
                    splat=splat, downscale=down)
                if len(ref) < 100:
                    continue
                sr, sg = np.zeros(len(coord), bool), np.zeros(len(coord), bool)
                sr[ref] = True
                sg[got] = True
                inter = np.count_nonzero(sr & sg)
                ious.append(inter / max(np.count_nonzero(sr | sg), 1))
                recs.append(inter / max(len(ref), 1))
                precs.append(inter / max(len(got), 1))
                nz.append(len(got))
                nd.append(len(ref))
            print(f"{splat:>5} {down:>5.1f} {np.mean(ious):>7.4f} {np.mean(recs):>7.4f} "
                  f"{np.mean(precs):>7.4f} {np.mean(nz):>9.0f} {np.mean(nd):>8.0f}")


if __name__ == "__main__":
    main()
