"""Condense each ARKitScenes video's trajectory + intrinsics into one poses.npz.

The raw form is a 56 KB text trajectory plus a zip of ~2.9k one-line .pincam
files.  Re-reading that per training sample is not viable, and unpacking the
zips would put ~14M tiny files on GPFS, so every video is condensed once into
``arkit_raw/<video_id>/poses.npz`` holding the poses already paired with their
nearest-in-time intrinsics:

    ts (F,) float64 | R (F,3,3) float32 | t (F,3) float32
    K  (F,4) float32 (fx, fy, cx, cy)   | wh (F,2) int32

~160 KB per video, ~0.8 GB for all 5071.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.egocentric.frustum import load_arkit_track  # noqa: E402


def one(video_dir: str, overwrite: bool = False) -> str:
    out = os.path.join(video_dir, "poses.npz")
    if os.path.exists(out) and not overwrite:
        return f"skip {os.path.basename(video_dir)}"
    tr = load_arkit_track(video_dir)
    np.savez(out, ts=tr.ts, R=tr.R.astype(np.float32), t=tr.t.astype(np.float32),
             K=tr.K.astype(np.float32), wh=tr.wh.astype(np.int32))
    return f"ok   {os.path.basename(video_dir)} F={len(tr)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/home/jovyan/cholab/datasets/mosaic3d/arkit_raw")
    ap.add_argument("--video-id", default=None, help="one video; default = all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    vids = [args.video_id] if args.video_id else sorted(os.listdir(args.raw_dir))
    for v in vids:
        d = os.path.join(args.raw_dir, v)
        if not os.path.isdir(d):
            continue
        try:
            print(one(d, args.overwrite), flush=True)
        except Exception:
            print(f"FAIL {v}: {traceback.format_exc(limit=1).strip()}", flush=True)


if __name__ == "__main__":
    main()
