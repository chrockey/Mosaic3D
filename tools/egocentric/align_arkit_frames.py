"""Recover which camera pose produced each Mosaic3D annotation group.

The release keeps the frame grouping (the outer nesting level of
``point_indices.<src>.npz``) but not the frame id, so the correspondence between
annotation group ``g`` and trajectory pose ``t`` has to be recovered.  Every
stored mask is already the set of scene vertices *visible in that frame*, so the
right pose is the one whose frustum contains essentially all of the group:

    score(g, t) = |U_g \\cap V_t| / |U_g|,     U_g = union of group g's masks

Two estimates are reported.  The free per-group ``argmax_t`` is noisy — adjacent
poses of a 10 Hz trajectory look almost identical, and the margin over the
runner-up is ~0.01.  The constrained fit ``t = round(a*g + b)`` over a grid of
(a, b) is what should actually be used: annotations were produced at a fixed
frame stride, so two parameters describe the whole video and neighbouring-pose
noise averages out.

The recovered mapping doubles as an end-to-end check of the geometry: with a
wrong pose convention, intrinsics pairing or z-buffer, no pose would cover a
group and the score would collapse.

Run with OMP_NUM_THREADS=1 — the projection is a (N,3)@(3,3) matmul and threaded
BLAS makes it 10x slower.

Writes ``frames.<src>.npz`` into the scene dir.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.egocentric.frustum import load_arkit_track, visible_indices  # noqa: E402


def load_groups(scene_dir: str, src: str):
    """-> (per-group list of mask index arrays, per-group list of captions)."""
    z = np.load(os.path.join(scene_dir, f"point_indices.{src}.npz"))
    masks = np.split(z["packed"], np.cumsum(z["inner_lengths"])[:-1])
    groups, k = [], 0
    for n in z["outer_lengths"]:
        groups.append(masks[k : k + n])
        k += n
    c = np.load(os.path.join(scene_dir, f"captions.{src}.npz"), allow_pickle=True)
    caps, k = [], 0
    for n in c["lengths"]:
        caps.append([str(x) for x in c["packed"][k : k + n]])
        k += n
    return groups, caps


def visibility_matrix(coord, track, **kw):
    V = np.zeros((len(track), len(coord)), dtype=bool)
    for i in range(len(track)):
        V[i, visible_indices(coord, track.R[i], track.t[i], track.K[i], track.wh[i], **kw)] = True
    return V


def fit_stride(unions, V, n_pose, n_group):
    """Grid-search t = round(a*g + b); returns (a, b, mapping, mean score)."""
    a0 = n_pose / max(n_group, 1)
    best = None
    for a in np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.02):
        for b in np.arange(-1.5 * a0, 1.5 * a0 + 1e-9, 1.0):
            t = np.clip(np.round(a * np.arange(n_group) + b).astype(int), 0, n_pose - 1)
            s = np.mean([V[t[g], u].mean() if len(u) else 0.0 for g, u in enumerate(unions)])
            if best is None or s > best[0]:
                best = (float(s), float(a), float(b), t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--src", default="gsam2", choices=["gsam2", "seem"])
    ap.add_argument("--z-far", type=float, default=8.0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--refine-window", type=int, default=5,
                    help="poses either side of the fitted line to search")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scene = os.path.basename(args.scene_dir.rstrip("/"))
    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float32)
    groups, _ = load_groups(args.scene_dir, args.src)
    track = load_arkit_track(args.video_dir)
    G, F = len(groups), len(track)

    V = visibility_matrix(coord, track, z_far=args.z_far)
    unions = [np.unique(np.concatenate(g)) if len(g) else np.empty(0, np.int64) for g in groups]

    S = np.stack([V[:, u].mean(axis=1) if len(u) else np.zeros(F) for u in unions])  # (G, F)
    free_t = S.argmax(1)
    free_s = S.max(1)
    fit_s, a, b, fit_t = fit_stride(unions, V, F, G)

    # Refine inside a window around the fitted line: keeps the per-group accuracy
    # of the free argmax while dropping the gross outliers a 10 Hz trajectory's
    # near-identical neighbouring poses produce.
    w = args.refine_window
    lo = np.clip(fit_t - w, 0, F - 1)
    ref_t = np.array([lo[g] + int(np.argmax(S[g, lo[g]:min(lo[g] + 2 * w + 1, F)]))
                      for g in range(G)])
    ref_s = S[np.arange(G), ref_t]

    print(f"{scene}  N={len(coord)} groups={G} poses={F} poses/group={F/G:.2f} "
          f"| free={free_s.mean():.4f} fit={fit_s:.4f} refined={ref_s.mean():.4f} a={a:.2f} b={b:.1f} "
          f"| agree(|dt|<=2)={np.mean(np.abs(free_t - fit_t) <= 2):.3f} "
          f"vis/pose={V.sum(1).mean():.0f}")
    if not args.quiet:
        print(f"  free  : median={np.median(free_s):.4f} min={free_s.min():.4f} "
              f"frac>0.90={np.mean(free_s > 0.90):.3f} "
              f"margin-over-2nd={np.mean(free_s - np.sort(S,1)[:,-2]):.4f}")
        print(f"  refnd : median={np.median(ref_s):.4f} min={ref_s.min():.4f} "
              f"frac>0.90={np.mean(ref_s > 0.90):.3f} "
              f"|t_ref - t_fit| median={np.median(np.abs(ref_t - fit_t)):.1f}")

    if args.write:
        out = os.path.join(args.scene_dir, f"frames.{args.src}.npz")
        np.savez(out, group_to_pose=ref_t.astype(np.int64), score=ref_s.astype(np.float32),
                 fit_pose=fit_t.astype(np.int64),
                 free_pose=free_t.astype(np.int64), free_score=free_s.astype(np.float32),
                 stride_a=np.float32(a), stride_b=np.float32(b),
                 n_visible=V.sum(1)[ref_t].astype(np.int64))
        print("  wrote", out)


if __name__ == "__main__":
    main()
