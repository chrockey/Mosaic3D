"""Recover which camera pose produced each Mosaic3D annotation group.

The release keeps the frame grouping (the outer nesting level of
``point_indices.<src>.npz``) but not the frame id, so the correspondence between
annotation group ``g`` and trajectory pose ``t`` has to be recovered.  Every
stored mask is already the set of scene vertices *visible in that frame*, so the
right pose is the one whose frustum contains essentially all of the group:

    score(g, t) = |U_g \\cap V_t| / |U_g|,     U_g = union of group g's masks

Two estimates are reported.  The free per-group ``argmax_t`` is not safe — adjacent
poses of a 10 Hz trajectory look almost identical and the margin over the
runner-up is ~0.015, so argmax flips between neighbours and can even cross them.
The line ``t = round(a*g + b)`` fitted over a grid of (a, b) pins the stride,
and a banded monotone DP around that line then picks the best *increasing*
assignment — groups are emitted in capture order, so an assignment that goes
backwards is wrong by construction.  ``group_to_pose`` in the output file is the
DP result; the free argmax is stored alongside only for diagnosis.

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


def score_matrix(coord, track, unions, **kw):
    """(G, F) containment scores, without ever holding a (F, N) visibility matrix.

    Materialising visibility for every pose costs len(track) x len(coord) bytes --
    ~1 GB for a large scene -- and 48 workers streaming that are bound by memory
    bandwidth, not cores.  Only |U_g n V_t| / |U_g| is ever needed, so the poses
    are streamed: one N-byte boolean at a time, scored against every group on the
    spot.  Same numbers, ~1000x less memory.

    Also returns the visible-point count per pose, for the frame-size filter.
    """
    G, F, N = len(unions), len(track), len(coord)
    S = np.zeros((G, F), dtype=np.float32)
    n_vis = np.zeros(F, dtype=np.int64)
    vis = np.zeros(N, dtype=bool)
    for i in range(F):
        idx = visible_indices(coord, track.R[i], track.t[i], track.K[i], track.wh[i], **kw)
        n_vis[i] = len(idx)
        vis[idx] = True
        for g, u in enumerate(unions):
            if len(u):
                S[g, i] = vis[u].mean()
        vis[idx] = False       # cheaper than reallocating an N-byte array per pose
    return S, n_vis


def monotone_align(S, fit_t, band=30):
    """Best strictly increasing assignment of groups to poses, inside a band.

    Independent per-group argmax is not safe here: neighbouring poses of a 10 Hz
    trajectory are near-identical and the margin over the runner-up is ~0.015, so
    argmax flips back and forth and can cross frames.  Annotation groups are
    emitted in capture order, so the assignment must be increasing; enforcing
    that turns a set of near-ties into one globally consistent path.  The band
    around the fitted line keeps it O(G x band) and stops a single bad group from
    dragging the path away.

        dp[g][f] = S[g][f] + max_{f' < f} dp[g-1][f']

    Returns (assignment, total score).
    """
    G, F = S.shape
    lo = np.clip(fit_t - band, 0, F - 1)
    hi = np.clip(fit_t + band + 1, 1, F)
    NEG = -1e18
    dp = np.full((G, F), NEG)
    back = np.zeros((G, F), dtype=np.int64)
    dp[0, lo[0]:hi[0]] = S[0, lo[0]:hi[0]]
    for g in range(1, G):
        run, arg = NEG, -1
        best = np.full(F, NEG)
        argbest = np.zeros(F, dtype=np.int64)
        for f in range(F):                      # prefix max over strictly smaller f
            best[f], argbest[f] = run, arg
            if dp[g - 1, f] > run:
                run, arg = dp[g - 1, f], f
        sl = slice(lo[g], hi[g])
        dp[g, sl] = S[g, sl] + best[sl]
        back[g, sl] = argbest[sl]
        dp[g, sl] = np.where(best[sl] <= NEG / 2, NEG, dp[g, sl])
    end = int(np.argmax(dp[G - 1]))
    if dp[G - 1, end] <= NEG / 2:
        return fit_t.copy(), float("nan")       # no feasible increasing path in band
    out = np.zeros(G, dtype=np.int64)
    out[G - 1] = end
    for g in range(G - 1, 0, -1):
        out[g - 1] = back[g, out[g]]
    return out, float(dp[G - 1, end])


def fit_stride(S, n_pose, n_group):
    """Grid-search t = round(a*g + b) against the score matrix; -> (score, a, b, t)."""
    a0 = n_pose / max(n_group, 1)
    rows = np.arange(n_group)
    best = None
    for a in np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.02):
        for b in np.arange(-1.5 * a0, 1.5 * a0 + 1e-9, 1.0):
            t = np.clip(np.round(a * rows + b).astype(int), 0, n_pose - 1)
            s = float(S[rows, t].mean())
            if best is None or s > best[0]:
                best = (s, float(a), float(b), t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--src", default="gsam2", choices=["gsam2", "seem"])
    ap.add_argument("--z-far", type=float, default=8.0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--refine-window", type=int, default=30,
                    help="band, in poses, around the fitted line for the monotone DP")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scene = os.path.basename(args.scene_dir.rstrip("/"))
    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float32)
    groups, _ = load_groups(args.scene_dir, args.src)
    track = load_arkit_track(args.video_dir)
    G, F = len(groups), len(track)

    unions = [np.unique(np.concatenate(g)) if len(g) else np.empty(0, np.int64) for g in groups]
    S, n_vis = score_matrix(coord, track, unions, z_far=args.z_far)
    free_t = S.argmax(1)
    free_s = S.max(1)
    fit_s, a, b, fit_t = fit_stride(S, F, G)

    ref_t, _ = monotone_align(S, fit_t, band=args.refine_window)
    ref_s = S[np.arange(G), ref_t]

    print(f"{scene}  N={len(coord)} groups={G} poses={F} poses/group={F/G:.2f} "
          f"| free={free_s.mean():.4f} fit={fit_s:.4f} refined={ref_s.mean():.4f} a={a:.2f} b={b:.1f} "
          f"| agree(|dt|<=2)={np.mean(np.abs(free_t - fit_t) <= 2):.3f} "
          f"vis/pose={n_vis.mean():.0f}")
    if not args.quiet:
        print(f"  free  : median={np.median(free_s):.4f} min={free_s.min():.4f} "
              f"frac>0.90={np.mean(free_s > 0.90):.3f} "
              f"margin-over-2nd={np.mean(free_s - np.sort(S,1)[:,-2]):.4f}")
        print(f"  refnd : median={np.median(ref_s):.4f} min={ref_s.min():.4f} "
              f"frac>0.90={np.mean(ref_s > 0.90):.3f} "
              f"|t_ref-t_fit| med={np.median(np.abs(ref_t - fit_t)):.1f} "
              f"increasing={bool(np.all(np.diff(ref_t) > 0))}")

    if args.write:
        out = os.path.join(args.scene_dir, f"frames.{args.src}.npz")
        np.savez(out, group_to_pose=ref_t.astype(np.int64), score=ref_s.astype(np.float32),
                 fit_pose=fit_t.astype(np.int64),
                 free_pose=free_t.astype(np.int64), free_score=free_s.astype(np.float32),
                 stride_a=np.float32(a), stride_b=np.float32(b),
                 n_visible=n_vis[ref_t].astype(np.int64))
        print("  wrote", out)


if __name__ == "__main__":
    main()
