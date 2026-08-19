"""Frustum (single-view) restriction of a reconstructed scene point cloud.

Mosaic3D was trained on complete reconstructions.  Its annotations, however, are
already frame-restricted: the outer nesting level of ``point_indices.<src>.npz``
is the sampled RGB-D frame, and every stored mask is the set of *scene* vertices
that project into one 2D region of that frame and pass a depth-visibility test
(paper Eq. 1; the original implementation survives as dead code in
``src/data/utils/geometry.py``).

This module rebuilds the *geometric* half of that operation from the camera
trajectory alone, so a training sample can be the partial cloud a single camera
actually sees while the caption masks stay exact index subsets.  Working in the
scene's own index space is the point: mask restriction is a set intersection,
not a nearest-neighbour transfer.  Measured against real ARKitScenes sensor
depth, projecting the Mosaic3D ``coord.npy`` with the convention below agrees to
a median 2.9-4.7 cm (85-100 % of points within 15 cm).

Coordinate convention (verified empirically, see ``verify_arkit_frustum.py``):
one row of ``lowres_wide.traj`` is ``[timestamp, r_x, r_y, r_z, t_x, t_y, t_z]``
where ``r`` is an axis-angle giving **world -> camera** rotation and ``t`` the
matching translation, in **OpenCV** camera axes (x right, y down, z forward).
No axis flip is applied anywhere.  The world frame is gravity aligned with +z up.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "CameraTrack",
    "axis_angle_to_matrix",
    "load_arkit_traj",
    "load_arkit_intrinsics",
    "project",
    "visible_indices",
    "visible_indices_from_depth",
]


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues, on a single (3,) axis-angle vector."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass
class CameraTrack:
    """A video's poses and intrinsics, already paired frame by frame."""

    ts: np.ndarray  # (F,)      float64, seconds
    R: np.ndarray  # (F, 3, 3) float64, world -> camera
    t: np.ndarray  # (F, 3)    float64, world -> camera
    K: np.ndarray  # (F, 4)    float64, (fx, fy, cx, cy)
    wh: np.ndarray  # (F, 2)    int64,   (width, height)

    def __len__(self) -> int:
        return len(self.ts)


def load_arkit_traj(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ``lowres_wide.traj`` -> (ts, R_world2cam, t_world2cam)."""
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 7:
        raise ValueError(f"{path}: expected 7 columns, got {raw.shape[1]}")
    ts = raw[:, 0]
    R = np.stack([axis_angle_to_matrix(a) for a in raw[:, 1:4]], axis=0)
    return ts, R, raw[:, 4:7]


_PINCAM_RE = re.compile(r"_(\d+\.\d+)\.pincam$")


def load_arkit_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``lowres_wide_intrinsics.zip`` (or an extracted dir) without unpacking.

    The zip is kept compressed on disk on purpose: extracting every video would
    put ~14M tiny files on GPFS.  Returns (ts, K(fx,fy,cx,cy), wh) sorted by ts.
    """
    entries = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".pincam"):
                    entries.append((os.path.join(root, f), None))
        reader = lambda p, _z: np.loadtxt(p)  # noqa: E731
        zf = None
    else:
        zf = zipfile.ZipFile(path)
        entries = [(n, zf) for n in zf.namelist() if n.endswith(".pincam")]
        reader = lambda p, z: np.fromstring(z.read(p).decode(), sep=" ")  # noqa: E731

    ts, vals = [], []
    for name, z in entries:
        m = _PINCAM_RE.search(os.path.basename(name))
        if m is None:
            continue
        v = reader(name, z)
        if v.size != 6:
            continue
        ts.append(float(m.group(1)))
        vals.append(v)
    if zf is not None:
        zf.close()
    if not ts:
        raise ValueError(f"{path}: no .pincam entries")
    ts = np.asarray(ts)
    vals = np.asarray(vals)  # (M, 6) = w, h, fx, fy, cx, cy
    order = np.argsort(ts)
    ts, vals = ts[order], vals[order]
    return ts, vals[:, 2:6], vals[:, 0:2].astype(np.int64)


def load_arkit_track(video_dir: str) -> CameraTrack:
    """Pair every trajectory pose with its nearest-in-time intrinsics."""
    ts, R, t = load_arkit_traj(os.path.join(video_dir, "lowres_wide.traj"))
    izip = os.path.join(video_dir, "lowres_wide_intrinsics.zip")
    idir = os.path.join(video_dir, "lowres_wide_intrinsics")
    its, K, wh = load_arkit_intrinsics(izip if os.path.exists(izip) else idir)
    j = np.clip(np.searchsorted(its, ts), 0, len(its) - 1)
    j = np.where(
        (j > 0) & (np.abs(its[np.maximum(j - 1, 0)] - ts) < np.abs(its[j] - ts)), j - 1, j
    )
    return CameraTrack(ts=ts, R=R, t=t, K=K[j], wh=wh[j])


def project(
    coord: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, z) in one camera.  ``K`` is (fx, fy, cx, cy)."""
    cam = coord @ R.T + t  # (N, 3), world -> camera
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0] * cam[:, 0] / safe + K[2]
    v = K[1] * cam[:, 1] / safe + K[3]
    return u, v, z


def visible_indices(
    coord: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    wh: Tuple[int, int],
    z_near: float = 0.1,
    z_far: float = 8.0,
    splat: int = 1,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
    downscale: float = 1.0,
) -> np.ndarray:
    """Indices of ``coord`` a camera at (R, t, K) actually sees.

    A plain z-buffer over a point cloud lets background points leak through the
    gaps between foreground points, so each point is *splatted* over a
    (2*splat+1)^2 pixel neighbourhood when the buffer is built, while the
    visibility test itself reads only the point's own pixel.  ``downscale``
    shrinks the raster (and the intrinsics with it), which trades boundary
    precision for a denser buffer on sparse clouds.

    A point survives if it is inside the frustum and its depth is within
    ``rel_tol * z + abs_tol`` of the nearest surface splatted onto its pixel.
    The 5 cm / 5 % defaults are set by the reconstruction's own thickness: on
    ARKitScenes the Mosaic3D cloud sits a median 3.9 cm from the sensor surface.
    Against real depth they give IoU 0.883 (recall 0.974, precision 0.902) with
    ``splat=0, downscale=1.0``; looser values score higher only because they stop
    culling occlusion at all, and the reference itself has no-return holes.
    """
    w, h = int(wh[0]), int(wh[1])
    if downscale != 1.0:
        K = np.asarray(K, dtype=np.float64) / downscale
        w = max(1, int(round(w / downscale)))
        h = max(1, int(round(h / downscale)))

    u, v, z = project(coord, R, t, K)
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inside = (z > z_near) & (z < z_far) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not inside.any():
        return np.empty(0, dtype=np.int64)

    ui_in, vi_in, z_in = ui[inside], vi[inside], z[inside]
    # Scatter-min without np.minimum.at, which is ~2 orders of magnitude slower:
    # write the pixels in order of DECREASING depth, so the surviving value of
    # each pixel is the last (= smallest) one written.
    far_first = np.argsort(-z_in, kind="stable")
    zbuf = np.full(h * w, np.inf, dtype=np.float64)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            uu = ui_in[far_first] + du
            vv = vi_in[far_first] + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            zbuf[vv[ok] * w + uu[ok]] = z_in[far_first][ok]

    front = zbuf[vi_in * w + ui_in]
    keep = z_in <= front + rel_tol * z_in + abs_tol
    return np.flatnonzero(inside)[keep]


def visible_indices_from_depth(
    coord: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    rel_tol: float = 0.2,
    z_near: float = 0.1,
    z_far: float = 8.0,
) -> np.ndarray:
    """Reference visibility using a real depth map, i.e. Mosaic3D's own Eq. 1.

    ``depth`` is (H, W) in metres; zero means no return.  ``rel_tol=0.2`` is the
    20 % relative threshold from ``src/data/utils/geometry.py:32``.  Only used to
    validate :func:`visible_indices` — training never needs the depth images.
    """
    h, w = depth.shape
    Ks = np.asarray(K, dtype=np.float64).copy()
    u, v, z = project(coord, R, t, Ks)
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inside = (z > z_near) & (z < z_far) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not inside.any():
        return np.empty(0, dtype=np.int64)
    d = depth[vi[inside], ui[inside]]
    keep = (d > 0) & (np.abs(d - z[inside]) <= rel_tol * z[inside])
    return np.flatnonzero(inside)[keep]
