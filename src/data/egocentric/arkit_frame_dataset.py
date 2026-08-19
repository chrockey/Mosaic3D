"""ARKitScenes as single-view frames instead of complete reconstructions.

One sample is one annotated RGB-D frame: the subset of the scene's reconstructed
points that the camera saw at that moment, plus the captions of the regions
annotated in that frame.  Mosaic3D's own annotations already carry this
structure — the outer nesting level of ``point_indices.<src>.npz`` is the frame,
and the masks inside one frame are a disjoint 2D panoptic partition lifted to 3D
— so nothing has to be re-derived semantically.  Only the *geometric* half is
rebuilt here, from the camera trajectory (see ``frustum.visible_indices``).

Why restrict the reconstruction instead of unprojecting the real depth image:
the masks are stored as indices into ``coord.npy``, so restricting them to a
view is an exact set intersection.  Unprojecting depth creates new points with no
index, forcing a nearest-neighbour transfer whose error (a measured median
2.9-4.7 cm on ARKitScenes) lands at 1.5-2.5 voxels of a 2 cm grid, concentrated
on object boundaries.  Keeping the scene index space also keeps the cloud
gravity-aligned for free.

Frame-to-pose correspondence comes from ``frames.<src>.npz``, written by
``tools/egocentric/align_arkit_frames.py``; measured over 10 scenes it is a
uniform stride of 9.99-10.00 poses per annotation group (10 Hz trajectory,
1 Hz annotation) with a matched-frustum coverage of 0.93-0.99.

Differences from the reconstruction pipeline, all deliberate:

* the returned cloud is already shifted by the **full scene's** x/y bbox centre
  and z-min, so ``CenterShift(apply_z=True)`` must be dropped from the transform
  list.  Applied per frame it would set the lowest *visible* point to z=0, and a
  view of a tabletop has no floor in it — the height prior the encoder learned
  would be different for every frame.
* ``SphereCrop`` must be dropped.  It is a no-op at frame scale anyway
  (``point_max=250000`` never triggers), and cutting a sphere out of a frustum
  produces a shape that occurs neither in pretraining nor at deployment.
* masks are filtered by ``min_mask_points`` after restriction, since a caption
  whose object is 95 % out of frame is supervision noise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.dataset_base import AnnotatedDataset
from src.data.egocentric.frustum import visible_indices
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ARKitScenesFrameDataset(AnnotatedDataset):
    """Single-view frames of ARKitScenes.  ``CLASS_LABELS`` is empty: no GT semantics."""

    CLASS_LABELS: List[str] = []
    LOG_POSTFIX = "arkitscenes_frame"

    def __init__(
        self,
        data_dir: str,
        raw_dir: str,
        split: str,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
        anno_sources: Optional[List[str]] = None,
        align_source: str = "gsam2",
        min_score: float = 0.5,
        min_mask_points: int = 32,
        min_frame_points: int = 2000,
        max_frame_points: Optional[int] = None,
        pose_jitter: int = 0,
        fov_jitter: float = 0.0,
        z_far: float = 8.0,
        z_far_jitter: Optional[List[float]] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            repeat=repeat,
            ignore_label=ignore_label,
            transforms=transforms,
            num_masks=num_masks,
            anno_sources=anno_sources,
        )
        self.raw_dir = Path(raw_dir)
        self.align_source = align_source
        self.min_score = min_score
        self.min_mask_points = min_mask_points
        self.min_frame_points = min_frame_points
        self.max_frame_points = max_frame_points
        self.pose_jitter = pose_jitter
        self.fov_jitter = fov_jitter
        self.z_far = z_far
        self.z_far_jitter = z_far_jitter

        self._track_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.frames = self._build_index()
        log.info(
            f"{self.LOG_POSTFIX} | split={self.split} | "
            f"scenes={len(set(s for s, _ in self.frames))} | frames={len(self.frames)}"
        )

    # ------------------------------------------------------------------ index

    def _build_index(self):
        """(scene, group) pairs whose frame was matched to a pose confidently."""
        pairs, missing = [], 0
        for scene in self.scene_names:
            f = self.data_dir / scene / f"frames.{self.align_source}.npz"
            p = self.raw_dir / scene / "poses.npz"
            if not f.exists() or not p.exists():
                missing += 1
                continue
            z = np.load(f)
            keep = np.flatnonzero(
                (z["score"] >= self.min_score) & (z["n_visible"] >= self.min_frame_points)
            )
            pairs.extend((scene, int(g)) for g in keep)
        if missing:
            log.warning(f"{missing}/{len(self.scene_names)} scenes lack frames/poses npz")
        if not pairs:
            raise RuntimeError(
                "no frames indexed — run tools/egocentric/align_arkit_frames.py and "
                "tools/egocentric/prepare_arkit_tracks.py first"
            )
        return pairs

    def __len__(self):
        n = len(self.frames)
        return n * self.repeat if self.split == "train" else n

    # ------------------------------------------------------------------- io

    def _track(self, scene: str):
        tr = self._track_cache.get(scene)
        if tr is None:
            z = np.load(self.raw_dir / scene / "poses.npz")
            tr = {k: z[k] for k in ("R", "t", "K", "wh")}
            if len(self._track_cache) > 64:
                self._track_cache.clear()
            self._track_cache[scene] = tr
        return tr

    def _group_masks(self, scene: str, group: int, source: str):
        """Masks + captions of one annotated frame, as scene-level indices."""
        d = self.data_dir / scene
        z = np.load(d / f"point_indices.{source}.npz")
        outer, inner = z["outer_lengths"], z["inner_lengths"]
        if group >= len(outer):
            return [], []
        m0 = int(np.sum(outer[:group]))  # first mask of this group
        n_m = int(outer[group])
        p0 = int(np.sum(inner[:m0]))
        lens = inner[m0 : m0 + n_m]
        packed = z["packed"][p0 : p0 + int(lens.sum())]
        masks = np.split(packed, np.cumsum(lens)[:-1]) if n_m else []

        c = np.load(d / f"captions.{source}.npz", allow_pickle=True)
        c0 = int(np.sum(c["lengths"][:group]))
        caps = [str(x) for x in c["packed"][c0 : c0 + n_m]]
        return masks, caps

    # --------------------------------------------------------------- sample

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.frames)
        scene, group = self.frames[idx]
        d = self.data_dir / scene

        coord = np.load(d / "coord.npy").astype(np.float32)
        color = np.load(d / "color.npy")

        z = np.load(d / f"frames.{self.align_source}.npz")
        pose = int(z["group_to_pose"][group])
        tr = self._track(scene)
        F = len(tr["R"])
        if self.pose_jitter and self.is_train:
            pose += int(np.random.randint(-self.pose_jitter, self.pose_jitter + 1))
        pose = int(np.clip(pose, 0, F - 1))

        K = tr["K"][pose].astype(np.float64).copy()
        if self.fov_jitter and self.is_train:
            # Vary the virtual camera's focal length so the encoder is not tied to
            # the iPad's 256x192 @ f=213; robot cameras have a different FOV.
            K[:2] *= float(np.random.uniform(1.0 - self.fov_jitter, 1.0 + self.fov_jitter))
        z_far = self.z_far
        if self.z_far_jitter and self.is_train:
            z_far = float(np.random.uniform(*self.z_far_jitter))

        vis = visible_indices(
            coord, tr["R"][pose].astype(np.float64), tr["t"][pose].astype(np.float64),
            K, tr["wh"][pose], z_far=z_far,
        )
        if len(vis) < self.min_frame_points:
            return None  # point_collate_fn drops None samples
        if self.max_frame_points and len(vis) > self.max_frame_points:
            vis = np.sort(np.random.choice(vis, self.max_frame_points, replace=False))

        # Shift by the FULL scene's frame, not the frame's own, so "height above
        # the floor" means the same thing in every sample.
        shift = np.array(
            [(coord[:, 0].min() + coord[:, 0].max()) / 2,
             (coord[:, 1].min() + coord[:, 1].max()) / 2,
             coord[:, 2].min()], dtype=np.float32)

        data = dict(
            scene_name=f"{scene}#{group}",
            coord=coord[vis] - shift,
            color=color[vis],
            origin_idx=vis.astype(np.int64),
        )

        if self.is_train:
            to_local = np.full(len(coord), -1, dtype=np.int64)
            to_local[vis] = np.arange(len(vis), dtype=np.int64)
            source = str(np.random.choice(self.anno_sources))
            masks, caps = self._group_masks(scene, group, source)

            keep_idx, keep_cap = [], []
            for m, c in zip(masks, caps):
                local = to_local[m.astype(np.int64)]
                local = local[local >= 0]
                if len(local) >= self.min_mask_points:
                    keep_idx.append(torch.from_numpy(local).int())
                    keep_cap.append(c)
            if not keep_idx:
                return None  # a frame with no surviving caption would break collate
            if self.num_masks is not None and self.num_masks < len(keep_idx):
                sel = np.random.choice(len(keep_idx), self.num_masks, replace=False)
                keep_idx = [keep_idx[i] for i in sel]
                keep_cap = [keep_cap[i] for i in sel]
            data["caption_data"] = dict(idx=keep_idx, caption=keep_cap)

        return self.transforms(data)
