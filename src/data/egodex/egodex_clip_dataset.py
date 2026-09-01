"""EgoDex human demonstrations as single-view egocentric clips.

One sample is one CLIP of one episode: the points a head-mounted camera saw over an 11-frame
window, at the window's first frame, plus the captions of the instances annotated in it.  This is
the corpus for post-training the released Mosaic3D encoder on egocentric clouds -- the limitation
it targets is that spunet101 was pretrained on complete room reconstructions and is deployed on
what one robot camera sees.

WHAT THE CORPUS ACTUALLY IS (measured 2026-09-01 over 420 clips sampled across all seven parts,
0 read failures; per-episode file pair listed in the manifest, see `manifest` below):

    points per clip      mean  6,622   med  6,198   p95 11,029   max 12,000
    labelled fraction    mean  0.880   med  0.888   p05  0.783
    captions per clip    mean  10.05   med     10   p95     16   max     24
    caption words        mean   30.8   med   37.3   p95   41.6
    clips per episode    mean  15.18   med     10   p95     45   max    137
    => 323,980 episodes x 15.18 = ~4.92M clips, 2.7x the released pretraining exposure per epoch

FOUR THINGS DIFFER FROM ARKitScenesFrameDataset, all forced by the corpus:

* +y IS UP, NOT +z.  The clouds are in a world frame whose floor is y=0, not a camera frame (the
  h5 docstring upstream says camera frame; it is wrong).  Measured on two episodes: the head
  camera sits at y=1.403 / y=1.201, the right hand below it at y=1.292 / y=0.926, and a clip that
  sees the floor bottoms out at y=-0.023.  So this dataset applies R_x(+90 deg), (x,y,z) ->
  (x,-z,y), and everything downstream -- RandomRotate axis="z", CenterShift(apply_z=false) --
  keeps the meaning it has for every other dataset here.

* ABSOLUTE HEIGHT IS ALREADY COMPARABLE across episodes, because the frame is a world frame with
  the floor at 0.  So no z re-centring: `CenterShift(apply_z=false)` only, exactly as for ARKit
  but for the opposite reason (ARKit's frames are pre-shifted; these are already absolute).

* NORMALS ARE A ZEROS PLACEHOLDER in the flow h5s (`camera_0_scene_normals` exists and is
  identically zero -- verified on 4 clips of 2 episodes).  Harmless: `in_channel: 3`, colour only.

* CAPTIONS COME AS STRINGS, not as the precomputed `global_recap768` vectors.
  src/models/losses/caption_loss.py:41 accepts either, and the string path is the one that
  produced the released spunet101.ckpt: it dedupes on text and encodes only uniques under
  inference_mode, where the embedding path does a per-step `np.unique(...cpu().numpy(), axis=0)`.
  It also keeps FilterCaption meaningful, and it takes reembed off the critical path -- 13.1% of
  sampled clips still had no `global_recap768` on 2026-09-01.

CLIP SAMPLING.  An episode holds a variable number of clips (10 median, 137 max), so indexing
(episode, clip) pairs would mean opening 323,980 files at startup.  Train draws a clip uniformly
at random per __getitem__ instead -- one extra h5 open per sample, and over 0.78 epoch of episodes
it visits clips in proportion to nothing but chance.  Val is deterministic: clip index 0.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.dataset_base import AnnotatedDataset
from src.data.egodex.egodex_io import clip_keys, episode_paths, read_clip
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

# (x, y, z) -> (x, -z, y).  R_x(+90 deg): maps the corpus's +y up onto this codebase's +z up,
# right-handed, det +1.  See the module docstring for the measurement that fixes the direction.
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]], dtype=np.float32)


class EgoDexClipDataset(AnnotatedDataset):
    """Egocentric clips of EgoDex.  ``CLASS_LABELS`` is empty: no GT semantics, captions only."""

    CLASS_LABELS: List[str] = []
    LOG_POSTFIX = "egodex_clip"

    def __init__(
        self,
        data_dir: str,
        split: str,
        ignore_label: int = -100,
        repeat: int = 1,
        transforms: Optional[List[Dict]] = None,
        num_masks: Optional[int] = None,
        anno_sources: Optional[List[str]] = None,
        min_mask_points: int = 32,
        min_clip_points: int = 1000,
        max_clip_points: Optional[int] = None,
        require_visible: bool = True,
        with_captions: bool = False,
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
        self.min_mask_points = min_mask_points
        self.min_clip_points = min_clip_points
        self.max_clip_points = max_clip_points
        self.require_visible = require_visible
        # Training always builds caption_data; validation does not (the loss is not run there). The
        # retrieval evaluation (tools/egocentric/eval_egodex_retrieval.py) needs captions on the held-out
        # split, so it sets this.
        self.with_captions = with_captions
        # scene_names are "<part>/<stem>" lines from the split file, itself written from the
        # manifest (PointWAM tools/build_human_demo_manifest.py) so "what did we train on" stays
        # answerable after the fact.
        self.episodes = [tuple(s.split("/", 1)) for s in self.scene_names]
        self._n_skipped = 0

    def __getitem__(self, idx_original):
        idx = idx_original % len(self.episodes)
        part, stem = self.episodes[idx]
        maskcap_h5, flows_h5 = episode_paths(str(self.data_dir), part, stem)

        try:
            keys = clip_keys(maskcap_h5)
        except Exception as e:  # a truncated h5 must not take the worker down with it
            self._skip(f"{part}/{stem}: clip_keys {type(e).__name__}")
            return None
        if not keys:
            self._skip(f"{part}/{stem}: no clip groups")
            return None

        clip = str(np.random.choice(keys)) if self.is_train else keys[0]
        try:
            c = read_clip(maskcap_h5, flows_h5, clip, frame=0,
                          require_visible=self.require_visible)
        except Exception as e:
            self._skip(f"{part}/{stem}#{clip}: read_clip {type(e).__name__}")
            return None
        if c is None:
            return None  # nothing in this clip can supervise the loss; point_collate drops None
        if c.coord.shape[0] < self.min_clip_points:
            return None

        coord = c.coord @ _Y_UP_TO_Z_UP.T
        color = c.color.astype(np.float32)
        keep = np.arange(coord.shape[0], dtype=np.int64)
        if self.max_clip_points and coord.shape[0] > self.max_clip_points:
            keep = np.sort(np.random.choice(coord.shape[0], self.max_clip_points, replace=False))
            coord, color = coord[keep], color[keep]

        data = dict(
            scene_name=f"{part}/{stem}#{clip}",
            coord=coord,
            color=color,
            origin_idx=keep,
        )

        if self.is_train or self.with_captions:
            seg = c.segment[keep]
            keep_idx, keep_cap = [], []
            for s, cap in enumerate(c.captions):
                if not cap:
                    continue
                local = np.flatnonzero(seg == s).astype(np.int64)
                if len(local) >= self.min_mask_points:
                    keep_idx.append(torch.from_numpy(local).int())
                    keep_cap.append(cap)
            if not keep_idx:
                return None  # a clip with no surviving caption would break the collate
            if self.num_masks is not None and self.num_masks < len(keep_idx):
                sel = np.random.choice(len(keep_idx), self.num_masks, replace=False)
                keep_idx = [keep_idx[i] for i in sel]
                keep_cap = [keep_cap[i] for i in sel]
            data["caption_data"] = dict(idx=keep_idx, caption=keep_cap)

        data = self.transforms(data)
        # FilterCaption runs inside the transforms and can reject every caption that survived the
        # guard above; an empty list reaches torch.cat([]) in the collate and kills the worker,
        # which then hangs every other rank.
        if (self.is_train or self.with_captions) and not data["caption_data"]["idx"]:
            return None
        return data

    def _skip(self, why: str) -> None:
        self._n_skipped += 1
        if self._n_skipped <= 20 or self._n_skipped % 1000 == 0:
            log.warning(f"[{self.LOG_POSTFIX}] skipped {self._n_skipped}: {why}")
