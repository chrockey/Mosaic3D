# SPDX-License-Identifier: Apache-2.0
"""Corpus I/O for beomjun's EgoDex human demos: one clip -> (points, per-point segment, captions).

This module knows the CORPUS and nothing about Mosaic3D. Keeping the two apart is deliberate: the
h5 layout below is fixed by the producer and can be checked on a CPU box with no torch, while the
framework contract belongs in the Dataset class that wraps this.

LAYOUT, verified on disk 2026-09-01. An episode is a `stem` and lives as TWO files:

    flows/<flowpart>/<stem>.h5
        <clip>/camera_0_scene_flows          (F, N, 3) float16   F frames of N points, metres
        <clip>/camera_0_scene_colors         (N, 3)    uint8
        <clip>/camera_0_scene_normals        (N, 3)    float32
        <clip>/camera_0_scene_visibility     (F, N)    bool
        <clip>/camera_0_scene_depth_valid_mask (F, N)  bool
        <clip>/camera_0_extrinsic            (4, 4)    float32
        intrinsic                            (3, 3)    float32
        attrs: grid_size_m, frames_per_clip, step_stride, lang, task, uuid, domain

    maskcap/<part>/<stem>.h5
        <clip>/camera_0_instance_id          (N,)      uint16    aligned with the N points above
        global_siglip2                       (G, 1152) float16
        global_recap768                      (G, 768)  float16   written by
        recap768_mask                        (G,)      uint8     PointWAM tools/reembed_captions_recap.py
        attrs: global_captions, global_is_agent, global_view_clips  (JSON, keys are id STRINGS)

`<clip>` keys look like "0:11" and are the same set in both files. **Row g-1 of every `global_*`
array holds instance id g**, and id 0 is unlabelled background.

WHAT IS DROPPED, and why:
  * id 0            — background, no caption exists
  * `global_is_agent` — the human hand and forearm. They are the demonstrator, not scene content,
                        and Mosaic3D is being taught object geometry.
  * empty captions  — `egodex` (only that part) ships some; nothing to align against.
  * ids with `recap768_mask == 0` — the re-embedding pass declined them for one of the two reasons
                        above, so this check is belt-and-braces and also catches a partial pass.
A clip with no surviving segment yields None: the caption loss has nothing to reduce over.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import h5py
import numpy as np

# maskcap part -> flows part. They differ for the first one only.
FLOWS_PART = {
    "egodex": "egodex_part1",
    "egodex_part2": "egodex_part2",
    "egodex_part3": "egodex_part3",
    "egodex_part4": "egodex_part4",
    "egodex_part5": "egodex_part5",
    "egodex_extra": "egodex_extra",
    "egodex_test": "egodex_test",
    "vitra": "vitra",
}

TEXT_DIM = 768


@dataclass(frozen=True)
class Clip:
    """One training sample's worth of corpus data, before any framework transform."""

    coord: np.ndarray        # (N, 3) float32, metres, camera frame at t=0
    color: np.ndarray        # (N, 3) float32 in [0, 255]
    normal: np.ndarray       # (N, 3) float32
    segment: np.ndarray      # (N,)   int64, -1 = no caption, else an index into `text`
    text: np.ndarray | None   # (S, 768) float32 L2-normalised, or None when not re-embedded
    captions: list[str]      # S raw strings, parallel to `text`, for logging and retrieval evals
    lang: str                # the episode's task sentence
    uuid: str


def episode_paths(root: str, part: str, stem: str) -> tuple[str, str]:
    """(maskcap h5, flows h5) for one episode."""
    return (
        os.path.join(root, "maskcap", part, stem + ".h5"),
        os.path.join(root, "flows", FLOWS_PART[part], stem + ".h5"),
    )


def read_manifest(path: str) -> list[tuple[str, str]]:
    """The `<part>\\t<stem>` TSV written by PointWAM tools/build_human_demo_manifest.py."""
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            part, stem = line.split("\t", 1)
            out.append((part, stem))
    return out


def clip_keys(maskcap_h5: str) -> list[str]:
    """Clip group names, in the file's own order. Cheap: opens the file and reads nothing else."""
    with h5py.File(maskcap_h5, "r") as f:
        return [k for k in f.keys() if isinstance(f[k], h5py.Group)]


def read_clip(maskcap_h5: str, flows_h5: str, clip: str, *, frame: int = 0,
              require_visible: bool = True) -> Clip | None:
    """Read one clip. Returns None when nothing in it can supervise the caption loss."""
    with h5py.File(maskcap_h5, "r") as m:
        if clip not in m:
            return None
        inst = np.asarray(m[clip]["camera_0_instance_id"][...]).astype(np.int64)
        caps = json.loads(m.attrs["global_captions"])
        # `global_recap768` is OPTIONAL. The training path passes caption STRINGS and lets the
        # model's own frozen Recap-CLIP text tower embed them (src/models/losses/caption_loss.py:41
        # takes either), which is the loss path that produced the released spunet101.ckpt and keeps
        # FilterCaption meaningful. Requiring the embeddings here would have gated 34% of the corpus
        # on a reembed pass that training does not need; `recap768_mask` filtered caption quality,
        # and FilterCaption in the transforms does that job instead.
        emb = ok = None
        if "global_recap768" in m:
            emb = np.asarray(m["global_recap768"][...]).astype(np.float32)  # (G, 768)
            if "recap768_mask" in m:
                ok = np.asarray(m["recap768_mask"][...]).astype(bool)       # (G,)

    with h5py.File(flows_h5, "r") as fl:
        if clip not in fl:
            return None
        g = fl[clip]
        flows = g["camera_0_scene_flows"]
        coord = np.asarray(flows[frame]).astype(np.float32)                 # (N, 3)
        color = np.asarray(g["camera_0_scene_colors"][...]).astype(np.float32)
        normal = (np.asarray(g["camera_0_scene_normals"][...]).astype(np.float32)
                  if "camera_0_scene_normals" in g else np.zeros_like(coord))
        keep = np.ones(coord.shape[0], dtype=bool)
        if require_visible:
            # A point the camera cannot see at this frame has no colour worth aligning text to.
            for k in ("camera_0_scene_visibility", "camera_0_scene_depth_valid_mask"):
                if k in g:
                    keep &= np.asarray(g[k][frame]).astype(bool)
        lang = str(fl.attrs.get("lang", ""))
        uuid = str(fl.attrs.get("uuid", ""))

    n = min(coord.shape[0], inst.shape[0])
    if n == 0:
        return None
    coord, color, normal, inst, keep = coord[:n], color[:n], normal[:n], inst[:n], keep[:n]

    # Instance ids that survive: labelled, and carrying a caption. When the embeddings are present
    # an id must also be inside the table and accepted by the re-embedding pass.
    ids = np.unique(inst[keep & (inst > 0)])
    ids = np.array([g for g in ids
                    if str(int(g)) in caps and str(caps[str(int(g))]).strip()
                    and (ok is None or (0 <= g - 1 < ok.shape[0] and ok[g - 1]))
                    and (emb is None or 0 <= g - 1 < emb.shape[0])], dtype=np.int64)
    if ids.size == 0:
        return None

    # Compact ids -> 0..S-1 so `segment` indexes `text` directly.
    lut = np.full(int(inst.max()) + 2, -1, dtype=np.int64)
    lut[ids] = np.arange(ids.size)
    segment = np.where(keep, lut[inst], -1)

    text = emb[ids - 1] if emb is not None else None
    captions = [str(caps.get(str(int(g)), "")) for g in ids]
    return Clip(coord=coord, color=color, normal=normal, segment=segment,
                text=text, captions=captions, lang=lang, uuid=uuid)
