#!/usr/bin/env python3
"""Caption retrieval on egodex_test -- the "is the language alignment still there" signal.

For a checkpoint of the post-trained encoder, take N held-out clips (egodex_val.txt = the upstream
egodex_test part, 3,104 episodes, never in egodex_train.txt), run the network, mean-pool the per-point
features over each captioned instance exactly as CaptionAlignmentLoss does (segment_csr over
caption_offsets, L2-normalised), embed every UNIQUE caption in the evaluated pool with the frozen
Recap-CLIP text tower, and ask: among all unique captions of the pool, where does a segment's own
caption rank by cosine similarity?  Reported as recall@1/5/10 and median rank, over ~7.3 segments per
clip, so N=512 clips is ~3,700 queries against ~3,000 unique captions.

Why this and not a validation caption loss: the loss answers "does it caption human demos better",
which is not what the encoder is being post-trained for.  Retrieval on held-out clips answers whether
the Recap-CLIP alignment the rest of the stack relies on (PointWAM's word tokens live in the same
768-d space) is preserved -- the second of the two snapshot-selection signals in
experiments/dexjoco-human-pretraining-plan.md; the first is PointWAM tools/probe_nail_depth.py.

    python3.10 tools/egocentric/eval_egodex_retrieval.py --ckpt <lightning .ckpt | *.pointwam.ckpt | spunet101.ckpt> \
        [--n 512] [--batch 64] [--seed 0] [--out results.json]

Deterministic: clip index 0 of each episode (val mode), a fixed subset of episodes, no augmentation
(the val transform list drops every Random*/Elastic/Chromatic op and keeps FilterCaption, Copy,
centring, colour normalisation and the condition tag; voxelisation happens inside the net from the
collate's grid_size).  Locally: --data-dir /home/jovyan/cholab/datasets/mosaic3d-posttrain/egodex.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_scatter import segment_csr

# transforms of the training list that must NOT run at eval
_AUG = {"RandomRotate", "RandomScale", "RandomFlip", "RandomJitter", "ElasticDistortion",
        "ChromaticAutoContrast", "ChromaticTranslation", "ChromaticJitter"}
# NOT Copy: it is deterministic and the only producer of origin_coord, which Collect's pc_count reads.


def load_net_state(path: str) -> dict:
    """net.* tensors from a Lightning ckpt, a *.pointwam.ckpt, or the released spunet101.ckpt."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    return {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=512, help="held-out episodes to evaluate (clip 0 of each)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default=os.environ.get("M3D_DATASETS", "/datasets") + "/mosaic3d-posttrain/egodex")
    a = ap.parse_args()

    import hydra
    from hydra import compose, initialize_config_dir
    from src.data.utils.collate import point_collate_fn_with_masks
    from src.utils.caption_utils import get_unique_caption_batch

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
        cfg = compose(config_name="train",
                      overrides=["experiment=posttrain_spunet101_egodex", "logger=[]", "trainer.devices=1"])

    # --- model: the LightningModule, its net built, our weights loaded, text tower on the GPU ---
    module = hydra.utils.instantiate(cfg.model)
    module.configure_model()
    net_sd = load_net_state(a.ckpt)
    missing, unexpected = module.net.load_state_dict(net_sd, strict=False)
    print(f"[ckpt] {a.ckpt}: {len(net_sd)} net tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    if not net_sd or missing or unexpected:
        # a wrong file (e.g. a PointWAM model-step*.pt, whose keys start with scene_feature_encoder.)
        # must not score a randomly initialised backbone
        raise SystemExit(f"checkpoint does not match module.net: {len(net_sd)} net tensors, "
                         f"missing {missing[:5]}, unexpected {unexpected[:5]}")
    dev = torch.device("cuda")
    module = module.to(dev).eval()
    module.clip_encoder = module.clip_encoder.to(dev).eval()

    # --- data: the val split, captions on, augmentation off ---
    ds_cfg = OmegaConf.to_container(cfg.data.train_dataset.datasets[0], resolve=True)
    tf = [t for t in ds_cfg["transforms"] if t["type"] not in _AUG]
    from src.data.egodex.egodex_clip_dataset import EgoDexClipDataset
    ds = EgoDexClipDataset(data_dir=a.data_dir, split="val", transforms=OmegaConf.create(tf), with_captions=True,
                           min_mask_points=ds_cfg["min_mask_points"], min_clip_points=ds_cfg["min_clip_points"],
                           require_visible=ds_cfg["require_visible"])
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(ds.episodes), size=min(a.n, len(ds.episodes)), replace=False)
    grid = float(cfg.data.collate_fn.grid_size)
    print(f"[data] val episodes {len(ds.episodes)}, evaluating {len(idx)}, grid {grid}")

    # --- forward: per-segment features and their captions ---
    seg_feats, seg_caps = [], []
    t0 = time.time(); n_ok = 0
    for b0 in range(0, len(idx), a.batch):
        samples = [ds[int(i)] for i in idx[b0:b0 + a.batch]]
        samples = [s for s in samples if s is not None]
        if not samples:
            continue
        batch = point_collate_fn_with_masks(samples, grid_size=grid)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(dev, non_blocking=True)
        cd = batch["caption_data"]
        for k in ("point_indices", "caption_offsets", "num_points_per_caption"):
            cd[k] = cd[k].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = module(batch)
        feat = torch.nn.functional.normalize(out["clip_feat"].float(), dim=-1)
        rep = feat[cd["point_indices"].long()]
        seg = segment_csr(rep, cd["caption_offsets"].long(), reduce="mean")
        seg = torch.nn.functional.normalize(seg, dim=-1)
        caps = [c for sub in cd["caption"] for c in sub]
        assert seg.shape[0] == len(caps), (seg.shape, len(caps))
        seg_feats.append(seg.cpu()); seg_caps += caps; n_ok += len(samples)
    seg_feats = torch.cat(seg_feats, 0)
    print(f"[fwd] {n_ok} clips, {seg_feats.shape[0]} segments in {time.time()-t0:.0f}s")

    # --- text: every unique caption of the pool, once ---
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        text, seg_to_unique, _ = get_unique_caption_batch([seg_caps], module.clip_encoder)
    text = torch.nn.functional.normalize(text.float(), dim=-1).cpu()
    seg_to_unique = seg_to_unique.long().cpu()
    sim = seg_feats @ text.T                                   # (S, U)
    target = sim[torch.arange(sim.shape[0]), seg_to_unique]    # own caption's similarity
    # ties: optimistic rank counts only strictly-better captions, pessimistic counts ties against us
    rank_lo = (sim > target[:, None]).sum(1)                   # 0 = top-1
    rank_hi = (sim >= target[:, None]).sum(1) - 1
    rank = (rank_lo.float() + rank_hi.float()) / 2             # mid-rank, what is reported
    res = dict(ckpt=a.ckpt, clips=n_ok, segments=int(sim.shape[0]), unique_captions=int(sim.shape[1]),
               recall_at_1=float((rank < 1).float().mean()), recall_at_5=float((rank < 5).float().mean()),
               recall_at_10=float((rank < 10).float().mean()),
               recall_at_1_optimistic=float((rank_lo < 1).float().mean()),
               median_rank=float(torch.quantile(rank, 0.5)) + 1, ties_at_top=int((rank_hi != rank_lo).sum()),
               mean_own_sim=float(target.mean()), seed=a.seed)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
