"""Convert a Lightning checkpoint of this repo into the bare state_dict PointWAM loads.

Why a conversion is needed
--------------------------
PointWAM reads the encoder checkpoint with

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    net_sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}

(pointwam/mosaic3d_encoder.py:166-168 for the spconv backend,
 pointwam/mosaic3d_wcn.py:626-631 for the wcn backend, which loads strict=True).

A checkpoint written by lightning.pytorch.callbacks.ModelCheckpoint is not a bare
state_dict.  Besides `state_dict` it carries `hyper_parameters`, and this repo's
LightningModule/LightningDataModule both call `save_hyperparameters()`, so
`hyper_parameters` holds hydra `_partial_` objects -- functools.partial instances
bound to `src.models.networks.ppt.model.PPT`,
`src.data.egocentric.arkit_frame_dataset.ARKitScenesFrameDataset`,
`src.data.utils.collate.point_collate_fn_with_masks`, ... .
`torch.load(..., weights_only=False)` unpickles the WHOLE file, so it raises
`ModuleNotFoundError: No module named 'src'` inside PointWAM's environment, which
does not have this repo on sys.path.  It also carries `optimizer_states`
(~1.0 GB of SGD momentum) and `clip_encoder.*` (640 M frozen parameters), neither
of which PointWAM reads.

This script emits exactly the shape of the RELEASED spunet101.ckpt:

    {"state_dict": {"net.backbone.*": ..., "net.embedding_table.weight": ...,
                    "caption_loss.logit_scale": ...}}

-- plain tensors only, so the output also loads under `weights_only=True`.

Usage
-----
    python tools/egocentric/to_pointwam_ckpt.py IN.ckpt OUT.ckpt \
        [--reference /path/to/spunet101.ckpt]

`--reference` checks the emitted `net.*` key set and per-tensor shapes against the
released checkpoint, which is the checkpoint PointWAM is known to load today.
That is the drop-in test: identical key set + identical shapes => the spconv path
(strict=False plus an assert on missing `backbone.*`) and the wcn path
(strict=True over `backbone.*` + `embedding_table.weight`) both accept it.
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import types
from typing import Any, Dict

import torch

# What PointWAM builds: PPT(conditions=["ScanNet", "ARKitScenes", "ScanNetPP"],
# context_channels=256) -- pointwam/mosaic3d_encoder.py:46-47, and
# PPTWCN(num_conditions=3, context_channels=256) -- pointwam/mosaic3d_wcn.py:596-600.
EXPECTED_EMBEDDING_SHAPE = (3, 256)
KEEP_PREFIXES = ("net.",)
KEEP_EXACT = ("caption_loss.logit_scale",)


class _StubUnpickler(pickle.Unpickler):
    """Unpickle a Lightning ckpt without this repo importable.

    Anything under `src.` (and any other module that is genuinely absent) is
    replaced by a placeholder class.  Only `state_dict`, which holds nothing but
    tensors, is read out afterwards, so the placeholders are never used.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            stub = types.new_class(name, (), {})
            stub.__module__ = module
            return stub


def _load_any(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        print(f"  [info] {exc}; retrying with stubbed unpickler")

        class _M:
            Unpickler = _StubUnpickler
            load = staticmethod(lambda f, **kw: _StubUnpickler(f, **kw).load())
            loads = staticmethod(lambda b, **kw: _StubUnpickler(io.BytesIO(b), **kw).load())
            dump = staticmethod(pickle.dump)
            dumps = staticmethod(pickle.dumps)
            HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
            Pickler = pickle.Pickler

        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_M)


def convert(src_path: str, dst_path: str, reference: str | None = None,
            allow_condition_mismatch: bool = False) -> Dict[str, torch.Tensor]:
    ckpt = _load_any(src_path)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{src_path}: expected a dict, got {type(ckpt)}")
    print(f"  input top-level keys: {sorted(ckpt.keys())}")
    sd = ckpt.get("state_dict", ckpt)

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(KEEP_PREFIXES) or k in KEEP_EXACT:
            if not isinstance(v, torch.Tensor):
                raise SystemExit(f"{k}: expected a tensor, got {type(v)}")
            # detach + contiguous clone: never carry a storage view of a 4 GB file,
            # and float32 is what the released ckpt stores.
            out[k] = v.detach().to(torch.float32).contiguous().clone()

    n_bb = sum(1 for k in out if k.startswith("net.backbone."))
    if not n_bb:
        raise SystemExit(f"{src_path}: no 'net.backbone.*' keys -- not a Mosaic3D checkpoint")

    emb = out.get("net.embedding_table.weight")
    if emb is None:
        raise SystemExit(
            "net.embedding_table.weight is missing. PointWAM builds PPT/PPTWCN with a "
            "3-row embedding_table and loads the whole net.* subtree (strict=True on "
            "the wcn backend), so this key must exist."
        )
    if tuple(emb.shape) != EXPECTED_EMBEDDING_SHAPE and not allow_condition_mismatch:
        raise SystemExit(
            f"net.embedding_table.weight is {tuple(emb.shape)}, PointWAM expects "
            f"{EXPECTED_EMBEDDING_SHAPE}. torch.nn.Module.load_state_dict raises on a "
            "size mismatch even with strict=False, so this checkpoint cannot be loaded "
            "by pointwam/mosaic3d_encoder.py. Train with net.conditions set to the "
            "released 3-condition list [ScanNet, ARKitScenes, ScanNetPP]."
        )

    if reference:
        ref = _load_any(reference)
        ref_sd = ref.get("state_dict", ref)
        ref_net = {k: tuple(v.shape) for k, v in ref_sd.items() if k.startswith("net.")}
        got_net = {k: tuple(v.shape) for k, v in out.items() if k.startswith("net.")}
        only_ref = sorted(set(ref_net) - set(got_net))
        only_got = sorted(set(got_net) - set(ref_net))
        bad = sorted(k for k in set(ref_net) & set(got_net) if ref_net[k] != got_net[k])
        print(f"  reference net.* tensors: {len(ref_net)} | converted: {len(got_net)}")
        if only_ref or only_got or bad:
            for k in only_ref[:10]:
                print(f"    MISSING vs reference: {k} {ref_net[k]}")
            for k in only_got[:10]:
                print(f"    EXTRA   vs reference: {k} {got_net[k]}")
            for k in bad[:10]:
                print(f"    SHAPE   {k}: ref {ref_net[k]} != got {got_net[k]}")
            raise SystemExit("key/shape mismatch against the reference checkpoint")
        print("  key set and every tensor shape match the released checkpoint")

    torch.save({"state_dict": out}, dst_path)
    n_par = sum(v.numel() for v in out.values())
    print(f"  wrote {dst_path}: {len(out)} tensors ({n_bb} backbone), {n_par / 1e6:.2f} M params")

    # Re-read exactly the way PointWAM does, and with weights_only=True as well.
    back = torch.load(dst_path, map_location="cpu", weights_only=False)
    back = back.get("state_dict", back)
    net_sd = {k[len("net."):]: v for k, v in back.items() if k.startswith("net.")}
    assert net_sd, "no net.* keys survived the round trip"
    assert "embedding_table.weight" in net_sd
    torch.load(dst_path, map_location="cpu", weights_only=True)
    print(f"  round trip OK: {len(net_sd)} net.* tensors, also loads with weights_only=True")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--reference", default=None,
                    help="released spunet101.ckpt to check key set and shapes against")
    ap.add_argument("--allow-condition-mismatch", action="store_true",
                    help="emit even if embedding_table is not 3x256 (it will NOT load in PointWAM)")
    a = ap.parse_args(argv)
    convert(a.src, a.dst, a.reference, a.allow_condition_mismatch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
