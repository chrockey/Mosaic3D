from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F
from torch_scatter import scatter, segment_csr

import src.utils.dist_utils as dist_utils
from src.models.losses.loss_base import LossBase
from src.utils.caption_utils import get_caption_batch, get_unique_caption_batch


class CaptionLossBase(LossBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, pred_feats, batch_dict: Dict) -> Tensor:
        return pred_feats, batch_dict

    def extract_text_features(
        self,
        captions: List[str],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ):
        if embeddings is not None:
            text_features = torch.stack([item for sublist in embeddings for item in sublist], 0)
            device = text_features.device
            _, labels_per_caption, labels_per_segment = np.unique(
                text_features.cpu().numpy(), return_index=True, return_inverse=True, axis=0
            )
            text_features = text_features[labels_per_caption]
            labels_per_segment = torch.from_numpy(labels_per_segment).to(device)
            labels_per_caption = torch.from_numpy(labels_per_caption).to(device)
        else:
            # extract text features
            with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
                text_features, labels_per_segment, labels_per_caption = get_unique_caption_batch(
                    captions, clip_encoder
                )
            text_features = (
                text_features.clone() if isinstance(text_features, torch.Tensor) else text_features
            )
        return text_features, labels_per_segment, labels_per_caption


class CaptionAlignmentLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.reduction = reduction

    def loss(
        self,
        point_features: Float[Tensor, "M D"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        # extract text features
        text_features, *_ = self.extract_text_features(captions, clip_encoder, embeddings)

        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)
        rep_point_features = point_features[point_indices]
        segment_features = segment_csr(
            rep_point_features,
            caption_offsets.to(rep_point_features.device),
            reduce="sum",
        )

        segment_features = nn.functional.normalize(segment_features, dim=-1)
        text_features = torch.cat(text_features, 0)

        # Compute the cosine similarity
        loss = 1 - torch.einsum("ij,ij->i", segment_features, text_features)
        num_points_per_caption = num_points_per_caption.to(loss.device)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "weighted_sum":
            loss = (loss * num_points_per_caption).sum() / num_points_per_caption.sum()
        return loss


class DenseCaptionAlignmentLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        is_entity: bool = False,
        interpolate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.is_entity = is_entity
        self.interpolate = interpolate

    def extract_text_features(
        self,
        captions: List[str],
        clip_encoder: nn.Module,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ):
        if embeddings is not None:
            text_features = torch.stack([item for sublist in embeddings for item in sublist], 0)
        else:
            with torch.cuda.amp.autocast(enabled=True) and torch.inference_mode():
                text_features = get_caption_batch(
                    captions, clip_encoder, is_entity=self.is_entity, interpolate=self.interpolate
                )
            text_features = (
                text_features.clone() if isinstance(text_features, torch.Tensor) else text_features
            )
        return text_features

    def loss(
        self,
        point_features: Float[Tensor, "M D"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offests: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
    ) -> Tensor:
        device, dtype = point_features.device, point_features.dtype

        # extract text features
        text_features, *_ = self.extract_text_features(captions, clip_encoder, embeddings)

        # Scatter and reduce caption embeddings
        flat_caption_embeddings = torch.cat(text_features, dim=0).to(dtype=dtype, device=device)
        caption_indices = torch.arange(len(flat_caption_embeddings)).repeat_interleave(
            num_points_per_caption
        )
        rep_caption_embeddings = flat_caption_embeddings[caption_indices]

        scattered_caption_embeddings = torch.zeros_like(point_features)
        scattered_caption_embeddings = scatter(
            rep_caption_embeddings,
            point_indices,
            dim=0,
            out=scattered_caption_embeddings,
            reduce="mean",
        )

        # Find which indices are not in corr_idx from 0 to len(pred_feats)
        mask = torch.zeros(len(point_features), dtype=torch.bool, device=device)
        mask[point_indices.unique()] = True

        # Use this mask to index into pred_feats and scattered_caption_embeddings
        pred_feats_masked = point_features[mask]
        if self.normalize:
            pred_feats_masked = nn.functional.normalize(pred_feats_masked, dim=-1)
        scattered_caption_embeddings_masked = scattered_caption_embeddings[mask]

        # Compute the cosine similarity (feats and embeddings are already normalized)
        loss = 1 - torch.einsum("ij,ij->i", pred_feats_masked, scattered_caption_embeddings_masked)

        return loss.mean()


class CaptionLoss(CaptionLossBase):
    """Per-point contrastive loss against the batch's caption embeddings.

    Two options exist for post-training on single-view clouds, both off by
    default so the released recipe is bit-for-bit unchanged:

    ``freeze_logit_scale``
        The released spunet101.ckpt carries a *learned* temperature
        (logit_scale = 5.08385, exp = 161.4, tau = 0.0062) calibrated on the
        statistics of full-scene batches.  A single frame contributes ~5 masks
        instead of up to 300, so the negative pool changes by an order of
        magnitude and a temperature left free will chase that shift.  Pinning it
        keeps the objective's geometry identical to pretraining.

    ``gather_text``
        The softmax denominator is the set of unique captions in the *local*
        batch.  Frames carry ~5 masks each against a full scene's ~300, so the
        denominator collapses (with one unique caption log_softmax is exactly 0
        and the gradient vanishes).  This shares the caption embeddings across
        ranks so the denominator is the whole global batch, which widens the
        negatives without touching the loss form the way switching to
        ``CaptionCLIPLoss`` would.  Text features are frozen, so no autograd has
        to cross the all_gather.
    """

    def __init__(
        self,
        normalize: bool = True,
        use_logit_scale: Optional[bool] = False,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        freeze_logit_scale: bool = False,
        gather_text: bool = False,
        max_logit_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        # Cap on the learned log-temperature, applied at use (as CLIP does with ln(100)). Without it the
        # learned scale runs away: on EgoDex post-training exp(logit_scale) went 146 -> 626 -> 2,419
        # (run 5, lr 0.004) and 146 -> 836 -> 3,192 (run 4, lr 0.01) over steps 150 -> 1,050 -> 1,350,
        # and the loss turned up once it passed ~600-800, at the same step for both learning rates.
        self.max_logit_scale = max_logit_scale
        self.loss_func = nn.NLLLoss(reduction="none")
        assert reduction in ["mean", "weighted_sum"]
        self.reduction = reduction
        self.use_logit_scale = use_logit_scale
        self.gather_text = gather_text
        if use_logit_scale:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07), requires_grad=not freeze_logit_scale
            )

        self.kwargs = kwargs

    @staticmethod
    def _gather_text(text_features: Tensor, labels_per_segment: Tensor):
        """Union the per-rank caption embeddings; returns (features, shifted labels).

        Rows are deduplicated after the gather, so a caption produced on two
        ranks stays one column instead of becoming its own negative.
        """
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return text_features, labels_per_segment
        world = dist.get_world_size()
        n = torch.tensor([text_features.shape[0]], device=text_features.device)
        sizes = [torch.zeros_like(n) for _ in range(world)]
        dist.all_gather(sizes, n)
        sizes = [int(x.item()) for x in sizes]
        biggest = max(sizes)
        padded = text_features.new_zeros((biggest, text_features.shape[1]))
        padded[: text_features.shape[0]] = text_features
        bucket = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(bucket, padded)
        gathered = torch.cat([b[:s] for b, s in zip(bucket, sizes)], dim=0)
        offset = sum(sizes[: dist.get_rank()])
        uniq, inverse = torch.unique(gathered, dim=0, return_inverse=True)
        return uniq, inverse[labels_per_segment.to(inverse.device) + offset]

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        # extract text features
        text_features, labels_per_segment, labels_per_caption = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # normalize point features
        if self.normalize:
            point_features = nn.functional.normalize(point_features, dim=-1)

        if self.gather_text:
            text_features, labels_per_segment = self._gather_text(
                text_features.to(device), labels_per_segment.to(device)
            )

        # Only the rows named by point_indices are ever read, so build the (N, K)
        # matrix over the unique ones instead of every point in the batch.
        # log_softmax is row-wise, so this is exactly equal to indexing afterwards
        # -- it just does not allocate the rows that get thrown away.  With single
        # views the saving is what makes a large batch fit: a frame's masks cover
        # roughly half its points, and both `logits` and the fp32 log_softmax
        # output are held for backward.
        point_indices = point_indices.to(device)
        rows, inverse = torch.unique(point_indices, return_inverse=True)

        # Logit
        logits = point_features[rows] @ text_features.T.to(device)
        if self.use_logit_scale:
            ls = self.logit_scale
            if self.max_logit_scale is not None:
                ls = ls.clamp(max=float(self.max_logit_scale))
            logits = ls.exp() * logits
        scores = F.log_softmax(logits, dim=-1)

        rep_scores = scores[inverse]
        reduced_scores = segment_csr(rep_scores, caption_offsets.to(device), reduce="mean")

        # Compute the loss
        loss = self.loss_func(reduced_scores, labels_per_segment.to(device))

        # Compute the cosine similarity
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "weighted_sum":
            num_points_per_caption = num_points_per_caption.to(loss.device)
            return (loss * num_points_per_caption).sum() / (num_points_per_caption.sum())
        else:
            raise ValueError(f"Unknown reduce type: {self.reduce}")


class CaptionCLIPLoss(CaptionLossBase):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        init_logit_scale: Optional[float] = 1 / 0.07,
        all_gather: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        assert reduction in ["mean", "weighted_sum"]
        self.all_gather = all_gather

        if init_logit_scale is not None:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(init_logit_scale), requires_grad=True
            )
        else:
            self.logit_scale = torch.ones([]) * np.log(init_logit_scale)

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_segment, labels_per_caption = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        if world_size > 1 and self.all_gather:
            all_captions = [None for _ in range(world_size)]
            dist.all_gather_object(all_captions, captions)
            all_captions: List[List[str]] = [item for sublist in all_captions for item in sublist]
            (
                all_text_features,
                all_labels_per_segment,
                all_labels_per_caption,
            ) = self.extract_text_features(all_captions, clip_encoder)

        # aggregate point features
        rep_point_features = point_features[point_indices]
        segment_features = segment_csr(
            rep_point_features, caption_offsets.to(device), reduce="mean"
        )
        if self.normalize:
            segment_features = nn.functional.normalize(segment_features, dim=-1)

        if world_size > 1 and self.all_gather:
            all_segment_features = dist_utils.all_gather_different_shapes(segment_features)
            all_segment_features[dist.get_rank()] = segment_features  # this is for gradient
            all_segment_features = torch.cat(all_segment_features, 0)

        # logits
        if world_size > 1 and self.all_gather:
            logits_per_segment = self.logit_scale.exp() * (
                all_segment_features @ all_text_features.T
            )
            logits_per_caption = logits_per_segment.T
        else:
            logits_per_segment = self.logit_scale.exp() * (segment_features @ text_features.T)
            logits_per_caption = self.logit_scale.exp() * (text_features @ segment_features.T)

        target_per_segment = (
            all_labels_per_segment if world_size > 1 and self.all_gather else labels_per_segment
        )
        target_per_caption = (
            all_labels_per_caption if world_size > 1 and self.all_gather else labels_per_caption
        )
        total_loss = (
            torch.nn.functional.cross_entropy(logits_per_segment, target_per_segment.to(device))
            + torch.nn.functional.cross_entropy(logits_per_caption, target_per_caption.to(device))
        ) / 2

        return total_loss


class CaptionSigLIPLoss(CaptionCLIPLoss):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "weighted_sum"] = "weighted_sum",
        init_logit_scale: Optional[float] = 10,
        init_logit_bias: Optional[float] = -10,
        bidir: bool = False,
        all_gather: bool = True,
        pooling_first: bool = True,
        **kwargs,
    ):
        super().__init__(normalize, reduction, init_logit_scale, **kwargs)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        self.bidir = bidir
        self.all_gather = all_gather
        self.pooling_first = pooling_first

    def get_ground_truth(
        self,
        shape,
        device,
        labels_per_segment: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
        negative_only: bool = False,
    ) -> torch.Tensor:
        labels = torch.zeros(shape, device=device)
        if not negative_only:
            assert labels_per_segment is not None
            labels[range(shape[0]), labels_per_segment] = 1
        return labels

    def get_logits(self, features, text_features):
        logits = self.logit_scale.exp() * features @ text_features.T
        if self.logit_bias is not None:
            logits += self.logit_bias
        return logits

    def _loss(
        self,
        point_features,
        text_features,
        point_indices,
        caption_offsets,
        labels_per_segment: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
        negative_only: bool = False,
    ):
        device = point_features.device

        # feature pooling first -> compute logits
        if self.pooling_first:
            rep_point_features = point_features[point_indices]
            segment_features = segment_csr(
                rep_point_features, caption_offsets.to(device), reduce="mean"
            )
            segment_features = nn.functional.normalize(segment_features, dim=-1)
            logits = self.get_logits(segment_features, text_features)
            labels = self.get_ground_truth(logits.shape, device, labels_per_segment, negative_only)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        # compute logits first -> probability pooling
        else:
            point_features = nn.functional.normalize(point_features, dim=-1)
            logits = self.get_logits(point_features, text_features)
            rep_logits = logits[point_indices]
            reduced_logits = segment_csr(rep_logits, caption_offsets.to(device), reduce="mean")
            labels = self.get_ground_truth(
                reduced_logits.shape, device, labels_per_segment, negative_only
            )
            loss = F.binary_cross_entropy_with_logits(reduced_logits, labels)
        return loss

    def loss(
        self,
        point_features: Float[Tensor, "M 512"],  # noqa: F722
        point_indices: Int[Tensor, "L"],  # noqa: F821
        caption_offsets: Int[Tensor, "B + 1"],  # noqa: F821
        num_points_per_caption: Int[Tensor, "B"],  # noqa: F821
        clip_encoder: nn.Module,
        captions: Optional[List[List[str]]] = None,
        embeddings: Optional[List[List[Float[Tensor, "D"]]]] = None,  # noqa: F722,F821
        **kwargs,
    ) -> Tensor:
        device = point_features.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # extract text features
        text_features, labels_per_segment, _ = self.extract_text_features(
            captions, clip_encoder, embeddings
        )

        # loss
        loss = self._loss(
            point_features, text_features, point_indices, caption_offsets, labels_per_segment
        )

        if world_size > 1 and self.all_gather:
            # get max num captions
            all_shapes = dist_utils.all_gather_tensor_shapes(text_features)
            all_num_captions = all_shapes[:, 0]
            max_num_captions = all_num_captions.max()
            num_captions = text_features.shape[0]

            # pad text features
            text_features_padded = torch.zeros(
                max_num_captions, text_features.shape[1], device=device
            )
            text_features_padded[:num_captions] = text_features

            # exchange text features
            rank = dist.get_rank()
            right_rank = (rank + 1) % world_size
            left_rank = (rank - 1 + world_size) % world_size
            if self.bidir:
                text_features_to_right, text_features_to_left = text_features_padded
                num_captions_to_right = num_captions_to_left = all_num_captions[rank]
                num_bidir, remainder = divmod(world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = dist_utils.neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    num_captions_rev = dist_utils.neighbour_exchange_bidir(
                        left_rank,
                        right_rank,
                        num_captions_to_left,
                        num_captions_to_right,
                    )
                    for f, n in zip(text_features_recv, num_captions_rev):
                        loss += self._loss(
                            point_features,
                            f[:n],
                            point_indices,
                            caption_offsets,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv
                    num_captions_to_left, num_captions_to_right = num_captions_rev

                if remainder:
                    text_features_recv = dist_utils.neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        point_features,
                        text_features_recv,
                        point_indices,
                        caption_offsets,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features_padded
                num_captions_to_right = all_num_captions[rank]
                for i in range(world_size - 1):
                    text_features_from_left = dist_utils.neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    num_captions_from_left = dist_utils.neighbour_exchange(
                        left_rank, right_rank, num_captions_to_right
                    )
                    _loss = self._loss(
                        point_features,
                        text_features_from_left[:num_captions_from_left],
                        point_indices,
                        caption_offsets,
                        negative_only=True,
                    )
                    loss += _loss
                    text_features_to_right = text_features_from_left
                    num_captions_to_right = num_captions_from_left

        return loss
