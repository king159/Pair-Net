# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.match_costs.builder import build_match_cost
from mmdet.core.bbox.match_costs.match_cost import MATCH_COST
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth for
    mask.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, mask focal cost and mask dice cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (:obj:`mmcv.ConfigDict` | dict): Classification cost config.
        mask_cost (:obj:`mmcv.ConfigDict` | dict): Mask cost config.
        dice_cost (:obj:`mmcv.ConfigDict` | dict): Dice cost config.
    """

    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),
        mask_cost=dict(type="FocalLossCost", weight=1.0, binary_input=True),
        dice_cost=dict(type="DiceCost", weight=1.0),
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)

    def assign(
        self,
        cls_pred,
        mask_pred,
        gt_labels,
        gt_mask,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred (Tensor | None): Class prediction in shape
                (num_query, cls_out_channels).
            mask_pred (Tensor): Mask prediction in shape (num_query, H, W).
            gt_labels (Tensor): Label of 'gt_mask'in shape = (num_gt, ).
            gt_mask (Tensor): Ground truth mask in shape = (num_gt, H, W).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        # K-Net sometimes passes cls_pred=None to this assigner.
        # So we should use the shape of mask_pred
        num_gt, num_query = gt_labels.shape[0], mask_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_query,), -1, dtype=torch.long)
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0

        if self.mask_cost.weight != 0:
            # mask_pred shape = [num_query, h, w]
            # gt_mask shape = [num_gt, h, w]
            # mask_cost shape = [num_query, num_gt]
            mask_cost = self.mask_cost(mask_pred, gt_mask)
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_mask)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(mask_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(mask_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)


@MATCH_COST.register_module()
class CrossEntropyLossCost:
    """CrossEntropyLossCost.
    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    Examples:
         >>> from mmdet.core.bbox.match_costs import CrossEntropyLossCost
         >>> import torch
         >>> bce = CrossEntropyLossCost(use_sigmoid=True)
         >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
         >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
         >>> print(bce(cls_pred, gt_labels))
    """

    def __init__(self, weight=1.0, use_sigmoid=True):
        assert use_sigmoid, "use_sigmoid = False is not supported yet."
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).
        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction="none"
        )
        cls_cost = torch.einsum("nc,mc->nm", pos, gt_labels) + torch.einsum(
            "nc,mc->nm", neg, 1 - gt_labels
        )
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.
        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight


@MATCH_COST.register_module()
class DiceCost:
    """Cost of mask assignments based on dice losses.
    Args:
        weight (int | float, optional): loss_weight. Defaults to 1.
        pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float, optional): default 1e-12.
        naive_dice (bool, optional): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If Flase, use the second power that
            is adopted by K-Net and SOLO.
            Defaults to True.
    """

    def __init__(self, weight=1.0, pred_act=False, eps=1e-3, naive_dice=True):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_query, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.
        Returns:
            Tensor: Dice cost matrix in shape (num_query, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum("nc,mc->nm", mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        else:
            denominator = (
                mask_preds.pow(2).sum(1)[:, None] + gt_masks.pow(2).sum(1)[None, :]
            )
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction logits in shape (num_query, *)
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
        Returns:
            Tensor: Dice cost matrix with weight in shape (num_query, num_gt).
        """
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight


from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult


class MaskSamplingResult(SamplingResult):
    """Mask sampling result."""

    def __init__(self, pos_inds, neg_inds, masks, gt_masks, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_masks = masks[pos_inds]
        self.neg_masks = masks[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_masks.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_masks.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_masks = torch.empty_like(gt_masks)
        else:
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def masks(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_masks, self.neg_masks])

    def __nice__(self):
        data = self.info.copy()
        data["pos_masks"] = data.pop("pos_masks").shape
        data["neg_masks"] = data.pop("neg_masks").shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = "    " + ",\n    ".join(parts)
        return "{\n" + body + "\n}"

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            "pos_inds": self.pos_inds,
            "neg_inds": self.neg_inds,
            "pos_masks": self.pos_masks,
            "neg_masks": self.neg_masks,
            "pos_is_gt": self.pos_is_gt,
            "num_gts": self.num_gts,
            "pos_assigned_gt_inds": self.pos_assigned_gt_inds,
        }


@BBOX_SAMPLERS.register_module()
class MaskPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.
        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            masks (torch.Tensor): Bounding boxes
            gt_masks (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        gt_flags = masks.new_zeros(masks.shape[0], dtype=torch.uint8)
        sampling_result = MaskSamplingResult(
            pos_inds, neg_inds, masks, gt_masks, assign_result, gt_flags
        )
        return sampling_result
