# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import AssignResult, BaseAssigner, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from scipy.optimize import linear_sum_assignment


@BBOX_ASSIGNERS.register_module()
class MaskHTriMatcher(BaseAssigner):
    def __init__(
        self,
        s_cls_cost=dict(type="ClassificationCost", weight=2.0),
        s_mask_cost=dict(type="BBoxL1Cost", weight=5.0),
        s_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
        o_cls_cost=dict(type="ClassificationCost", weight=1.0),
        o_mask_cost=dict(type="BBoxL1Cost", weight=5.0),
        o_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=2.0),
    ):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_mask_cost = build_match_cost(s_mask_cost)
        self.s_dice_cost = build_match_cost(s_dice_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_mask_cost = build_match_cost(o_mask_cost)
        self.o_dice_cost = build_match_cost(o_dice_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        # gt_sub_bboxes,
        # gt_obj_bboxes,
        sub_mask_points_pred,
        obj_mask_points_pred,
        sub_gt_points_masks,
        obj_gt_points_masks,
        gt_sub_labels,
        gt_obj_labels,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gt, num_query = gt_sub_labels.shape[0], sub_mask_points_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = sub_mask_points_pred.new_full(
            (num_query,), -1, dtype=torch.long
        )
        assigned_labels = sub_mask_points_pred.new_full(
            (num_query,), -1, dtype=torch.long
        )
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        sub_cls_cost = self.s_cls_cost(s_cls_score, gt_sub_labels)
        sub_mask_cost = self.s_mask_cost(sub_mask_points_pred, sub_gt_points_masks)
        sub_dice_cost = self.s_dice_cost(sub_mask_points_pred, sub_gt_points_masks)
        obj_cls_cost = self.o_cls_cost(o_cls_score, gt_obj_labels)
        obj_mask_cost = self.o_mask_cost(obj_mask_points_pred, obj_gt_points_masks)
        obj_dice_cost = self.o_dice_cost(obj_mask_points_pred, obj_gt_points_masks)
        rel_cls_cost = self.r_cls_cost(r_cls_score, gt_rel_labels)

        cost = (
            sub_cls_cost
            + sub_mask_cost
            + sub_dice_cost
            + obj_cls_cost
            + obj_mask_cost
            + obj_dice_cost
            + rel_cls_cost
        )

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            sub_gt_points_masks.device
        )
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            sub_gt_points_masks.device
        )

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class HTriMatcher(BaseAssigner):
    def __init__(
        self,
        s_cls_cost=dict(type="ClassificationCost", weight=1.0),
        s_reg_cost=dict(type="BBoxL1Cost", weight=1.0),
        s_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=1.0),
        o_cls_cost=dict(type="ClassificationCost", weight=1.0),
        o_reg_cost=dict(type="BBoxL1Cost", weight=1.0),
        o_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_reg_cost = build_match_cost(s_reg_cost)
        self.s_iou_cost = build_match_cost(s_iou_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_reg_cost = build_match_cost(o_reg_cost)
        self.o_iou_cost = build_match_cost(o_iou_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_bbox_pred,
        obj_bbox_pred,
        sub_cls_score,
        obj_cls_score,
        rel_cls_score,
        gt_sub_bboxes,
        gt_obj_bboxes,
        gt_sub_labels,
        gt_obj_labels,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_sub_bboxes.size(0), sub_bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = sub_bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)
        img_h, img_w, _ = img_meta["img_shape"]
        factor = gt_sub_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        s_cls_cost = self.s_cls_cost(sub_cls_score, gt_sub_labels)
        o_cls_cost = self.o_cls_cost(obj_cls_score, gt_obj_labels)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)
        # regression L1 cost
        normalize_gt_sub_bboxes = gt_sub_bboxes / factor
        normalize_gt_obj_bboxes = gt_obj_bboxes / factor
        s_reg_cost = self.s_reg_cost(sub_bbox_pred, normalize_gt_sub_bboxes)
        o_reg_cost = self.o_reg_cost(obj_bbox_pred, normalize_gt_obj_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred) * factor
        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred) * factor
        s_iou_cost = self.s_iou_cost(sub_bboxes, gt_sub_bboxes)
        o_iou_cost = self.o_iou_cost(obj_bboxes, gt_obj_bboxes)
        # weighted sum of above three costs
        beta_1, beta_2 = 1.2, 1
        alpha_s, alpha_o, alpha_r = 1, 1, 1
        cls_cost = (
            alpha_s * s_cls_cost + alpha_o * o_cls_cost + alpha_r * r_cls_cost
        ) / (alpha_s + alpha_o + alpha_r)
        bbox_cost = (s_reg_cost + o_reg_cost + s_iou_cost + o_iou_cost) / 2
        cost = beta_1 * cls_cost + beta_2 * bbox_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(sub_bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(sub_bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_s_labels
        ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)


@BBOX_ASSIGNERS.register_module()
class IdMatcher(BaseAssigner):
    def __init__(
        self,
        sub_id_cost=dict(type="ClassificationCost", weight=1.0),
        obj_id_cost=dict(type="ClassificationCost", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_score,
        obj_score,
        rel_cls_score,
        gt_sub_cls,
        gt_obj_cls,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """gt_ids are mapped from previous Hungarian matching results.

        ~[0,99]
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_rel_labels.shape[0], rel_cls_score.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

        # 2. compute the weighted costs
        # -object confidence
        sub_id_cost = self.sub_id_cost(sub_score, gt_sub_cls)
        obj_id_cost = self.obj_id_cost(obj_score, gt_obj_cls)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_cls[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_s_labels)


# the old Id matcher
@BBOX_ASSIGNERS.register_module()
class OldIdMatcher(BaseAssigner):
    def __init__(
        self,
        sub_id_cost=dict(type="ClassificationCost", weight=1.0),
        obj_id_cost=dict(type="ClassificationCost", weight=1.0),
        r_cls_cost=dict(type="ClassificationCost", weight=1.0),
    ):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(
        self,
        sub_match_score,
        obj_match_score,
        rel_cls_score,
        gt_sub_ids,
        gt_obj_ids,
        gt_rel_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        """gt_ids are mapped from previous Hungarian matchinmg results.
        ~[0,99]
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_rel_labels.size(0), rel_cls_score.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_s_labels
            ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        sub_id_cost = self.sub_id_cost(sub_match_score, gt_sub_ids)
        obj_id_cost = self.obj_id_cost(obj_match_score, gt_obj_ids)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_ids[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_ids[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_s_labels
        ), AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_o_labels)
