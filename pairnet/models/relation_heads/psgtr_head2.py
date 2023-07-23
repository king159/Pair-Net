# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, build_assigner, build_sampler,
                        multi_apply, reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness


@HEADS.register_module()
class PSGTrHead2(AnchorFreeHead):
    # PSGTR using Mask2former
    def __init__(
        self,
        num_classes,
        num_relations,
        in_channels=[256, 512, 1024, 2048],
        use_mask=True,
        num_obj_query=100,
        num_reg_fcs=2,
        n_heads=8,
        embed_dims=256,
        swin_backbone=None,
        sync_cls_avg_factor=False,
        bg_cls_weight=0.02,
        pixel_decoder=None,
        transformer_decoder=None,
        feat_channels=256,
        out_channels=256,
        num_transformer_feat_level=3,
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        sub_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        sub_loss_mask=dict(type="GIoULoss", loss_weight=2.0),
        sub_loss_dice=dict(type="DiceLoss", loss_weight=1.0),
        obj_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        obj_loss_mask=None,
        obj_loss_dice=None,
        rel_loss_cls=None,
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            sampler=dict(type="MaskPseudoSampler"),
            assigner=dict(
                type="MaskHTriMatcher",
                s_cls_cost=dict(type="ClassificationCost", weight=2.0),
                s_mask_cost=dict(
                    type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                ),
                s_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
                o_cls_cost=dict(type="ClassificationCost", weight=1.0),
                o_mask_cost=dict(
                    type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                ),
                o_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
                r_cls_cost=dict(type="ClassificationCost", weight=2.0),
            ),
        ),
        test_cfg=None,
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.sync_cls_avg_factor = sync_cls_avg_factor
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        assert isinstance(bg_cls_weight, float), (
            "Expected "
            "bg_cls_weight to have type float. Found "
            f"{type(bg_cls_weight)}."
        )
        self.bg_cls_weight = bg_cls_weight

        assert isinstance(use_mask, bool), (
            "Expected " "use_mask to have type bool. Found " f"{type(use_mask)}."
        )
        self.use_mask = use_mask

        s_class_weight = sub_loss_cls.get("class_weight", None)
        assert isinstance(s_class_weight, float), (
            "Expected "
            "class_weight to have type float. Found "
            f"{type(s_class_weight)}."
        )

        s_class_weight = torch.ones(num_classes + 1) * s_class_weight
        # NOTE set background class as the last indice
        s_class_weight[-1] = bg_cls_weight
        sub_loss_cls.update({"class_weight": s_class_weight})

        o_class_weight = obj_loss_cls.get("class_weight", None)
        assert isinstance(o_class_weight, float), (
            "Expected "
            "class_weight to have type float. Found "
            f"{type(o_class_weight)}."
        )

        o_class_weight = torch.ones(num_classes + 1) * o_class_weight
        # NOTE set background class as the last indice
        o_class_weight[-1] = bg_cls_weight
        obj_loss_cls.update({"class_weight": o_class_weight})
        self.obj_class_weight = o_class_weight
        r_class_weight = rel_loss_cls.get("class_weight", None)
        assert isinstance(r_class_weight, float), (
            "Expected "
            "class_weight to have type float. Found "
            f"{type(r_class_weight)}."
        )

        r_class_weight = torch.ones(num_relations + 1) * r_class_weight
        self.r_class_weight = r_class_weight
        # NOTE set background class as the first indice for relations as they are 1-based
        r_class_weight[0] = bg_cls_weight
        rel_loss_cls.update({"class_weight": r_class_weight})
        if "bg_cls_weight" in rel_loss_cls:
            rel_loss_cls.pop("bg_cls_weight")
        if train_cfg:
            self.train_cfg = train_cfg
            self.assigner = build_assigner(self.train_cfg.assigner, context=self)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get("num_points", 12544)
            self.oversample_ratio = self.train_cfg.get("oversample_ratio", 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                "importance_sample_ratio", 0.75
            )

        self.num_queries = num_obj_query
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.in_channels = in_channels
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.swin = swin_backbone

        self.obj_loss_cls = build_loss(obj_loss_cls)
        self.obj_loss_mask = build_loss(obj_loss_mask)

        self.sub_loss_cls = build_loss(sub_loss_cls)
        self.sub_loss_mask = build_loss(sub_loss_mask)
        if self.use_mask:
            # self.obj_focal_loss = build_loss(obj_focal_loss)
            self.obj_loss_dice = build_loss(obj_loss_dice)
            # self.sub_focal_loss = build_loss(sub_focal_loss)
            self.sub_loss_dice = build_loss(sub_loss_dice)

        self.rel_loss_cls = build_loss(rel_loss_cls)

        if self.obj_loss_cls.use_sigmoid:
            self.obj_cls_out_channels = num_classes
        else:
            self.obj_cls_out_channels = num_classes + 1

        if self.sub_loss_cls.use_sigmoid:
            self.sub_cls_out_channels = num_classes
        else:
            self.sub_cls_out_channels = num_classes + 1

        if rel_loss_cls["use_sigmoid"]:
            self.rel_cls_out_channels = num_relations
        else:
            self.rel_cls_out_channels = num_relations + 1

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.n_heads = n_heads
        self.embed_dims = embed_dims
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
        )
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (
                self.decoder_embed_dims != feat_channels
                or enforce_decoder_input_project
            ):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1)
                )
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding
        )
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.sub_cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.obj_cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.rel_cls_embed = nn.Linear(feat_channels, self.num_relations + 1)
        self.sub_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )
        self.obj_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred_sub = self.sub_cls_embed(decoder_out)
        cls_pred_obj = self.obj_cls_embed(decoder_out)
        cls_pred_rel = self.rel_cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
        mask_embed_sub = self.sub_mask_embed(decoder_out)
        mask_pred_sub = torch.einsum("bqc,bchw->bqhw", mask_embed_sub, mask_feature)
        mask_embed_obj = self.obj_mask_embed(decoder_out)
        mask_pred_obj = torch.einsum("bqc,bchw->bqhw", mask_embed_obj, mask_feature)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat((1, self.n_heads, 1, 1))
            .flatten(0, 1)
        )
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return (
            cls_pred_sub,
            cls_pred_obj,
            cls_pred_rel,
            mask_pred_sub,
            mask_pred_obj,
            attn_mask,
        )

    def forward(self, feats, img_metas):
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        decoder_inputs = []
        decoder_positional_encodings = []
        decoder_positional_encodings_rel = []

        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(2, 0, 1)
            decoder_positional_encoding_rel = self.decoder_positional_encoding(mask)
            decoder_positional_encoding_rel = decoder_positional_encoding_rel.flatten(
                2
            ).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
            decoder_positional_encodings_rel.append(decoder_positional_encoding_rel)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_feat_list = []
        cls_pred_rel_list = []
        cls_pred_sub_list = []
        cls_pred_obj_list = []
        mask_pred_sub_list = []
        mask_pred_obj_list = []
        (
            cls_pred_sub,
            cls_pred_obj,
            cls_pred_rel,
            mask_pred_sub,
            mask_pred_obj,
            attn_mask,
        ) = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_sub_list.append(cls_pred_sub)
        cls_pred_obj_list.append(cls_pred_obj)
        cls_pred_rel_list.append(cls_pred_rel)
        mask_pred_sub_list.append(mask_pred_sub)
        mask_pred_obj_list.append(mask_pred_obj)
        for i, layer in enumerate(self.transformer_decoder.layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            (
                cls_pred_sub,
                cls_pred_obj,
                cls_pred_rel,
                mask_pred_sub,
                mask_pred_sub,
                attn_mask,
            ) = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )
            query_feat_list.append(query_feat)
            cls_pred_sub_list.append(cls_pred_sub)
            cls_pred_obj_list.append(cls_pred_obj)
            cls_pred_rel_list.append(cls_pred_rel)
            mask_pred_sub_list.append(mask_pred_sub)
            mask_pred_obj_list.append(mask_pred_obj)

        all_cls_scores = dict(
            sub=torch.stack(cls_pred_sub_list)[-1:],
            obj=torch.stack(cls_pred_obj_list)[-1:],
            rel=torch.stack(cls_pred_rel_list)[-1:],
        )
        all_bbox_preds = dict(
            sub_seg=torch.stack(mask_pred_sub_list)[-1:],
            obj_seg=torch.stack(mask_pred_obj_list)[-1:],
        )

        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(
        self,
        all_cls_scores_list,
        all_bbox_preds_list,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert (
            gt_bboxes_ignore is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        all_s_cls_scores = all_cls_scores["sub"]
        all_o_cls_scores = all_cls_scores["obj"]

        # all_s_bbox_preds = all_bbox_preds["sub"]
        # all_o_bbox_preds = all_bbox_preds["obj"]

        # num_dec_layers = len(all_s_cls_scores)

        all_s_mask_preds = all_bbox_preds["sub_seg"]
        all_o_mask_preds = all_bbox_preds["obj_seg"]
        # all_s_mask_preds = [all_s_mask_preds for _ in range(num_dec_layers)]
        # all_o_mask_preds = [all_o_mask_preds for _ in range(num_dec_layers)]

        # all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        # img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_gt_bboxes_list = [gt_bboxes_list]
        all_gt_labels_list = [gt_labels_list]
        all_gt_rels_list = [gt_rels_list]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore]
        all_gt_masks_list = [gt_masks_list]
        img_metas_list = [img_metas]

        all_r_cls_scores = all_cls_scores["rel"]
        # all_r_cls_scores = [None for _ in range(num_dec_layers)]

        if self.use_mask:
            (
                s_losses_cls,
                o_losses_cls,
                r_losses_cls,
                # s_losses_bbox,
                # o_losses_bbox,
                s_losses_mask,
                o_losses_mask,
                s_losses_dice,
                o_losses_dice,
            ) = multi_apply(
                self.loss_single,
                all_s_cls_scores,
                all_o_cls_scores,
                all_r_cls_scores,
                all_s_mask_preds,
                all_o_mask_preds,
                all_gt_rels_list,
                all_gt_bboxes_list,
                all_gt_labels_list,
                all_gt_masks_list,
                img_metas_list,
                all_gt_bboxes_ignore_list,
            )
        else:
            all_s_mask_preds = [None for _ in range(num_dec_layers)]
            all_o_mask_preds = [None for _ in range(num_dec_layers)]
            (
                s_losses_cls,
                o_losses_cls,
                r_losses_cls,
                # s_losses_bbox,
                # o_losses_bbox,
                s_mask_losses,
                o_mask_losses,
                s_dice_losses,
                o_dice_losses,
            ) = multi_apply(
                self.loss_single,
                all_s_cls_scores,
                all_o_cls_scores,
                all_r_cls_scores,
                # all_s_bbox_preds,
                # all_o_bbox_preds,
                all_s_mask_preds,
                all_o_mask_preds,
                all_gt_rels_list,
                all_gt_bboxes_list,
                all_gt_labels_list,
                all_gt_masks_list,
                img_metas_list,
                all_gt_bboxes_ignore_list,
            )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["s_loss_cls"] = s_losses_cls[-1]
        loss_dict["o_loss_cls"] = o_losses_cls[-1]
        loss_dict["r_loss_cls"] = r_losses_cls[-1]
        # loss_dict["s_loss_bbox"] = s_losses_bbox[-1]
        # loss_dict["o_loss_bbox"] = o_losses_bbox[-1]
        loss_dict["s_loss_mask"] = s_losses_mask[-1]
        loss_dict["o_loss_mask"] = o_losses_mask[-1]
        if self.use_mask:
            loss_dict["s_loss_dice"] = s_losses_dice[-1]
            loss_dict["o_loss_dice"] = o_losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for (
            s_loss_cls_i,
            o_loss_cls_i,
            r_loss_cls_i,
            # s_loss_bbox_i,
            # o_loss_bbox_i,
            s_mask_loss_i,
            o_mask_loss_i,
            s_dice_loss_i,
            o_dice_loss_i,
        ) in zip(
            s_losses_cls[:-1],
            o_losses_cls[:-1],
            r_losses_cls[:-1],
            # s_losses_bbox[:-1],
            # o_losses_bbox[:-1],
            s_losses_mask[:-1],
            o_losses_mask[:-1],
            s_losses_dice[:-1],
            o_losses_dice[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.s_loss_cls"] = s_loss_cls_i
            loss_dict[f"d{num_dec_layer}.o_loss_cls"] = o_loss_cls_i
            loss_dict[f"d{num_dec_layer}.r_loss_cls"] = r_loss_cls_i
            loss_dict[f"d{num_dec_layer}.s_loss_mask"] = s_mask_loss_i
            loss_dict[f"d{num_dec_layer}.s_loss_dice"] = s_dice_loss_i
            loss_dict[f"d{num_dec_layer}.o_loss_mask"] = o_mask_loss_i
            loss_dict[f"d{num_dec_layer}.o_loss_dice"] = o_dice_loss_i

            num_dec_layer += 1
        return loss_dict

    def loss_single(
        self,
        s_cls_scores,
        o_cls_scores,
        r_cls_scores,
        # s_bbox_preds,
        # o_bbox_preds,
        s_mask_preds,
        o_mask_preds,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = s_cls_scores.size(0)

        s_cls_scores_list = [s_cls_scores[i] for i in range(num_imgs)]
        o_cls_scores_list = [o_cls_scores[i] for i in range(num_imgs)]
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        # s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        # o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        if self.use_mask:
            s_mask_preds_list = [s_mask_preds[i] for i in range(num_imgs)]
            o_mask_preds_list = [o_mask_preds[i] for i in range(num_imgs)]
        else:
            s_mask_preds_list = [None for i in range(num_imgs)]
            o_mask_preds_list = [None for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            s_cls_scores_list,
            o_cls_scores_list,
            r_cls_scores_list,
            # s_bbox_preds_list,
            # o_bbox_preds_list,
            s_mask_preds_list,
            o_mask_preds_list,
            gt_rels_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        (
            s_labels_list,
            o_labels_list,
            r_labels_list,
            s_label_weights_list,
            o_label_weights_list,
            r_label_weights_list,
            # s_bbox_targets_list,
            # o_bbox_targets_list,
            # s_bbox_weights_list,
            # o_bbox_weights_list,
            s_mask_targets_list,
            o_mask_targets_list,
            num_total_pos,
            num_total_neg,
            s_mask_preds_list,
            o_mask_preds_list,
            s_mask_weights_list,
            o_mask_weights_list,
        ) = cls_reg_targets
        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)
        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        s_mask_targets = torch.cat(s_mask_targets_list, dim=0)
        s_mask_weights = torch.stack(s_mask_weights_list, dim=0)
        o_mask_targets = torch.cat(o_mask_targets_list, dim=0)
        o_mask_weights = torch.stack(o_mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        s_cls_scores = s_cls_scores.flatten(0, 1)
        # s_labels = s_labels.flatten(0, 1)
        # s_label_weights = s_label_weights.flatten(0, 1)
        s_class_weight = s_cls_scores.new_tensor(
            self.obj_class_weight, device=s_cls_scores.device
        )
        s_loss_cls = self.sub_loss_cls(
            s_cls_scores,
            s_labels,
            s_label_weights,
            avg_factor=s_class_weight[s_labels].sum(),
        )

        o_cls_scores = o_cls_scores.flatten(0, 1)
        # o_labels = o_labels.flatten(0, 1)
        # o_label_weights = o_label_weights.flatten(0, 1)
        o_class_weight = o_cls_scores.new_tensor(
            self.obj_class_weight, device=o_cls_scores.device
        )
        o_loss_cls = self.obj_loss_cls(
            o_cls_scores,
            o_labels,
            o_label_weights,
            avg_factor=o_class_weight[o_labels].sum(),
        )

        r_cls_scores = r_cls_scores.flatten(0, 1)
        # r_labels = r_labels.flatten(0, 1)
        # r_label_weights = r_label_weights.flatten(0, 1)
        r_class_weight = r_cls_scores.new_tensor(
            self.r_class_weight, device=r_cls_scores.device
        )
        r_loss_cls = self.rel_loss_cls(
            r_cls_scores,
            r_labels,
            r_label_weights,
            avg_factor=r_class_weight[r_labels].sum(),
        )
        # subject mask
        s_num_total_masks = max(
            reduce_mean(s_cls_scores.new_tensor([num_total_pos])), 1
        )
        s_mask_preds = s_mask_preds[s_mask_weights > 0]

        if s_mask_targets.shape[0] == 0:
            # zero match
            s_loss_dice = s_mask_preds.sum()
            s_loss_mask = s_mask_preds.sum()
            return s_loss_cls, s_loss_mask, s_loss_dice
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                s_mask_preds.unsqueeze(1),
                None,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            s_mask_point_targets = point_sample(
                s_mask_targets.unsqueeze(1).float(), points_coords
            ).squeeze(1)
        s_mask_point_preds = point_sample(
            s_mask_preds.unsqueeze(1), points_coords
        ).squeeze(1)
        s_loss_dice = self.sub_loss_dice(
            s_mask_point_preds, s_mask_point_targets, avg_factor=s_num_total_masks
        )
        s_mask_point_preds = s_mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        s_mask_point_targets = s_mask_point_targets.reshape(-1)
        s_loss_mask = self.sub_loss_mask(
            s_mask_point_preds,
            s_mask_point_targets,
            avg_factor=s_num_total_masks * self.num_points,
        )
        # object mask
        o_num_total_masks = max(
            reduce_mean(o_cls_scores.new_tensor([num_total_pos])), 1
        )
        o_mask_preds = o_mask_preds[o_mask_weights > 0]
        if o_mask_targets.shape[0] == 0:
            # zero match
            o_loss_dice = o_mask_preds.sum()
            o_loss_mask = o_mask_preds.sum()
            return o_loss_cls, o_loss_mask, o_loss_dice
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                o_mask_preds.unsqueeze(1),
                None,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            o_mask_point_targets = point_sample(
                o_mask_targets.unsqueeze(1).float(), points_coords
            ).squeeze(1)
        o_mask_point_preds = point_sample(
            o_mask_preds.unsqueeze(1), points_coords
        ).squeeze(1)
        o_loss_dice = self.obj_loss_dice(
            o_mask_point_preds, o_mask_point_targets, avg_factor=o_num_total_masks
        )
        o_mask_point_preds = o_mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        o_mask_point_targets = o_mask_point_targets.reshape(-1)
        o_loss_mask = self.obj_loss_mask(
            o_mask_point_preds,
            o_mask_point_targets,
            avg_factor=o_num_total_masks * self.num_points,
        )

        return (
            s_loss_cls,
            o_loss_cls,
            r_loss_cls,
            # s_loss_bbox,
            # o_loss_bbox,
            s_loss_mask,
            o_loss_mask,
            s_loss_dice,
            o_loss_dice,
        )

    def get_targets(
        self,
        s_cls_scores_list,
        o_cls_scores_list,
        r_cls_scores_list,
        # s_bbox_preds_list,
        # o_bbox_preds_list,
        s_mask_preds_list,
        o_mask_preds_list,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(s_cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            s_labels_list,
            o_labels_list,
            r_labels_list,
            s_label_weights_list,
            o_label_weights_list,
            r_label_weights_list,
            # s_bbox_targets_list,
            # o_bbox_targets_list,
            # s_bbox_weights_list,
            # o_bbox_weights_list,
            s_mask_targets_list,
            o_mask_targets_list,
            pos_inds_list,
            neg_inds_list,
            s_mask_preds_list,
            o_mask_preds_list,
            s_mask_weights_list,
            o_mask_weights_list,
        ) = multi_apply(
            self._get_target_single,
            s_cls_scores_list,
            o_cls_scores_list,
            r_cls_scores_list,
            # s_bbox_preds_list,
            # o_bbox_preds_list,
            s_mask_preds_list,
            o_mask_preds_list,
            gt_rels_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            s_labels_list,
            o_labels_list,
            r_labels_list,
            s_label_weights_list,
            o_label_weights_list,
            r_label_weights_list,
            # s_bbox_targets_list,
            # o_bbox_targets_list,
            # s_bbox_weights_list,
            # o_bbox_weights_list,
            s_mask_targets_list,
            o_mask_targets_list,
            num_total_pos,
            num_total_neg,
            s_mask_preds_list,
            o_mask_preds_list,
            s_mask_weights_list,
            o_mask_weights_list,
        )

    def _get_target_single(
        self,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        # s_bbox_pred,
        # o_bbox_pred,
        s_mask_preds,
        o_mask_preds,
        gt_rels,
        gt_bboxes,
        gt_labels,
        gt_masks,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = s_mask_preds.size(0)
        # gt_sub_bboxes = []
        # gt_obj_bboxes = []

        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        if self.use_mask:
            gt_sub_masks = []
            gt_obj_masks = []

        assert len(gt_masks) == len(gt_bboxes)
        num_gts = gt_rels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=r_cls_score.device)
        for rel_id in range(gt_rels.size(0)):
            # gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            # gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            if self.use_mask:
                gt_sub_masks.append(gt_masks[int(gt_rels[rel_id, 0])].unsqueeze(0))
                gt_obj_masks.append(gt_masks[int(gt_rels[rel_id, 1])].unsqueeze(0))
        gt_sub_masks = torch.cat(gt_sub_masks, dim=0)
        gt_obj_masks = torch.cat(gt_obj_masks, dim=0)
        sub_mask_points_pred = point_sample(
            s_mask_preds.unsqueeze(1), point_coords.repeat(self.num_queries, 1, 1)
        ).squeeze(1)
        obj_mask_points_pred = point_sample(
            o_mask_preds.unsqueeze(1), point_coords.repeat(self.num_queries, 1, 1)
        ).squeeze(1)
        # shape (num_gts, num_points)

        sub_gt_points_masks = point_sample(
            gt_sub_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)
        ).squeeze(1)
        obj_gt_points_masks = point_sample(
            gt_obj_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)
        ).squeeze(1)
        # gt_sub_bboxes = torch.vstack(gt_sub_bboxes).type_as(gt_bboxes).reshape(-1, 4)
        # gt_obj_bboxes = torch.vstack(gt_obj_bboxes).type_as(gt_bboxes).reshape(-1, 4)
        gt_sub_labels = torch.vstack(gt_sub_labels).type_as(gt_labels).reshape(-1)
        gt_obj_labels = torch.vstack(gt_obj_labels).type_as(gt_labels).reshape(-1)
        gt_rel_labels = torch.vstack(gt_rel_labels).type_as(gt_labels).reshape(-1)

        # assigner and sampler, only return subject&object assign result
        assign_result = self.assigner.assign(
            # s_bbox_pred,
            # o_bbox_pred,
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
        )

        # s_sampling_result = self.sampler.sample(
        #     s_assign_result, s_bbox_pred, gt_sub_bboxes
        # )
        # o_sampling_result = self.sampler.sample(
        #     o_assign_result, o_bbox_pred, gt_obj_bboxes
        # )
        sampling_result = self.sampler.sample(assign_result, s_mask_preds, gt_sub_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds  #### no-rel class indices in prediction

        sub_labels = gt_labels.new_full(
            (self.num_queries,), self.num_classes, dtype=torch.long
        )
        sub_labels[pos_inds] = gt_sub_labels[sampling_result.pos_assigned_gt_inds]

        obj_labels = gt_labels.new_full(
            (self.num_queries,), self.num_classes, dtype=torch.long
        )
        obj_labels[pos_inds] = gt_obj_labels[sampling_result.pos_assigned_gt_inds]

        rel_labels = gt_labels.new_full(
            (self.num_queries,), self.num_relations, dtype=torch.long
        )
        rel_labels[pos_inds] = gt_rel_labels[sampling_result.pos_assigned_gt_inds]

        s_label_weights = gt_labels.new_ones((self.num_queries,))
        o_label_weights = gt_labels.new_ones((self.num_queries,))
        r_label_weights = gt_labels.new_ones((self.num_queries,))

        # mask target
        s_mask_targets = gt_sub_masks[sampling_result.pos_assigned_gt_inds]
        sub_mask_weights = sub_mask_points_pred.new_zeros((self.num_queries,))
        sub_mask_weights[pos_inds] = 1.0

        o_mask_targets = gt_obj_masks[sampling_result.pos_assigned_gt_inds]
        obj_mask_weights = obj_mask_points_pred.new_zeros((self.num_queries,))
        obj_mask_weights[pos_inds] = 1.0

        return (
            sub_labels,
            obj_labels,
            rel_labels,
            s_label_weights,
            o_label_weights,
            r_label_weights,
            # s_bbox_targets,
            # o_bbox_targets,
            # s_bbox_weights,
            # o_bbox_weights,
            s_mask_targets,
            o_mask_targets,
            pos_inds,
            neg_inds,
            s_mask_preds,
            o_mask_preds,
            sub_mask_weights,
            obj_mask_weights,
        )  ###return the interpolated predicted masks

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(
        self,
        x,
        img_metas,
        gt_rels,
        gt_bboxes,
        gt_labels=None,
        gt_masks=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        **kwargs,
    ):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_rels (Tensor): Ground truth relation triplets for one image with
                shape (num_gts, 3) in [gt_sub_id, gt_obj_id, gt_rel_class] format.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds"))
    def get_bboxes(self, cls_scores, mask_preds, img_metas, rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            s_cls_score = cls_scores["sub"][img_id, ...]
            o_cls_score = cls_scores["obj"][img_id, ...]
            r_cls_score = cls_scores["rel"][img_id, ...]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            s_mask_pred = mask_preds["sub_seg"][img_id, ...]
            o_mask_pred = mask_preds["obj_seg"][img_id, ...]
            triplets = self._get_bboxes_single(
                s_cls_score,
                o_cls_score,
                r_cls_score,
                s_mask_pred,
                o_mask_pred,
                img_shape,
                scale_factor,
                rescale,
            )
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(
        self,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        s_mask_pred,
        o_mask_pred,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        assert len(s_cls_score) == len(o_cls_score)

        mask_size = (
            round(img_shape[0] / scale_factor[1]),
            round(img_shape[1] / scale_factor[0]),
        )

        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1].squeeze(0)
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1].squeeze(0)

        s_labels = s_logits.argmax(-1) + 1
        o_labels = o_logits.argmax(-1) + 1

        r_dists = F.softmax(r_cls_score, dim=-1).squeeze(0)

        complete_labels = torch.cat((s_labels, o_labels), 0)
        #### for panoptic postprocessing ####
        s_mask_pred = F.interpolate(
            s_mask_pred,
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        s_mask_pred = torch.sigmoid(s_mask_pred) > 0.5
        o_mask_pred = F.interpolate(
            o_mask_pred,
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        o_mask_pred = torch.sigmoid(o_mask_pred) > 0.5
        masks = torch.cat((s_mask_pred, o_mask_pred), 0)
        pan_img = torch.ones(mask_size).cpu().to(torch.long)
        # dummy bboxes
        det_bboxes = torch.zeros(
            (self.num_queries * 2, 5), device=complete_labels.device
        )
        # dummy r_scores and r_labels
        r_scores = torch.zeros((self.num_queries), device=complete_labels.device)
        r_labels = torch.zeros((self.num_queries), device=complete_labels.device)
        rel_pairs = torch.arange(len(det_bboxes), dtype=torch.int).reshape(2, -1).T
        # (200, 5), (200), (100, 2), (200, h, w), (h, w), (100), (100), (100, 57)
        return (
            det_bboxes,
            complete_labels,
            rel_pairs,
            masks,
            pan_img,
            r_scores,
            r_labels,
            r_dists,
        )

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
