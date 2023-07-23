# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness


@HEADS.register_module()
class CrossHead4(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_relations,
        object_classes,
        predicate_classes,
        num_obj_query=100,
        num_rel_query=100,
        use_mask=True,
        pixel_decoder=None,
        transformer_decoder=None,
        feat_channels=256,
        out_channels=256,
        num_transformer_feat_level=3,
        embed_dims=256,
        relation_decoder=None,
        enforce_decoder_input_project=False,
        n_heads=8,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        rel_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
        sub_id_loss=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
        obj_id_loss=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
        loss_cls=None,
        loss_mask=None,
        loss_dice=None,
        train_cfg=None,
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # new config
        self.num_classes = num_classes
        self.num_rel_query = num_rel_query
        self.num_relations = num_relations
        self.relation_decoder = build_transformer_layer_sequence(relation_decoder)
        self.rel_query_embed = nn.Embedding(self.num_rel_query, feat_channels)
        self.rel_query_feat = nn.Embedding(self.num_rel_query, feat_channels)

        # mask2former init
        self.n_heads = n_heads
        self.embed_dims = embed_dims
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )

        self.num_classes = num_classes
        self.num_queries = num_obj_query
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert (
            pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels
            == num_transformer_feat_level
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

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.mask_assigner = build_assigner(self.train_cfg.mask_assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get("num_points", 12544)
            self.oversample_ratio = self.train_cfg.get("oversample_ratio", 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                "importance_sample_ratio", 0.75
            )
            self.id_assigner = build_assigner(self.train_cfg.id_assigner)

        assert num_obj_query == num_rel_query
        self.num_obj_query = num_obj_query
        self.use_mask = use_mask
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.in_channels = in_channels

        self.class_weight = loss_cls.class_weight
        # self.rel_class_weight = rel_loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        # id losses
        self.rel_loss_cls = build_loss(rel_loss_cls)
        self.sub_id_loss = build_loss(sub_id_loss)
        self.obj_id_loss = build_loss(obj_id_loss)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.rel_cls_out_channels = num_relations + 1

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.rel_cls_embed = Linear(self.embed_dims, self.rel_cls_out_channels)

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

        # init like this?
        if self.relation_decoder is not None:
            for p in self.relation_decoder.parameters():
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
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            convert_dict = {
                ".self_attn.": ".attentions.0.",
                ".ffn.": ".ffns.0.",
                ".multihead_attn.": ".attentions.1.",
                ".decoder1.norm.": ".decoder1.post_norm.",
                ".decoder2.norm.": ".decoder2.post_norm.",
                ".query_embedding.": ".query_embed.",
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]
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
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
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

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas):
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
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
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        query_feat_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )

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
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        cls_preds = torch.stack(cls_pred_list)
        mask_preds = torch.stack(mask_pred_list)
        query_feats = torch.stack(query_feat_list)

        rel_query_feat = self.rel_query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_query_embed = self.rel_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )

        for i, layer in enumerate(self.relation_decoder.layers):
            level_idx = i % self.num_transformer_feat_level
            rel_query_feat = layer(
                query=rel_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=rel_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
        rel_query = rel_query_feat.transpose(0, 1)

        updated_sub_embed = self.sub_query_update(query_feats)
        updated_obj_embed = self.obj_query_update(query_feats)
        # (batch, 100, 256)
        sub_q_normalized = F.normalize(
            updated_sub_embed[-1], p=2, dim=-1, eps=1e-12
        ).transpose(0, 1)
        # (batch, 100, 256)
        obj_q_normalized = F.normalize(
            updated_obj_embed[-1], p=2, dim=-1, eps=1e-12
        ).transpose(0, 1)

        rel_q_normalized = F.normalize(rel_query, p=2, dim=-1, eps=1e-12)
        # rel * obj
        subject_scores = torch.matmul(
            rel_q_normalized, sub_q_normalized.transpose(1, 2)
        )
        object_scores = torch.matmul(rel_q_normalized, obj_q_normalized.transpose(1, 2))
        # (batch, 100)
        _, sub_ids = subject_scores.max(-1)
        _, obj_ids = object_scores.max(-1)

        rel_preds = self.rel_cls_embed(rel_query)

        sub_cls_preds = torch.gather(
            cls_preds[-1], 1, sub_ids.unsqueeze(-1).expand(-1, -1, cls_preds.shape[-1])
        )
        obj_cls_preds = torch.gather(
            cls_preds[-1], 1, obj_ids.unsqueeze(-1).expand(-1, -1, cls_preds.shape[-1])
        )
        sub_seg = torch.gather(
            mask_preds[-1],
            1,
            sub_ids[..., None, None].expand(
                -1, -1, mask_preds.shape[-2], mask_preds.shape[-1]
            ),
        )
        obj_seg = torch.gather(
            mask_preds[-1],
            1,
            obj_ids[..., None, None].expand(
                -1, -1, mask_preds.shape[-2], mask_preds.shape[-1]
            ),
        )

        all_cls_scores = dict(
            sub=sub_cls_preds,  # (b, 100, 134)
            obj=obj_cls_preds,
            cls=cls_preds,  # (9, b, 100, 134)
            rel=rel_preds,  # (b, 100, 57)
            subject_scores=subject_scores,  # (b, 100, 100)
            object_scores=object_scores,  # same
        )
        all_mask_preds = dict(
            mask=mask_preds,  # (9,b,100,h,w)
            sub_seg=sub_seg,  # (b,100,h,w)
            obj_seg=obj_seg,  # same
        )

        return all_cls_scores, all_mask_preds

    @force_fp32(apply_to=("all_cls_scores", "all_mask_preds"))
    def loss(
        self,
        all_cls_scores,
        all_mask_preds,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only supports for gt_bboxes_ignore setting to None."

        # object detection and panoptic segmentation
        all_od_cls_scores = all_cls_scores["cls"]
        num_dec_layers = len(all_od_cls_scores)
        all_od_mask_preds = all_mask_preds["mask"]
        all_s_mask_preds = all_mask_preds["sub_seg"]
        all_o_mask_preds = all_mask_preds["obj_seg"]
        all_s_mask_preds = [all_s_mask_preds for _ in range(num_dec_layers)]
        all_o_mask_preds = [all_o_mask_preds for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = all_cls_scores["rel"]
        subject_scores = all_cls_scores["subject_scores"]
        object_scores = all_cls_scores["object_scores"]
        all_r_cls_scores = [all_r_cls_scores for _ in range(num_dec_layers)]
        subject_scores = [subject_scores for _ in range(num_dec_layers)]
        object_scores = [object_scores for _ in range(num_dec_layers)]

        (
            losses_cls,
            losses_mask,
            losses_dice,
            r_losses_cls,
            loss_subject_match,
            loss_object_match,
        ) = multi_apply(
            self.loss_single,
            subject_scores,
            object_scores,
            all_od_cls_scores,
            all_od_mask_preds,
            all_r_cls_scores,
            all_s_mask_preds,
            all_o_mask_preds,
            all_gt_rels_list,
            all_gt_labels_list,
            all_gt_masks_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()

        # loss of segmentation

        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_mask"] = losses_mask[-1]
        loss_dict["loss_dice"] = losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_mask"] = loss_mask_i
            loss_dict[f"d{num_dec_layer}.loss_dice"] = loss_dice_i
            num_dec_layer += 1

        # loss of scene graph
        loss_dict["loss_subject_match"] = loss_subject_match[-1]
        loss_dict["loss_object_match"] = loss_object_match[-1]
        loss_dict["r_loss_cls"] = r_losses_cls[-1]

        return loss_dict

    def loss_single(
        self,
        subject_scores,
        object_scores,
        od_cls_scores,
        mask_preds,
        r_cls_scores,
        s_mask_preds,
        o_mask_preds,
        gt_rels_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        # before get targets
        num_imgs = od_cls_scores.size(0)
        # obj det&seg
        cls_scores_list = [od_cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # scene graph
        s_mask_preds_list = [s_mask_preds[i] for i in range(num_imgs)]
        o_mask_preds_list = [o_mask_preds[i] for i in range(num_imgs)]
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        subject_scores_list = [subject_scores[i] for i in range(num_imgs)]
        object_scores_list = [object_scores[i] for i in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_od_pos,
            num_total_od_neg,
            r_labels_list,
            r_label_weights_list,
            num_total_pos,
            num_total_neg,
            filtered_subject_scores,
            filtered_object_scores,
            gt_subject_id_list,
            gt_object_id_list,
        ) = self.get_targets(
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            mask_preds_list,
            r_cls_scores_list,
            s_mask_preds_list,
            o_mask_preds_list,
            gt_rels_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        # obj seg
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classification loss
        # shape (batch_size * num_queries, )
        cls_scores = od_cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum()
        )

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_od_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1),
                None,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords
            ).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(
            1
        )

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks
        )

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points,
        )

        # id loss
        loss_subject_match_list = []
        loss_object_match_list = []
        for (
            filtered_subject_score,
            filtered_object_score,
            gt_subject_id,
            gt_object_id,
        ) in zip(
            filtered_subject_scores,
            filtered_object_scores,
            gt_subject_id_list,
            gt_object_id_list,
        ):
            gt_subject_id = F.one_hot(
                gt_subject_id, num_classes=filtered_subject_score.shape[-1]
            )
            gt_object_id = F.one_hot(
                gt_object_id, num_classes=filtered_object_score.shape[-1]
            )
            loss_subject_match_list.append(
                self.sub_id_loss(filtered_subject_score, gt_subject_id)
            )
            loss_object_match_list.append(
                self.obj_id_loss(filtered_object_score, gt_object_id)
            )
        loss_subject_match = torch.stack(loss_subject_match_list, 0).mean()
        loss_object_match = torch.stack(loss_object_match_list, 0).mean()

        # scene graph
        r_labels = torch.cat(r_labels_list, 0)
        # r_label_weights = torch.cat(r_label_weights_list, 0)
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)
        # rel_class_weight = r_cls_scores.new_tensor(self.rel_class_weight)

        # r_loss_cls = self.rel_loss_cls(
        #     r_cls_scores,
        #     r_labels,
        #     r_label_weights,
        #     avg_factor=rel_class_weight[r_labels].sum(),
        # )

        # r_label_weights_mask = r_label_weights > 0

        # for seesaw loss
        dummy_objectness = torch.zeros((200, 2)).to(r_cls_scores.device)
        r_loss_cls = self.rel_loss_cls(
            torch.cat([r_cls_scores, dummy_objectness], dim=1),
            r_labels,
        )["loss_cls_classes"]

        return (
            loss_cls,
            loss_mask,
            loss_dice,
            r_loss_cls,
            loss_subject_match,
            loss_object_match,
        )

    def get_targets(
        self,
        subject_scores_list,
        object_scores_list,
        cls_scores_list,
        mask_preds_list,
        r_cls_scores_list,
        s_mask_preds_list,
        o_mask_preds_list,
        gt_rels_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(r_cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            od_pos_inds_list,
            od_neg_inds_list,
            r_labels_list,
            r_label_weights_list,
            pos_inds_list,
            neg_inds_list,
            filtered_subject_scores,
            filtered_object_scores,
            gt_subject_id_list,
            gt_object_id_list,
        ) = multi_apply(
            self._get_target_single,
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            mask_preds_list,
            r_cls_scores_list,
            s_mask_preds_list,
            o_mask_preds_list,
            gt_rels_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        num_total_od_pos = sum((inds.numel() for inds in od_pos_inds_list))
        num_total_od_neg = sum((inds.numel() for inds in od_neg_inds_list))
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_od_pos,
            num_total_od_neg,
            r_labels_list,
            r_label_weights_list,
            num_total_pos,
            num_total_neg,
            filtered_subject_scores,
            filtered_object_scores,
            gt_subject_id_list,
            gt_object_id_list,
        )

    def _get_target_single(
        self,
        subject_scores,
        object_scores,
        cls_score,
        mask_pred,
        r_cls_score,
        s_mask_pred,
        o_mask_pred,
        gt_rels,
        gt_labels,
        gt_masks,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        # obj seg
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)
        ).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)
        ).squeeze(1)

        # assign and sample
        assign_result = self.mask_assigner.assign(
            cls_score, mask_points_pred, gt_labels, gt_points_masks, img_metas
        )
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        od_pos_inds = sampling_result.pos_inds
        od_neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full(
            (self.num_queries,), self.num_classes, dtype=torch.long
        )
        labels[od_pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[od_pos_inds] = 1.0

        # scene graph
        gt_label_assigned_query = torch.ones_like(gt_labels)
        gt_label_assigned_query[sampling_result.pos_assigned_gt_inds] = od_pos_inds
        num_rels = s_mask_pred.shape[0]

        gt_sub_mask_list = []
        gt_obj_mask_list = []
        gt_rel_label_list = []
        gt_sub_id_list = []
        gt_obj_id_list = []

        for rel_id in range(gt_rels.shape[0]):
            gt_sub_mask_list.append(gt_masks[int(gt_rels[rel_id, 0])])
            gt_obj_mask_list.append(gt_masks[int(gt_rels[rel_id, 1])])
            gt_rel_label_list.append(gt_rels[rel_id, 2])
            gt_sub_id_list.append(gt_label_assigned_query[int(gt_rels[rel_id, 0])])
            gt_obj_id_list.append(gt_label_assigned_query[int(gt_rels[rel_id, 1])])

        gt_sub_masks = torch.stack(gt_sub_mask_list)
        gt_obj_masks = torch.stack(gt_obj_mask_list)
        gt_rel_labels = torch.vstack(gt_rel_label_list).long().flatten()
        gt_sub_ids = torch.vstack(gt_sub_id_list).flatten()
        gt_obj_ids = torch.vstack(gt_obj_id_list).flatten()
        ########################################
        #### overwrite relation labels above####
        ########################################
        # assigner and sampler for relation-oriented id match
        s_assign_result, o_assign_result = self.id_assigner.assign(
            subject_scores,
            object_scores,
            r_cls_score,
            gt_sub_ids,
            gt_obj_ids,
            gt_rel_labels,
            img_metas,
            gt_bboxes_ignore,
        )

        s_sampling_result = self.sampler.sample(
            s_assign_result, s_mask_pred, gt_sub_masks
        )
        o_sampling_result = self.sampler.sample(
            o_assign_result, o_mask_pred, gt_obj_masks
        )
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  # no-rel class indices in prediction

        # match id targets
        gt_subject_ids = gt_sub_masks.new_full((num_rels,), -1, dtype=torch.long)
        gt_subject_ids[pos_inds] = gt_sub_ids[s_sampling_result.pos_assigned_gt_inds]

        gt_object_ids = gt_obj_masks.new_full((num_rels,), -1, dtype=torch.long)
        gt_object_ids[pos_inds] = gt_obj_ids[o_sampling_result.pos_assigned_gt_inds]

        # filtering unmatched subject/object id predictions
        gt_subject_ids = gt_subject_ids[pos_inds]
        gt_subject_ids_res = torch.zeros_like(gt_subject_ids)
        for idx, gt_subject_id in enumerate(gt_subject_ids):
            gt_subject_ids_res[idx] = (od_pos_inds == gt_subject_id).nonzero(
                as_tuple=True
            )[0]
        gt_subject_ids = gt_subject_ids_res

        gt_object_ids = gt_object_ids[pos_inds]
        gt_object_ids_res = torch.zeros_like(gt_object_ids)
        for idx, gt_object_id in enumerate(gt_object_ids):
            gt_object_ids_res[idx] = (od_pos_inds == gt_object_id).nonzero(
                as_tuple=True
            )[0]
        gt_object_ids = gt_object_ids_res

        filtered_subject_scores = subject_scores[pos_inds]
        filtered_subject_scores = filtered_subject_scores[:, od_pos_inds]
        filtered_object_scores = object_scores[pos_inds]
        filtered_object_scores = filtered_object_scores[:, od_pos_inds]

        r_labels = gt_obj_masks.new_full((num_rels,), 0, dtype=torch.long)  # 1-based

        r_labels[pos_inds] = gt_rel_labels[o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_masks.new_ones(num_rels)

        result = (
            labels,
            label_weights,
            mask_targets,
            mask_weights,
            od_pos_inds,
            od_neg_inds,
            r_labels,
            r_label_weights,
            pos_inds,
            neg_inds,
            filtered_subject_scores,
            filtered_object_scores,
            gt_subject_ids,
            gt_object_ids,
        )
        return result

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
            all_cls_score = cls_scores["cls"][-1, img_id, ...]
            all_masks = mask_preds["mask"][-1, img_id, ...]
            s_cls_score = cls_scores["sub"][img_id, ...]
            o_cls_score = cls_scores["obj"][img_id, ...]
            r_cls_score = cls_scores["rel"][img_id, ...]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            s_mask_pred = mask_preds["sub_seg"][img_id, ...]
            o_mask_pred = mask_preds["obj_seg"][img_id, ...]
            triplets = self._get_bboxes_single(
                all_masks,
                all_cls_score,
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
        all_masks,
        all_cls_score,
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
        max_per_img = self.test_cfg.get("max_per_img", self.num_obj_query)

        assert self.rel_loss_cls.use_sigmoid == False
        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = torch.div(r_indexes, self.num_relations, rounding_mode="trunc")

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index] + 1
        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index] + 1
        r_dists = r_lgs.reshape(-1, self.num_relations + 1)[
            triplet_index
        ]  # NOTE: to match the evaluation in vg
        complete_labels = torch.cat((s_labels, o_labels), 0)

        s_mask_pred = s_mask_pred[triplet_index]
        o_mask_pred = o_mask_pred[triplet_index]
        all_logits = F.softmax(all_cls_score, dim=-1)[..., :-1]

        all_scores, all_labels = all_logits.max(-1)
        all_masks = F.interpolate(
            all_masks.unsqueeze(1), size=mask_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        #### for panoptic postprocessing ####
        s_mask_pred = F.interpolate(
            s_mask_pred.unsqueeze(1),
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        s_mask_pred = torch.sigmoid(s_mask_pred) > 0.5
        o_mask_pred = F.interpolate(
            o_mask_pred.unsqueeze(1),
            size=mask_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        o_mask_pred = torch.sigmoid(o_mask_pred) > 0.5
        masks = torch.cat((s_mask_pred, o_mask_pred), 0)

        keep = (all_labels != s_logits.shape[-1] - 1) & (
            all_scores > 0.5
        )  # the threshold is set to 0.5
        all_labels = all_labels[keep]
        all_masks = all_masks[keep]
        all_scores = all_scores[keep]
        h, w = all_masks.shape[-2:]

        if all_labels.numel() == 0:
            pan_img = torch.ones(mask_size).cpu().to(torch.long)
        else:
            all_masks = all_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(all_labels):
                if label.item() >= 80:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(all_masks, all_scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = all_masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                seg_img = m_id * INSTANCE_OFFSET + all_labels[m_id]
                seg_img = seg_img.view(h, w).cpu().to(torch.long)
                m_id = m_id.view(h, w).cpu()
                area = []
                for i in range(len(all_scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, pan_img = get_ids_area(all_masks, all_scores, dedup=True)
            if all_labels.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(all_labels)],
                        dtype=torch.bool,
                        device=keep.device,
                    )
                    if filtered_small.any().item():
                        all_scores = all_scores[~filtered_small]
                        all_labels = all_labels[~filtered_small]
                        all_masks = all_masks[~filtered_small]
                        area, pan_img = get_ids_area(all_masks, all_scores)
                    else:
                        break
        # dummy bboxes for eval
        det_bboxes = torch.rand((200, 5), device=complete_labels.device)
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
