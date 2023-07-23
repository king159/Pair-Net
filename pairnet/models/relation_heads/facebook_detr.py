# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from pairnet.models.frameworks.unet import UNet


@HEADS.register_module()
class FacebookHead(AnchorFreeHead):
    """This class is for testing the detr mIoU"""

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
        rel_cls_loss=None,
        subobj_cls_loss=None,
        importance_match_loss=None,
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
        self.use_mask = True
        self.facebook_model, _ = torch.hub.load(
            "facebookresearch/detr",
            "detr_resnet101_panoptic",
            pretrained=True,
            return_postprocessor=True,
            num_classes=250,
        )
        self.facebook2ours = torch.load("/home/jhwang/PSG-mix/mapping.pt").cuda()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        # self.sub_query_update = nn.Sequential(
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims),
        # )
        self.sub_query_update = nn.Identity()
        self.obj_query_update = nn.Identity()

        # self.obj_query_update = nn.Sequential(
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims),
        # )

        self.rel_cls_embed = Linear(self.embed_dims, self.num_relations)

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

        for p in self.relation_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for p in self.update_importance.parameters():
            if p.dim() > 1:
                nn.init.dirac_(p)

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

    def forward(self, img, img_metas):
        result = self.facebook_model(img)
        cls_pred = result["pred_logits"]
        mask_pred = result["pred_masks"]

        all_cls_scores = dict(
            cls=cls_pred,  # (b, 100, 134)
        )
        all_mask_preds = dict(
            mask=mask_pred,  # (b,100,h,w)
        )

        return all_cls_scores, all_mask_preds

    def loss():
        pass

    def get_bboxes():
        pass

    def get_targets():
        pass

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds"))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            all_cls_score = cls_scores["cls"][img_id, ...]
            all_masks = bbox_preds["mask"][img_id, ...]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            triplets = self._get_bboxes_single(
                all_masks, all_cls_score, img_shape, scale_factor, rescale
            )
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(
        self,
        all_masks,
        all_cls_score,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        mask_size = (
            round(img_shape[0] / scale_factor[1]),
            round(img_shape[1] / scale_factor[0]),
        )

        # 0-based label input for objects and self.num_classes as default background cls
        all_logits = F.softmax(all_cls_score, dim=-1)[..., :-1]

        all_scores, all_labels = all_logits.max(-1)
        all_masks = F.interpolate(
            all_masks.unsqueeze(1), size=mask_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        keep = (all_labels != all_logits.shape[-1] - 1) & (
            all_scores > 0.85
        )  ## the threshold is set to 0.85
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
        all_labels = self.facebook2ours[all_labels]
        all_labels = torch.cat([all_labels] * 2) + 1
        all_masks = F.sigmoid(all_masks) > 0.5
        all_masks = torch.cat([all_masks] * 2)
        all_masks = all_masks.view(-1, h, w)
        # dummy bboxes
        det_bboxes = torch.zeros((all_labels.shape[0], 5), device=all_labels.device)
        # dummy r_scores and r_labels
        r_scores = torch.zeros((all_labels.shape[0] // 2), device=all_labels.device)
        r_labels = torch.zeros((all_labels.shape[0] // 2), device=all_labels.device)
        rel_pairs = torch.arange(len(det_bboxes), dtype=torch.int).reshape(2, -1).T
        r_dists = torch.zeros((all_labels.shape[0] // 2, 57), device=all_labels.device)
        # (200, 5), (200), (100, 2), (200, h, w), (h, w), (100), (100), (100, 57)
        return (
            det_bboxes,
            all_labels,
            rel_pairs,
            all_masks,
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


import mmcv
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER


@ATTENTION.register_module(force=True)
class MultiheadAttention2(mmcv.cnn.bricks.transformer.MultiheadAttention):
    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer,
            init_cfg,
            batch_first,
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out, map = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out)), map


@TRANSFORMER_LAYER.register_module(force=True)
class BaseTransformerLayer2(mmcv.cnn.bricks.transformer.BaseTransformerLayer):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__(
            attn_cfgs, ffn_cfgs, operation_order, norm_cfg, init_cfg, batch_first
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query, self_attn_map = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query, cross_attn_map = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query, self_attn_map, cross_attn_map
