# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from pairnet.models.frameworks.cnn_factory import creat_cnn


@HEADS.register_module()
class CrossHeadBBox(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        num_relations,
        use_mask=False,
        num_obj_query=100,
        num_rel_query=100,
        transformer=None,
        sync_cls_avg_factor=True,
        embed_dims=256,
        relation_decoder=None,
        num_reg_fcs=2,
        as_two_stage=False,
        with_box_refine=False,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        rel_cls_loss=None,
        subobj_cls_loss=None,
        importance_match_loss=None,
        loss_cls=None,
        loss_bbox=None,
        loss_iou=None,
        train_cfg=None,
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # new config
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.num_rel_query = num_rel_query
        self.use_mask = use_mask
        self.num_queries = num_obj_query
        self.embed_dims = embed_dims
        self.relation_decoder = build_transformer_layer_sequence(relation_decoder)
        self.rel_query_pos_embed = nn.Embedding(self.num_rel_query, self.embed_dims)
        self.rel_key_pos_embed = nn.Embedding(self.num_rel_query * 2, self.embed_dims)
        self.rel_value_pos_embed = nn.Embedding(self.num_rel_query * 2, self.embed_dims)
        self.rel_query_feat = nn.Embedding(self.num_rel_query, self.embed_dims)
        self.update_importance = creat_cnn("conv_tiny")
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.bbox_assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.id_assigner = build_assigner(self.train_cfg.id_assigner)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        ### scene graph losses
        self.rel_cls_loss = build_loss(rel_cls_loss)
        self.subobj_cls_loss = build_loss(subobj_cls_loss)
        self.importance_match_loss = build_loss(importance_match_loss)

        if self.loss_cls.use_sigmoid:
            # deformable_detr set use_sigmoid=True
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # detr part
        self.num_reg_fcs = num_reg_fcs
        self.as_two_stage = as_two_stage
        self.with_box_refine = with_box_refine
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
        )

        self.rel_cls_embed = Linear(self.embed_dims, self.num_relations)
        # self.rel_cls_embed = nn.Sequential(
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.num_relations),
        # )

        # detr init
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
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
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, mlvl_feats, img_metas):
        # deformable detr starts
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord,
        ) = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        index = (
            torch.softmax(outputs_classes[-1], dim=1).max(-1).values.topk(100).indices
        )
        outputs_class = torch.gather(
            outputs_classes[-1], 1, index.unsqueeze(-1).repeat(1, 1, self.num_classes)
        )
        outputs_coords = torch.stack(outputs_coords)
        outputs_coord = torch.gather(
            outputs_coords[-1], 1, index.unsqueeze(-1).repeat(1, 1, 4)
        )

        query_feats = hs.clone().detach().transpose(1, 2)
        query_feats = torch.gather(
            query_feats,
            1,
            index.transpose(0, 1).unsqueeze(-1).repeat(6, 1, 1, self.embed_dims),
        )
        bbox_pred = outputs_coord
        cls_pred = outputs_class

        sub_embed = self.sub_query_update(query_feats)
        obj_embed = self.obj_query_update(query_feats)
        sub_embed = F.normalize(sub_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        obj_embed = F.normalize(obj_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        importance = torch.matmul(sub_embed, obj_embed.transpose(1, 2))
        importance = self.update_importance(importance)
        _, updated_importance_idx = torch.topk(
            importance.flatten(-2, -1), k=self.num_rel_query
        )
        sub_pos = torch.div(updated_importance_idx, 100, rounding_mode="trunc")
        obj_pos = torch.remainder(updated_importance_idx, 100)

        query_feat = query_feats[-1]
        obj_query_feat = torch.gather(
            query_feat,
            0,
            obj_pos.unsqueeze(-1).repeat(1, 1, self.embed_dims).transpose(0, 1),
        )
        sub_query_feat = torch.gather(
            query_feat,
            0,
            sub_pos.unsqueeze(-1).repeat(1, 1, self.embed_dims).transpose(0, 1),
        )

        rel_query_feat = self.rel_query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_query_pos_embed = self.rel_query_pos_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_key_pos_embed = self.rel_key_pos_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_value_pos_embed = self.rel_value_pos_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        pair_feat = torch.cat([sub_query_feat, obj_query_feat], dim=0)
        for layer in self.relation_decoder.layers:
            rel_query_feat = layer(
                query=rel_query_feat,
                key=pair_feat,
                value=pair_feat,
                query_pos=rel_query_pos_embed,
                key_pos=rel_key_pos_embed,
                value_pos=rel_value_pos_embed,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
        rel_query = rel_query_feat.transpose(0, 1)
        rel_preds = self.rel_cls_embed(rel_query)

        sub_cls_preds = torch.gather(
            cls_pred.clone().detach(),
            1,
            sub_pos.unsqueeze(-1).expand(-1, -1, cls_pred.shape[-1]),
        )
        obj_cls_preds = torch.gather(
            cls_pred.clone().detach(),
            1,
            obj_pos.unsqueeze(-1).expand(-1, -1, cls_pred.shape[-1]),
        )
        sub_bbox = torch.gather(
            bbox_pred.clone().detach(),
            1,
            sub_pos[..., None].expand(-1, -1, bbox_pred.shape[-1]),
        )
        obj_bbox = torch.gather(
            bbox_pred.clone().detach(),
            1,
            obj_pos[..., None].expand(-1, -1, bbox_pred.shape[-1]),
        )

        all_cls_scores = dict(
            sub=sub_cls_preds,  # (b, 100, 150)
            obj=obj_cls_preds,  # (b, 100, 150)
            cls=outputs_class,  # (b, 300, 150)
            enc_cls_scores=enc_outputs_class,
            enc_bbox_preds=enc_outputs_coord.sigmoid()
            if enc_outputs_coord is not None
            else None,
            rel=rel_preds,  # (b, 100, 50)
            importance=importance,  # (b, 100, 100)
        )
        all_bbox_preds = dict(
            bbox=outputs_coord,  # (6, b, 300, 4)
            sub_bbox=sub_bbox,  # (b, 100, 4)
            obj_bbox=obj_bbox,  # (b, 100, 4)
        )

        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds"))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        assert (
            gt_bboxes_ignore is None
        ), "Only supports for gt_bboxes_ignore setting to None."

        ############################### obj det ####################################
        all_od_cls_scores = [all_cls_scores["cls"]]
        all_od_bbox_preds = [all_bbox_preds["bbox"]]
        # enc_cls_scores = all_cls_scores["enc_cls_scores"]
        # enc_bbox_preds = all_cls_scores["enc_bbox_preds"]
        # num_dec_layers = len(all_od_cls_scores)
        ############################### gt ####################################
        # all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        # img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list]
        all_gt_bboxes_list = [gt_bboxes_list]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore]
        img_metas_list = [img_metas]
        all_gt_labels_list = [gt_labels_list]
        ############################### scene graph #################################
        # importance = [all_cls_scores["importance"] for _ in range(num_dec_layers)]
        # all_r_cls_scores = [all_cls_scores["rel"] for _ in range(num_dec_layers)]
        # sub_cls_preds = [all_cls_scores["sub"] for _ in range(num_dec_layers)]
        # obj_cls_preds = [all_cls_scores["obj"] for _ in range(num_dec_layers)]
        importance = [all_cls_scores["importance"]]
        all_r_cls_scores = [all_cls_scores["rel"]]
        sub_cls_preds = [all_cls_scores["sub"]]
        obj_cls_preds = [all_cls_scores["obj"]]
        (
            # losses_cls,
            # losses_bbox,
            # losses_iou,
            r_losses_cls,
            sub_losses_cls,
            obj_losses_cls,
            loss_match,
        ) = multi_apply(
            self.loss_single,
            sub_cls_preds,
            obj_cls_preds,
            importance,
            all_od_cls_scores,
            all_od_bbox_preds,
            all_r_cls_scores,
            all_gt_bboxes_list,
            all_gt_rels_list,
            all_gt_labels_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()

        # loss of object detection
        # if enc_cls_scores is not None:
        #     binary_labels_list = [
        #         torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))
        #     ]
        #     enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_single_enc(
        #         enc_cls_scores,
        #         enc_bbox_preds,
        #         gt_bboxes_list,
        #         binary_labels_list,
        #         img_metas,
        #         gt_bboxes_ignore,
        #     )
        #     loss_dict["enc_loss_cls"] = enc_loss_cls
        #     loss_dict["enc_loss_bbox"] = enc_losses_bbox
        #     loss_dict["enc_loss_iou"] = enc_losses_iou

        # loss_dict["loss_cls"] = losses_cls[-1]
        # loss_dict["loss_bbox"] = losses_bbox[-1]
        # loss_dict["loss_iou"] = losses_iou[-1]
        # # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
        #     losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]
        # ):
        #     loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
        #     loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
        #     loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
        #     num_dec_layer += 1

        ## loss of scene graph
        loss_dict["loss_r_cls"] = r_losses_cls[-1]
        loss_dict["loss_sub_cls"] = sub_losses_cls[-1]
        loss_dict["loss_obj_cls"] = obj_losses_cls[-1]
        ### loss of importance score
        loss_dict["loss_match"] = loss_match[-1]

        return loss_dict

    # def loss_single_enc(
    #     self,
    #     cls_scores,
    #     bbox_preds,
    #     gt_bboxes_list,
    #     gt_labels_list,
    #     img_metas,
    #     gt_bboxes_ignore_list=None,
    # ):
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #     cls_reg_targets = self.get_targets_enc(
    #         cls_scores_list,
    #         bbox_preds_list,
    #         gt_bboxes_list,
    #         gt_labels_list,
    #         img_metas,
    #         gt_bboxes_ignore_list,
    #     )
    #     (
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_pos,
    #         num_total_neg,
    #     ) = cls_reg_targets
    #     labels = torch.cat(labels_list, 0)
    #     label_weights = torch.cat(label_weights_list, 0)
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)

    #     # classification loss
    #     cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
    #     cls_avg_factor = max(cls_avg_factor, 1)

    #     loss_cls = self.loss_cls(
    #         cls_scores, labels, label_weights, avg_factor=cls_avg_factor
    #     )

    #     # Compute the average number of gt boxes across all gpus, for
    #     # normalization purposes
    #     num_total_pos = loss_cls.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    #     # construct factors used for rescale bboxes
    #     factors = []
    #     for img_meta, bbox_pred in zip(img_metas, bbox_preds):
    #         img_h, img_w, _ = img_meta["img_shape"]
    #         factor = (
    #             bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
    #             .unsqueeze(0)
    #             .repeat(bbox_pred.size(0), 1)
    #         )
    #         factors.append(factor)
    #     factors = torch.cat(factors, 0)

    #     # DETR regress the relative position of boxes (cxcywh) in the image,
    #     # thus the learning target is normalized by the image size. So here
    #     # we need to re-scale them for calculating IoU loss
    #     bbox_preds = bbox_preds.reshape(-1, 4)
    #     bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
    #     bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

    #     # regression IoU loss, defaultly GIoU loss
    #     loss_iou = self.loss_iou(
    #         bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
    #     )

    #     # regression L1 loss
    #     # loss_bbox = self.loss_bbox(
    #     #     bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
    #     # )
    #     return loss_cls, loss_bbox, loss_iou

    def loss_single(
        self,
        sub_cls_preds,
        obj_cls_preds,
        importance,
        od_cls_scores,
        bbox_preds,
        r_cls_scores,
        gt_bboxes_list,
        gt_rels_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list,
    ):
        ## before get targets
        num_imgs = od_cls_scores.size(0)
        # obj det&seg
        cls_scores_list = [od_cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # scene graph
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        subject_scores_list = [sub_cls_preds[i] for i in range(num_imgs)]
        object_scores_list = [obj_cls_preds[i] for i in range(num_imgs)]

        (
            # labels_list,
            # label_weights_list,
            # bbox_targets_list,
            # bbox_weights_list,
            # num_total_pos,
            # num_total_neg,
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        ) = self.get_targets(
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            bbox_preds_list,
            r_cls_scores_list,
            gt_bboxes_list,
            gt_rels_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        ############################### obj det ####################################
        # labels = torch.cat(labels_list, 0)
        # label_weights = torch.cat(label_weights_list, 0)
        # bbox_targets = torch.cat(bbox_targets_list, 0)
        # bbox_weights = torch.cat(bbox_weights_list, 0)
        # cls_scores = od_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        # if self.sync_cls_avg_factor:
        #     cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        # cls_avg_factor = max(cls_avg_factor, 1)
        # loss_cls = self.loss_cls(
        #     cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        # )
        # # Compute the average number of gt boxes across all gpus, for
        # # normalization purposes
        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # # construct factors used for rescale bboxes
        # factors = []
        # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
        #     img_h, img_w, _ = img_meta["img_shape"]
        #     factor = (
        #         bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
        #         .unsqueeze(0)
        #         .repeat(bbox_pred.size(0), 1)
        #     )
        #     factors.append(factor)
        # factors = torch.cat(factors, 0)

        # # DETR regress the relative position of boxes (cxcywh) in the image,
        # # thus the learning target is normalized by the image size. So here
        # # we need to re-scale them for calculating IoU loss
        # bbox_preds = bbox_preds.reshape(-1, 4)
        # bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        # bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(
        #     bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        # )

        # # regression L1 loss
        # loss_bbox = self.loss_bbox(
        #     bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        # )

        ############################### scene graph #################################
        r_label_weights = torch.cat(r_label_weights_list, 0)
        # only calculate the loss of the positive samples in SGG
        r_label_weights_mask = r_label_weights > 0
        dummy_objectness = torch.zeros((r_label_weights_mask.sum(), 2)).to(
            r_cls_scores.device
        )
        # gt_object_ids = torch.cat(gt_object_id_list, 0)
        # loss_obj_cls = self.subobj_cls_loss(
        #     torch.cat(
        #         [obj_cls_preds.flatten(0, 1)[r_label_weights_mask], dummy_objectness],
        #         dim=1,
        #     ),
        #     gt_object_ids[r_label_weights_mask],
        # )["loss_cls_classes"]

        # gt_subject_ids = torch.cat(gt_subject_id_list, 0)
        # loss_sub_cls = self.subobj_cls_loss(
        #     torch.cat(
        #         [sub_cls_preds.flatten(0, 1)[r_label_weights_mask], dummy_objectness],
        #         dim=1,
        #     ),
        #     gt_subject_ids[r_label_weights_mask],
        # )["loss_cls_classes"]

        gt_subject_ids = torch.cat(gt_subject_id_list, 0)
        loss_sub_cls = self.subobj_cls_loss(
            sub_cls_preds.flatten(0, 1)[r_label_weights_mask],
            gt_subject_ids[r_label_weights_mask],
        )

        gt_object_ids = torch.cat(gt_object_id_list, 0)
        loss_obj_cls = self.subobj_cls_loss(
            obj_cls_preds.flatten(0, 1)[r_label_weights_mask],
            gt_object_ids[r_label_weights_mask],
        )

        r_labels = torch.cat(r_labels_list, 0)
        r_cls_scores = r_cls_scores.reshape(-1, self.num_relations)

        # r_loss_cls = self.rel_cls_loss(
        #     torch.cat([r_cls_scores[r_label_weights_mask], dummy_objectness], dim=1),
        #     r_labels[r_label_weights_mask],
        # )["loss_cls_classes"]
        # for other loss
        r_loss_cls = self.rel_cls_loss(
            r_cls_scores[r_label_weights_mask], r_labels[r_label_weights_mask]
        )
        # importance matrix after update
        gt_importance = torch.stack(gt_importance_list, 0)
        pos_weight = torch.numel(gt_importance) / (gt_importance > 0).sum()
        loss_match = self.importance_match_loss(importance, gt_importance, pos_weight)
        return (
            # loss_cls,
            # loss_bbox,
            # loss_iou,
            r_loss_cls,
            loss_sub_cls,
            loss_obj_cls,
            loss_match,
        )

    # def get_targets_enc(
    #     self,
    #     cls_scores_list,
    #     bbox_preds_list,
    #     gt_bboxes_list,
    #     gt_labels_list,
    #     img_metas,
    #     gt_bboxes_ignore_list=None,
    # ):
    #     assert (
    #         gt_bboxes_ignore_list is None
    #     ), "Only supports for gt_bboxes_ignore setting to None."
    #     num_imgs = len(cls_scores_list)
    #     gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

    #     (
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         pos_inds_list,
    #         neg_inds_list,
    #     ) = multi_apply(
    #         self._get_target_single_enc,
    #         cls_scores_list,
    #         bbox_preds_list,
    #         gt_bboxes_list,
    #         gt_labels_list,
    #         img_metas,
    #         gt_bboxes_ignore_list,
    #     )
    #     num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    #     num_total_neg = sum((inds.numel() for inds in neg_inds_list))
    #     return (
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_pos,
    #         num_total_neg,
    #     )

    # def _get_target_single_enc(
    #     self,
    #     cls_score,
    #     bbox_pred,
    #     gt_bboxes,
    #     gt_labels,
    #     img_meta,
    #     gt_bboxes_ignore=None,
    # ):

    #     num_bboxes = bbox_pred.size(0)
    #     # assigner and sampler
    #     assign_result = self.bbox_assigner.assign(
    #         bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
    #     )
    #     sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
    #     pos_inds = sampling_result.pos_inds
    #     neg_inds = sampling_result.neg_inds

    #     # label targets
    #     labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
    #     labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
    #     label_weights = gt_bboxes.new_ones(num_bboxes)

    #     # bbox targets
    #     bbox_targets = torch.zeros_like(bbox_pred)
    #     bbox_weights = torch.zeros_like(bbox_pred)
    #     bbox_weights[pos_inds] = 1.0
    #     img_h, img_w, _ = img_meta["img_shape"]

    #     # DETR regress the relative position of boxes (cxcywh) in the image.
    #     # Thus the learning target should be normalized by the image size, also
    #     # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
    #     factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    #     pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
    #     pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
    #     bbox_targets[pos_inds] = pos_gt_bboxes_targets
    #     return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        subject_scores_list,
        object_scores_list,
        cls_scores_list,
        bbox_preds_list,
        r_cls_scores_list,
        gt_bboxes_list,
        gt_rels_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(r_cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            # labels_list,
            # label_weights_list,
            # bbox_targets_list,
            # bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        ) = multi_apply(
            self._get_target_single,
            subject_scores_list,
            object_scores_list,
            cls_scores_list,
            bbox_preds_list,
            r_cls_scores_list,
            gt_bboxes_list,
            gt_rels_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        # num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        # num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            # labels_list,
            # label_weights_list,
            # bbox_targets_list,
            # bbox_weights_list,
            # num_total_pos,
            # num_total_neg,
            r_labels_list,
            r_label_weights_list,
            gt_subject_id_list,
            gt_object_id_list,
            gt_importance_list,
        )

    def _get_target_single(
        self,
        subject_score,
        object_score,
        cls_score,
        bbox_pred,
        r_cls_score,
        gt_bboxes,
        gt_rels,
        gt_labels,
        img_metas,
        gt_bboxes_ignore,
    ):
        ############################### obj det ####################################
        # num_bboxes = bbox_pred.size(0)
        assign_result = self.bbox_assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        # labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_bboxes.new_ones(num_bboxes)
        # label_weights[pos_inds] = 1.0

        # # bbox targets
        # bbox_targets = torch.zeros_like(bbox_pred)
        # bbox_weights = torch.zeros_like(bbox_pred)
        # bbox_weights[pos_inds] = 1.0
        # img_h, img_w, _ = img_metas["img_shape"]
        # factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        # pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        # pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        # bbox_targets[pos_inds] = pos_gt_bboxes_targets

        ############################### scene graph #################################
        gt_label_assigned_query = torch.ones_like(gt_labels)
        # gt label pos: object query pos
        gt_label_assigned_query[sampling_result.pos_assigned_gt_inds] = pos_inds
        gt_rels = gt_rels.T.long()
        # back to 0-based
        gt_rel_labels = gt_rels[2] - 1
        gt_sub_cls = gt_labels[gt_rels[0]]
        gt_obj_cls = gt_labels[gt_rels[1]]
        gt_sub_pos = gt_label_assigned_query[gt_rels[0]]
        gt_obj_pos = gt_label_assigned_query[gt_rels[1]]

        # gt_importance = torch.zeros(
        #     (self.num_queries, self.num_queries), device=gt_labels.device
        # )
        gt_importance = torch.zeros((100, 100), device=gt_labels.device)
        gt_importance[gt_sub_pos[:], gt_obj_pos[:]] += 1

        triplet_assign_result = self.id_assigner.assign(
            subject_score,
            object_score,
            r_cls_score,
            gt_sub_cls,
            gt_obj_cls,
            gt_rel_labels,
            img_metas,
            gt_bboxes_ignore,
        )

        triplet_sampling_result = self.sampler.sample(
            triplet_assign_result,
            torch.ones_like(subject_score),
            torch.ones_like(subject_score),
        )

        tri_pos_inds = triplet_sampling_result.pos_inds
        # match id targets
        gt_subject_ids = torch.full(
            (self.num_rel_query,),
            self.num_classes,
            dtype=torch.long,
            device=gt_labels.device,
        )
        gt_subject_ids[tri_pos_inds] = gt_sub_cls[
            triplet_sampling_result.pos_assigned_gt_inds
        ]

        gt_object_ids = torch.full(
            (self.num_rel_query,),
            self.num_classes,
            dtype=torch.long,
            device=gt_labels.device,
        )
        gt_object_ids[tri_pos_inds] = gt_obj_cls[
            triplet_sampling_result.pos_assigned_gt_inds
        ]

        r_labels = torch.full(
            (self.num_rel_query,), -1, dtype=torch.long, device=gt_labels.device
        )
        r_labels[tri_pos_inds] = gt_rel_labels[
            triplet_sampling_result.pos_assigned_gt_inds
        ]
        r_label_weights = gt_labels.new_zeros(self.num_rel_query)
        r_label_weights[tri_pos_inds] = 1.0

        result = (
            # labels,
            # label_weights,
            # bbox_targets,
            # bbox_weights,
            pos_inds,
            neg_inds,
            r_labels,
            r_label_weights,
            gt_subject_ids,
            gt_object_ids,
            gt_importance,
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
            loss_inputs = outs + (gt_rels, gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds"))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            all_cls_score = cls_scores["cls"][-1, img_id, ...]
            all_masks = bbox_preds["bbox"][-1, img_id, ...]
            s_cls_score = cls_scores["sub"][img_id, ...]
            o_cls_score = cls_scores["obj"][img_id, ...]
            r_cls_score = cls_scores["rel"][img_id, ...]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            s_bbox_pred = bbox_preds["sub_bbox"][img_id, ...]
            o_bbox_pred = bbox_preds["obj_bbox"][img_id, ...]
            triplets = self._get_bboxes_single(
                all_masks,
                all_cls_score,
                s_cls_score,
                o_cls_score,
                r_cls_score,
                s_bbox_pred,
                o_bbox_pred,
                img_shape,
                scale_factor,
                rescale,
            )
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(
        self,
        all_bbox,
        all_cls_score,
        s_cls_score,
        o_cls_score,
        r_cls_score,
        s_bbox_pred,
        o_bbox_pred,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        assert len(s_cls_score) == len(o_cls_score) == len(r_cls_score)

        s_logits = F.softmax(s_cls_score, dim=-1)
        o_logits = F.softmax(o_cls_score, dim=-1)
        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)
        s_labels += 1
        o_labels += 1

        r_dists = F.softmax(r_cls_score, dim=-1).reshape(-1, self.num_relations)
        r_dists = torch.cat(
            [torch.zeros(self.num_rel_query, 1, device=r_dists.device), r_dists], dim=-1
        )

        labels = torch.cat((s_labels, o_labels), 0)
        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        det_bboxes = torch.cat((s_det_bboxes, o_det_bboxes), 0)
        rel_pairs = torch.arange(len(det_bboxes), dtype=torch.int).reshape(2, -1).T
        # dummy r_scores and r_labels
        r_scores = torch.zeros((100), device=det_bboxes.device)
        r_labels = torch.zeros((100), device=det_bboxes.device)
        rel_pairs = torch.arange(len(det_bboxes), dtype=torch.int).reshape(2, -1).T

        # all_logits = F.softmax(all_cls_score, dim=-1)
        # all_scores, all_labels = all_logits.max(-1)
        # all_det_bboxes = bbox_cxcywh_to_xyxy(all_bbox)
        # all_det_bboxes[:, 0::2] = all_det_bboxes[:, 0::2] * img_shape[1]
        # all_det_bboxes[:, 1::2] = all_det_bboxes[:, 1::2] * img_shape[0]
        # all_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        # all_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        # if rescale:
        #     all_det_bboxes /= all_det_bboxes.new_tensor(scale_factor)
        # all_det_bboxes = torch.cat((all_det_bboxes, all_scores.unsqueeze(1)), -1)
        return (det_bboxes, labels, rel_pairs, r_scores, r_labels, r_dists)
        # return (all_det_bboxes, all_labels, rel_pairs, r_scores, r_labels, r_dists)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
