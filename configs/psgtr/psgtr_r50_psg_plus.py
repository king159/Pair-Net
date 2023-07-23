_base_ = [
    "../_base_/datasets/psg.py",
    "../_base_/custom_runtime.py",
]

find_unused_parameters = True

custom_imports = dict(
    imports=[
        "pairnet.models.frameworks.psgtr",
        "pairnet.models.losses.seg_losses",
        "pairnet.models.relation_heads.psgtr_head",
        "pairnet.datasets",
        "pairnet.datasets.pipelines.loading",
        "pairnet.datasets.pipelines.rel_randomcrop",
        "pairnet.models.relation_heads.approaches.matcher",
        "pairnet.utils",
    ],
    allow_failed_imports=False,
)

dataset_type = "PanopticSceneGraphDataset"

num_object_classes = 133
num_relation_classes = 56

model = dict(
    type="PSGTr",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    bbox_head=dict(
        type="PSGTrHead2",
        num_classes=num_object_classes,
        num_relations=num_relation_classes,
        use_mask=True,
        num_obj_query=100,
        pixel_decoder=dict(
            type="MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention",
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
                init_cfg=None,
            ),
        ),
        transformer_decoder=dict(
            type="DetrTransformerDecoder",
            return_intermediate=False,
            num_layers=9,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=dict(
                    type="MultiheadAttention",
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False,
                ),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
                operation_order=(
                    "cross_attn",
                    "norm",
                    "self_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        sub_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        sub_loss_mask=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0
        ),
        sub_loss_dice=dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        obj_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        obj_loss_mask=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0
        ),
        obj_loss_dice=dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        rel_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        sampler=dict(type="MaskPseudoSampler"),
        assigner=dict(
            type="MaskHTriMatcher",
            s_cls_cost=dict(type="ClassificationCost", weight=2.0),
            s_mask_cost=dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
            s_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
            o_cls_cost=dict(type="ClassificationCost", weight=2.0),
            o_mask_cost=dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
            o_dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
            r_cls_cost=dict(type="ClassificationCost", weight=2.0),
        ),
    ),
    test_cfg=dict(max_per_img=100),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadPanopticSceneGraphAnnotations",
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RelRandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=False,
                ),  # no empty relations
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="RelsFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_rels", "gt_masks"]),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            # dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            # dict(type='ToDataContainer', fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

evaluation = dict(
    interval=100000000000,
    metric="sgdet",
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method="pan_seg",
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    pin_memory=True,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
# optimizer
optimizer = dict(
    type="AdamW",
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1, decay_mult=1.0),
            "transformer_decoder": dict(lr_mult=0.1, decay_mult=1),
            "pixel_decoder": dict(lr_mult=0.1, decay_mult=1),
            "decoder_input_projs": dict(lr_mult=0.1, decay_mult=1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy="step", step=40)
runner = dict(type="EpochBasedRunner", max_epochs=60)

project_name = "psgformer"
expt_name = "psgtr_plus"
work_dir = f"./work_dirs/{expt_name}"
checkpoint_config = dict(interval=2, max_keep_ckpts=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project=project_name,
        #         name=expt_name,
        #         # config=work_dir + "/cfg.yaml"
        #     ),
        # )
    ],
)
load_from = "./pretrain/m2f_r50_coco.pth"
