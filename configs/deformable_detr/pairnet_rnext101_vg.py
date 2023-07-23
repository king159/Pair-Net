_base_ = ["../_base_/datasets/vg_detection.py", "../_base_/custom_runtime.py"]

num_object_classes = 150
num_relation_classes = 50
find_unused_parameters = True

model = dict(
    type="PSGTr",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=8,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        with_cp=True,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type="CrossHeadBBox",
        num_classes=num_object_classes,
        num_relations=num_relation_classes,
        num_obj_query=300,
        num_rel_query=100,
        in_channels=2048,
        sync_cls_avg_factor=True,
        embed_dims=256,
        as_two_stage=True,
        with_box_refine=True,
        transformer=dict(
            type="DeformableDetrTransformer",
            as_two_stage=True,
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention", embed_dims=256
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DeformableDetrTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(type="MultiScaleDeformableAttention", embed_dims=256),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        relation_decoder=dict(
            type="DetrTransformerDecoder",
            return_intermediate=True,
            num_layers=6,
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
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        rel_cls_loss=dict(
            type="SeesawLoss",
            num_classes=num_relation_classes,
            return_dict=True,
            loss_weight=2.0,
        ),
        subobj_cls_loss=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=4.0,
            reduction="mean",
            class_weight=[1.0] * num_object_classes,
        ),
        importance_match_loss=dict(
            type="BCEWithLogitsLoss",
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        id_assigner=dict(
            type="IdMatcher",
            sub_id_cost=dict(type="ClassificationCost", weight=1.0),
            obj_id_cost=dict(type="ClassificationCost", weight=1.0),
            r_cls_cost=dict(type="ClassificationCost", weight=0.0),
        ),
        bbox_assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        ),
        sampler=dict(type="PseudoSampler"),
    ),
    test_cfg=dict(max_per_img=100),
)


custom_imports = dict(
    imports=[
        "pairnet.models.frameworks.psgtr",
        "pairnet.models.losses.seg_losses",
        "pairnet.datasets",
        "pairnet.datasets.pipelines.loading",
        "pairnet.datasets.pipelines.rel_randomcrop",
        "pairnet.models.relation_heads.approaches.matcher",
        "pairnet.utils",
    ],
    allow_failed_imports=False,
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadSceneGraphAnnotations",
        with_bbox=True,
        with_rel=True,
        with_mask=False,
        with_seg=False,
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
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
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
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_rels"]),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadSceneGraphAnnotations", with_bbox=True, with_rel=True),
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
            dict(type="ToTensor", keys=["gt_bboxes", "gt_labels"]),
            dict(
                type="ToDataContainer",
                fields=(dict(key="gt_bboxes"), dict(key="gt_labels")),
            ),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

evaluation = dict(
    interval=1,
    metric="sgdet",
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
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
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        },
        norm_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy="step", step=[40])
runner = dict(type="EpochBasedRunner", max_epochs=24)

project_name = "OpenImages"
expt_name = "3091_rnext"
work_dir = f"./work_dirs/{expt_name}"
checkpoint_config = dict(interval=1, max_keep_ckpts=15)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
            ),
        ),
    ],
)
auto_scale_lr = dict(enable=True, base_batch_size=8)
load_from = "/home/jhwang/PSG-mix/work_dirs/od_rnext101_vg/epoch_11.pth"
