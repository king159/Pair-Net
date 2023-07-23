# dataset settings
dataset_type = "OIV6Dataset"
ann_file = "./data/oiv6/oiv6.json"
oiv6_root = "./data/oiv6"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadPanopticSceneGraphAnnotations",
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SegRescale", scale_factor=1 / 4),
    dict(type="SceneGraphFormatBundle"),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
            "gt_rels",
            "gt_relmaps",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type="LoadPanopticSceneGraphAnnotations", with_bbox=True, with_rel=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            # NOTE: Do not change the img to DC.
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_bboxes", "gt_labels"]),
            dict(
                type="ToDataContainer",
                fields=(dict(key="gt_bboxes"), dict(key="gt_labels")),
            ),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=oiv6_root,
        seg_prefix=oiv6_root,
        pipeline=train_pipeline,
        split="train",
        all_bboxes=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=oiv6_root,
        seg_prefix=oiv6_root,
        pipeline=test_pipeline,
        split="val",
        all_bboxes=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=oiv6_root,
        seg_prefix=oiv6_root,
        pipeline=test_pipeline,
        split="test",
        all_bboxes=True,
    ),
)
