_base_ = [
    "../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py",
]

custom_imports = dict(
    imports=[
        "pairnet.models.frameworks.psgtr",
        "pairnet.models.losses.seg_losses",
        "pairnet.models.frameworks.dual_transformer",
        "pairnet.models.relation_heads.psgformer_head",
        "pairnet.datasets",
        "pairnet.datasets.pipelines.loading",
        "pairnet.datasets.pipelines.rel_randomcrop",
        "pairnet.models.relation_heads.approaches.matcher",
        "pairnet.utils",
    ],
    allow_failed_imports=False,
)

model = dict(
    relation_head=dict(
        type="GPSHead",
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(type="SceneGraphBBoxHead"),
    ),
)

evaluation = dict(
    interval=1,
    metric="sgdet",
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method="pan_seg",
)

data = dict(samples_per_gpu=16)

# Log config
project_name = "pairnet"
expt_name = "gpsnet_panoptic_fpn_r50_fpn_1x_sgdet_psg"
work_dir = f"./work_dirs/{expt_name}"

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
