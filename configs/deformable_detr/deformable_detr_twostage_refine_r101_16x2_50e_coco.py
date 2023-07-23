_base_ = "deformable_detr_refine_r50_16x2_50e_coco.py"
model = dict(
    backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    bbox_head=dict(
        num_query=100,
        as_two_stage=True,
    ),
)
