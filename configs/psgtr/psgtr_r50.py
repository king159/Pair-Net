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
        type="PSGTrHead",
        num_classes=80,
        num_relations=117,
        in_channels=2048,
        transformer=dict(
            type="Transformer",
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        )
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DetrTransformerDecoder",
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=dict(
                        type="MultiheadAttention",
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1,
                    ),
                    feedforward_channels=2048,
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
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        sub_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        sub_loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        sub_loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        sub_focal_loss=dict(type="BCEFocalLoss", loss_weight=2.0),
        sub_dice_loss=dict(type="psgtrDiceLoss", loss_weight=2.0),
        obj_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        obj_loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        obj_loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        obj_focal_loss=dict(type="BCEFocalLoss", loss_weight=2.0),
        obj_dice_loss=dict(type="psgtrDiceLoss", loss_weight=2.0),
        rel_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HTriMatcher",
            s_cls_cost=dict(type="ClassificationCost", weight=1.0),
            s_reg_cost=dict(type="BBoxL1Cost", weight=5.0),
            s_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            o_cls_cost=dict(type="ClassificationCost", weight=1.0),
            o_reg_cost=dict(type="BBoxL1Cost", weight=5.0),
            o_iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            r_cls_cost=dict(type="ClassificationCost", weight=2.0),
        )
    ),
    test_cfg=dict(max_per_img=100),
)
