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
        type="PSGFormerHead",
        num_classes=80,
        num_relations=117,
        in_channels=2048,
        transformer=dict(
            type="DualTransformer",
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
            decoder1=dict(
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
            decoder2=dict(
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
        rel_loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=1.0,
        ),
        sub_id_loss=dict(type="MultilabelCrossEntropy", loss_weight=2.0),
        obj_id_loss=dict(type="MultilabelCrossEntropy", loss_weight=2.0),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=4.0,
            class_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=3.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        focal_loss=dict(type="BCEFocalLoss", loss_weight=1.0),
        dice_loss=dict(type="psgtrDiceLoss", loss_weight=1.0),
    ),
    # training and testing settings
    train_cfg=dict(
        id_assigner=dict(
            type="IdMatcher",
            sub_id_cost=dict(type="ClassificationCost", weight=1.0),
            obj_id_cost=dict(type="ClassificationCost", weight=1.0),
            r_cls_cost=dict(type="ClassificationCost", weight=1.0),
        ),
        bbox_assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="ClassificationCost", weight=4.0),
            reg_cost=dict(type="BBoxL1Cost", weight=3.0),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        ),
    ),
    test_cfg=dict(max_per_img=100),
)
