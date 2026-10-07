"""RepViT-M1.1 with the land-cover-specific anchor-to-uncertain sparse Mamba decoder.

The deployment model scans 25% of image tokens at strides 8/16/32. Each predicted class receives an
independent, scene-conditioned sequence beginning with a prototype pooled from confident pixels.
"""
_base_ = './gee_ablation.py'

epoch = 96
check_val_every_n_epoch = 4
exp_name = 'work_dirs/m11_mamba_landcover'

dataset_config = dict(
    train_mode=dict(
        transform=dict(
            _delete_=True,
            RandomScale=dict(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
            SmartCropV1=dict(crop_size=512, max_ratio=0.75, ignore_index=7, nopad=False),
            RandomHorizontallyFlip=None,
            RandomVerticalFlip=None,
            RandomRotate90=None,
            RandomGaussianBlur=None,
        ),
    ),
)

model_config = dict(
    backbone=dict(
        type='repvit_m1_1',
        out_indices=[3, 7, 21, 24],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_1_distill_450e.pth'),
        freeze_bn=False,
    ),
    seghead=dict(
        mode='mamba_only',
        in_channel=[64, 128, 256, 512],
        scan_cfg=dict(
            order='landcover',
            ratio=0.25,
            balance=0.5,
            anchor_topk=16,
            pos_bands=4,
            scene_condition=True,
            dirs=2,
            expand=1,
            d_state=8,
            n_heads=4,
            chunk=32,
            stages=(1, 2, 3),
            warmup_epochs=6,
            ramp_epochs=6,
        ),
    ),
)

loss_config = dict(
    loss_name=['CEDiceLoss', 'CELoss', 'CELoss', 'CELoss', 'CELoss'],
    loss_weight=[1.0, 0.4, 0.2, 0.2, 0.1],
    dice_weight=0.5,
)

optimizer_config = dict(
    optimizer=dict(
        type='AdamW',
        lr=6e-4,
        weight_decay=1e-2,
        lr_mode='multi',
        backbone_lr=6e-5,
        backbone_weight_decay=1e-2,
    ),
    scheduler=dict(
        type='WarmupCosine',
        warmup_epochs=5,
        eta_min=1e-6,
        max_epoch=epoch,
    ),
)
