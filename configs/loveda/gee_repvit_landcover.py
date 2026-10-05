"""RepViT-M1.1 with the land-cover-specific anchor-to-uncertain sparse Mamba decoder.

The deployment model stays below the M1.1 accuracy/efficiency branch: no class-center attention and scans
25% of image tokens at strides 8/16/32.  Each predicted class receives an independent, scene-conditioned
sequence beginning with a prototype pooled from confident pixels.
"""
_base_ = './gee_ablation.py'

epoch = 64
check_val_every_n_epoch = 8
exp_name = 'work_dirs/m11_mamba_landcover'

model_config = dict(
    backbone=dict(
        type='repvit_m1_1',
        out_indices=[3, 7, 21, 24],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_1_distill_450e.pth'),
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
        ),
    ),
)

optimizer_config = dict(scheduler=dict(max_epoch=epoch))
