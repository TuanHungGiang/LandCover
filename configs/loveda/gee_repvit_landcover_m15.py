"""Accuracy-oriented land-cover sparse Mamba model with a RepViT-M1.5 backbone.

Train:
    python train.py -c configs/loveda/gee_repvit_landcover_m15.py
"""
_base_ = './gee_repvit_landcover.py'

exp_name = 'work_dirs/m15_mamba_landcover'

model_config = dict(
    backbone=dict(
        type='repvit_m1_5',
        out_indices=[5, 11, 37, 42],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_5_distill_450e.pth'),
        freeze_bn=False,
    ),
)
