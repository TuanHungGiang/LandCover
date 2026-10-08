"""RepViT-M1.5 backbone with the land-cover Mamba v2 decoder."""
_base_ = './gee_repvit_landcover_v2.py'

exp_name = 'work_dirs/m15_mamba_landcover_v2'

model_config = dict(
    backbone=dict(
        type='repvit_m1_5',
        out_indices=[5, 11, 37, 42],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_5_distill_450e.pth'),
        freeze_bn=False,
    ),
)
