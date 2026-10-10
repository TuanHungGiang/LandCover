"""RepViT-M1.5 backbone with coverage-preserving quarter-token spatial Mamba."""
_base_ = './gee_repvit_landcover_coverage.py'

exp_name = 'work_dirs/m15_mamba_landcover_coverage'

model_config = dict(
    backbone=dict(
        type='repvit_m1_5',
        out_indices=[5, 11, 37, 42],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_5_distill_450e.pth'),
        freeze_bn=False,
    ),
)
