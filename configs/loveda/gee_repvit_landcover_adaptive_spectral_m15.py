"""RepViT-M1.5 backbone with exact-budget adaptive spectral Landscape Mamba (C2)."""
_base_ = './gee_repvit_landcover_adaptive_spectral.py'

exp_name = 'work_dirs/m15_mamba_landcover_adaptive_spectral'

model_config = dict(
    backbone=dict(
        type='repvit_m1_5',
        out_indices=[5, 11, 37, 42],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_5_distill_450e.pth'),
        freeze_bn=False,
    ),
)
