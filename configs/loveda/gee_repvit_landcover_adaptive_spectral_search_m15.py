"""RepViT-M1.5 C3 search using only the internal LoveDA-Train holdout."""
_base_ = './gee_repvit_landcover_adaptive_spectral_search.py'

exp_name = 'work_dirs/info_token_search_m15_base'

model_config = dict(
    backbone=dict(
        type='repvit_m1_5',
        out_indices=[5, 11, 37, 42],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/repvit_m1_5_distill_450e.pth'),
        freeze_bn=False,
    ),
)
