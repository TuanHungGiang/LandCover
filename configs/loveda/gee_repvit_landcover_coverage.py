"""RepViT-M1.1 + coverage-preserving quarter-token spatial Mamba.

Every 2x2 feature cell contributes to one context token, so Mamba processes exactly 25% spatial
tokens without top-k dropping 75% of the image. The dense decoder residual preserves the original
feature, while soft scene-specific class prototypes condition the spatial sequence without argmax
routing. Backbone, loss and training recipe are inherited unchanged from the V1 configuration.
"""
_base_ = './gee_repvit_landcover.py'

exp_name = 'work_dirs/m11_mamba_landcover_coverage'

model_config = dict(
    seghead=dict(
        scan_cfg=dict(
            _delete_=True,
            block_type='coverage',
            ratio=0.25,
            expand=1,
            d_state=8,
            n_heads=4,
            chunk=32,
            pos_bands=4,
            soft_prototypes=True,
            stages=(1, 2, 3),
            warmup_epochs=6,
            ramp_epochs=6,
        ),
    ),
)
