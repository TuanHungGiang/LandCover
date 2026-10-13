"""RepViT-M1.1 + exact-budget adaptive-resolution spectral Landscape Mamba (C2).

Starting from C1's N/4 uniform context tokens, every homogeneous 4x4 merge funds one complex 2x2
split into four pixel tokens.  This preserves full pixel coverage and the exact 25% SSM sequence
length while redistributing resolution.  A sparse feature/boundary graph supplies an approximate
Fiedler field; variable-size region tokens are ordered by its pixel-mass average.
"""
_base_ = './gee_repvit_landcover.py'

exp_name = 'work_dirs/m11_mamba_landcover_adaptive_spectral'
# The ablation base disables ``last.ckpt`` to avoid writing dozens of short-run
# checkpoints.  This is a long final run, so retain the most recent validated
# state as well as the best checkpoint; otherwise a transient Kaggle I/O error
# can discard many hours of training.
save_last = True

model_config = dict(
    seghead=dict(
        scan_cfg=dict(
            _delete_=True,
            block_type='adaptive_coverage',
            ratio=0.25,
            adapt_fraction=0.25,
            boundary_weight=0.75,
            entropy_weight=0.5,
            rarity_weight=0.25,
            spectral_iters=6,
            spectral_temperature=0.5,
            spectral_self_loop=1.0,
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
