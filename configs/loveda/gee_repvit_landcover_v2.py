"""Land-cover Mamba v2: rank-normalized 25% routing, dual prototypes and confidence gating.

The loss, RepViT-M1.1 backbone and training recipe are inherited unchanged from the v1 baseline so the
accuracy difference isolates the land-cover-specific sparse Mamba novelty.
"""
_base_ = './gee_repvit_landcover.py'

exp_name = 'work_dirs/m11_mamba_landcover_v2'

model_config = dict(
    seghead=dict(
        scan_cfg=dict(
            selector='landcover_v2',
            selector_weights=(0.35, 0.25, 0.25, 0.15),
            selector_quotas=(0.35, 0.25, 0.25, 0.15),
            class_quota=0.15,
            confusion_pairs=((5, 6), (4, 6), (2, 0), (2, 4), (1, 0), (3, 0)),
            num_prototypes=2,
            confidence_gate=True,
        ),
    ),
)
