"""Fast A1 ablation: quota-based 25% selector on the unchanged V1 Mamba path.

Only token selection changes versus ``gee_repvit_landcover.py``.  Loss, backbone, one-prototype
class sequences, residual path and training recipe remain identical, so the result isolates whether
LoveDA-specific quotas improve the V1 selector without the full-sort overhead of V2.
"""
_base_ = './gee_repvit_landcover.py'

exp_name = 'work_dirs/m11_mamba_landcover_fast'

model_config = dict(
    seghead=dict(
        scan_cfg=dict(
            selector='landcover_fast',
            # uncertainty / boundary / known confusion / present-class coverage
            selector_weights=(0.40, 0.25, 0.15, 0.20),
            selector_quotas=(0.40, 0.25, 0.15, 0.20),
            num_prototypes=1,
            confidence_gate=False,
        ),
    ),
)
