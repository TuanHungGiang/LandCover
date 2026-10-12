"""C3 policy-search config with an internal holdout carved only from LoveDA Train.

Official LoveDA Val is intentionally unreachable from both loaders in this config.  The final policy
must be trained with ``gee_repvit_landcover_adaptive_spectral.py``, which restores full Train and the
official Val split.
"""
_base_ = './gee_repvit_landcover_adaptive_spectral.py'

exp_name = 'work_dirs/info_token_search_base'

dataset_config = dict(
    train_mode=dict(
        source_split='Train',
        split_role='fit',
        holdout_seed=2026,
        num_folds=3,
        fold_index=0,
        loader=dict(batch_size=4, num_workers=2),
    ),
    val_mode=dict(
        source_split='Train',
        split_role='holdout',
        holdout_seed=2026,
        num_folds=3,
        fold_index=0,
        loader=dict(batch_size=2, num_workers=2),
    ),
)
