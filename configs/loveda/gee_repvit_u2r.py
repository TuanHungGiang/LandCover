# Cross-domain: train on Urban only, checkpoint selection on the Urban val split (source domain).
# Evaluate on Rural afterwards with tools/diagnose_preclf.py (never select checkpoints on the target domain).
_base_ = './gee_repvit.py'

exp_name = "work_dirs/gee_repvit_u2r_loveda"
dataset_config = dict(
    train_mode=dict(domains=['Urban']),
    val_mode=dict(domains=['Urban']),
)
