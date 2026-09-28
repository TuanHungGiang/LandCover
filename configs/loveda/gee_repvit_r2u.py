# Cross-domain: train on Rural only, checkpoint selection on the Rural val split (source domain).
# Evaluate on Urban afterwards with tools/diagnose_preclf.py (never select checkpoints on the target domain).
_base_ = './gee_repvit.py'

exp_name = "work_dirs/gee_repvit_r2u_loveda"
dataset_config = dict(
    train_mode=dict(domains=['Rural']),
    val_mode=dict(domains=['Rural']),
)
