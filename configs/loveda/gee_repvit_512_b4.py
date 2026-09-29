# Same as gee_repvit_512.py but batch_size=4. The first 512 run stayed at mem=2.03GB steady
# (out of ~15.6GB on a T4) at batch_size=1 -- huge unused headroom, since batch_size=1 was chosen
# conservatively to avoid repeating the crop=1024 OOM, which doesn't apply at 512 (1/4 the pixels).
# Fewer, bigger steps (2522/4 =~ 630 vs 2522) means less fixed per-step Python/kernel-launch
# overhead, which should meaningfully reduce wall-clock time per epoch. Not verified yet --
# watch the first epoch's mem= readings; if it approaches ~12-13GB, stop here rather than going higher.
_base_ = './gee_repvit_512.py'

exp_name = "work_dirs/gee_repvit_512_b4"
dataset_config = dict(
    train_mode=dict(loader=dict(batch_size=4)),
    val_mode=dict(loader=dict(batch_size=4)),
)
