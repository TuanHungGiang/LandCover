# Same as gee_repvit.py but crop=512x512 (LOGCAN++/SCSM/AFENet's standard crop) instead of native
# 1024x1024. Measured ~0.63 s/batch at 1024 (batch_size=1, T4) -- too slow for fast iteration
# (~9h for 15 epochs). 512 has 1/4 the pixels, expected ~2.5-4x faster (not a full 4x: fixed
# per-step overhead like the optimizer step and Python/kernel-launch cost doesn't shrink with crop).
# batch_size stays 1 for this first run to avoid re-triggering the earlier OOM (crop=1024, batch=2) --
# raise it later once 512 is confirmed stable, since peak memory should drop roughly with pixel count.
_base_ = './gee_repvit.py'

exp_name = "work_dirs/gee_repvit_512_1epoch"
dataset_config = dict(
    train_mode=dict(transform=dict(RandomSizeAndCrop=dict(size=512))),
)
