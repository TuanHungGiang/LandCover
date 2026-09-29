# Screening config for tools/run_ablation.py: GEE_Head variants trained for a few epochs each, so several
# of them fit in one Kaggle session. 2 GPUs (DDP), crop 512, batch 8 per GPU (= 16 total), validation as
# 512 tiles. tools/run_ablation.py overrides mode / sparse_ratio / epoch / exp_name through --set.
_base_ = './gee_repvit.py'

gpus = [0, 1]
epoch = 12
exp_name = "work_dirs/gee_ablation"
check_val_every_n_epoch = 3
save_top_k = 1
save_last = False

dataset_config = dict(
    train_mode=dict(
        transform=dict(RandomSizeAndCrop=dict(size=512)),
        loader=dict(batch_size=8, num_workers=4),
    ),
    val_mode=dict(loader=dict(batch_size=4, num_workers=4)),
)

# the scheduler max_epoch of the base file was already evaluated there, so restate it for this epoch
optimizer_config = dict(scheduler=dict(max_epoch=epoch))
