
######################## base_config #########################
gpus = [0]
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
pretrained_ckpt_path = None
monitor = 'val_miou'

test_ckpt_path = None

######################## dataset_config ######################
exp_name = "work_dirs/gee_r34_loveda"
_base_ = '../_base_/loveda_config.py'
epoch = 50
num_class = 7
ignore_index = 7

# Native LoveDA resolution (1024x1024). ignore_index=7 makes the padding added by RandomSizeAndCrop
# (when the random scale shrinks the image below the crop size) count as ignored instead of class 0 (building).
dataset_config = dict(
    train_mode=dict(
        transform=dict(
            RandomSizeAndCrop={"size": 1024, "crop_nopad": False, "ignore_index": ignore_index},
        ),
        loader=dict(batch_size=2),
    ),
    val_mode=dict(loader=dict(batch_size=2)),
    test_mode=dict(loader=dict(batch_size=2)),
)

######################### model_config #########################
model_config = dict(
    num_class = num_class,
    # torchvision ImageNet weights, downloaded automatically (needs internet)
    backbone = dict(
        type = 'get_resnet34_OS32',
        pretrained = True
    ),
    seghead = dict(
        type = 'GEE_Head',
        in_channel = [64, 128, 256, 512],
        transform_channel = 96,
        num_class = num_class,
        num_heads = 8,
        patch_size = (4, 4),
        explore_grid = 8,
        mode = 'gated',     # 'gated' | 'sum' | 'exploit_only' | 'explore_only'
    ),
    classifier = dict(
        type = 'Base_Classifier',
        transform_channel = 96,
        num_class = num_class,
    ),
    upsample=dict(
        type='Interpolate',
        mode='bilinear',
        scale=[4, 32, 16, 8, 4],
    )
)
# main CE + deep supervision on the classifier of each stage (coarse -> fine)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss'] * 5,
    loss_weight = [1, 0.8, 0.4, 0.4, 0.4],
    ignore_index = ignore_index
)

######################## optimizer_config ######################
# Same optimizer settings as configs/loveda/logcanplus.py (untuned for this model).
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        lr = 1e-4,
        weight_decay = 1e-4,
        momentum = 0.9,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'Poly',
        poly_exp = 0.9,
        max_epoch = epoch
    )
)
