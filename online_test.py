import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import time
import os
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser('description=online test')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tta", type=str, default="d4", help="none | lr (flips, 4 passes) | ms (flip + scales 0.75/1/1.25, 6 passes) | d4 (flips + rot90 + 5 scales, 40 passes)")
    parser.add_argument("--set", nargs="+", default=None, help="the same --set overrides the run was trained with")
    parser.add_argument("--batch", type=int, default=2, help="test batch size")
    parser.add_argument("--masks_output_dir", default=None)
    parser.add_argument("--label_offset", type=int, default=1,
                        help="added to the predicted class ids before saving: the official LoveDA labels are 1-7 (0 = no-data) while the "
                             "model predicts 0-6. If the first server score looks wrong (a few %%), try --label_offset 0")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)
    if args.set:
        cfg.merge_from_dict(parse_cfg_overrides(args.set))
    cfg.model_config.backbone.init_cfg = None            # weights come from the checkpoint
    cfg.dataset_config.test_mode.loader.batch_size = args.batch

    if args.masks_output_dir is not None:
        masks_output_dir = args.masks_output_dir
    else:
        masks_output_dir = cfg.exp_name + '/online_figs'

    model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg, strict = False)
    model = model.to('cuda')

    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "ms":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    results = []
    mask2RGB = False
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        print(len(test_loader))
        for input in tqdm(test_loader):
            raw_predictions, img_id = model(input[0].cuda(), True), input[2]
            pred = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask_pred = (pred[i].cpu().numpy() + args.label_offset).astype('uint8')
                mask_name = str(img_id[i])
                results.append((mask2RGB, mask_pred, cfg.dataset, masks_output_dir, mask_name))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    print("masks_save_dir: ", masks_output_dir)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(mask_save, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))