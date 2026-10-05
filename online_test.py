import torch
from tqdm import tqdm
import time
import os
import cv2
import numpy as np

from train import *

import argparse
from utils.config import Config
from tools.inference import build_predictor

def get_args():
    parser = argparse.ArgumentParser('description=online test')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tta", type=str, default="d4", help="none | lr (flips, 4 passes) | ms (flip + scales 0.75/1/1.25, 6 passes) | d4 (flips + rot90 + 5 scales, 40 passes)")
    parser.add_argument("--setting", default=None,
                        help="preferred unified setting, e.g. full-ms, tile512s256 or tile512s256-ms; overrides --tta")
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
    setting = args.setting or ('full' if args.tta == 'none' else f'full-{args.tta}')
    try:
        predictor = build_predictor(model, setting)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    os.makedirs(masks_output_dir, exist_ok=True)
    saved_names = set()
    label_hist = np.zeros(8, dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        print(f'setting={setting} batches={len(test_loader)} label_offset={args.label_offset}')
        for input in tqdm(test_loader):
            raw_predictions, img_id = predictor(input[0].cuda()), input[2]
            pred = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask_pred = (pred[i].cpu().numpy() + args.label_offset).astype('uint8')
                mask_name = str(img_id[i])
                if mask_name in saved_names:
                    raise RuntimeError(f'duplicate LoveDA test id: {mask_name}')
                saved_names.add(mask_name)
                label_hist += np.bincount(mask_pred.ravel(), minlength=8)[:8]
                path = os.path.join(masks_output_dir, mask_name + '.png')
                if not cv2.imwrite(path, mask_pred):
                    raise OSError(f'failed to write {path}')

    print(f'masks_save_dir: {masks_output_dir}')
    print(f'written={len(saved_names)} seconds={time.time() - t0:.1f} label_hist_0_to_7={label_hist.tolist()}')
