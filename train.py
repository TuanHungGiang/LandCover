import numpy as np
# pytorch-lightning 2.0.x still uses aliases that NumPy 2 removed (e.g. np.Inf in ModelCheckpoint); restore them
for _old, _new in (("Inf", "inf"), ("NaN", "nan")):
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import timedelta
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist
import prettytable
import numpy as np

import argparse
import ast
import json
import os
import time
from rsseg.models.build_model import build_model
from rsseg.datasets import build_dataloader
from rsseg.optimizers import build_optimizer
from rsseg.losses import build_loss
from utils.config import Config

seed_everything(2025, workers=True)

def parse_cfg_overrides(pairs):
    """--set key.path=value, e.g. --set dataset_config.train_mode.loader.batch_size=4
    Dotted key -> Config.merge_from_dict (already in utils/config.py, mmcv-style deep merge).
    Value is parsed with ast.literal_eval (4 -> int, 0.5 -> float, "'x'" -> str, [1,2] -> list, ...);
    left as a plain string if that fails, so batch_size=4 and mode=gated both just work.
    """
    options = {}
    for pair in pairs or []:
        key, sep, value = pair.partition('=')
        if not sep:
            raise ValueError(f"--set expects key=value, got: {pair!r}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        options[key] = value
    return options

def get_args():
    parser = argparse.ArgumentParser('rsseg: train model')
    parser.add_argument("-c", "--config", type=str, default="configs/docnet.py")
    parser.add_argument("--set", nargs="+", default=None, metavar="key.path=value",
                        help="Override config values without editing a file, e.g. "
                             "--set dataset_config.train_mode.loader.batch_size=4 model_config.seghead.mode=exploit_only")
    return parser.parse_args()

class myTrain(LightningModule):
    def __init__(self, cfg):
        super(myTrain, self).__init__()
        
        self.cfg = cfg
        self.net = build_model(cfg.model_config)
        self.params_M = sum(p.numel() for p in self.net.parameters()) / 1e6
        self.loss = build_loss(cfg.loss_config)

        self.loss.to("cuda")
        self.eval_label_id_left = cfg.eval_label_id_left
        self.eval_label_id_right = cfg.eval_label_id_right
        
        # Metrics come from a K x K confusion matrix kept per rank and summed over ranks with ONE explicit
        # all_reduce at each epoch end. (torchmetrics' sync issued several collectives per metric; the two
        # ranks got out of step at the first epoch end and NCCL hung until its watchdog killed the run.)
        self._K = cfg.metric_cfg2['num_classes']
        self._ignore = cfg.metric_cfg2['ignore_index']
        self._cm = {'tr': None, 'val': None}

    def _cm_update(self, key, pred, mask):
        K = self._K
        if self._cm[key] is None or self._cm[key].device != pred.device:
            self._cm[key] = torch.zeros(K, K, dtype=torch.long, device=pred.device)
        valid = (mask != self._ignore) & (mask >= 0) & (mask < K)
        idx = mask[valid] * K + pred[valid]
        self._cm[key] += torch.bincount(idx, minlength=K * K).view(K, K)

    def _cm_metrics(self, key):
        """-> ([precision, recall, f1, iou] per class (length K), overall accuracy). Resets the matrix."""
        cm = self._cm[key]
        cm = torch.zeros(self._K, self._K, dtype=torch.long, device=self.device) if cm is None else cm.clone()
        self._cm[key] = None
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cm)                      # every rank calls this exactly once per epoch end
        cm = cm.double()
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp

        def ratio(num, den):
            return torch.where(den > 0, num / den.clamp_min(1), torch.zeros_like(num))

        metrics = [ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn), ratio(tp, tp + fp + fn)]
        return metrics, float(tp.sum() / cm.sum().clamp_min(1))

    def forward(self, x, test = False) :
        pred = self.net(x)
        if test:
            return pred[0]
        return pred

    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(self.cfg.optimizer_config, self.net)
        return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def train_dataloader(self):
        loader = build_dataloader(self.cfg.dataset_config, mode='train')
        return loader

    def val_dataloader(self):
        loader = build_dataloader(self.cfg.dataset_config, mode='val')
        return loader

    def output(self, metrics, total_metrics, mode):
        result_table = prettytable.PrettyTable()
        result_table.field_names = ['Class', 'OA', 'Precision', 'Recall', 'F1_Score', 'IOU']

        # torchmetrics (average='none') returns one value per configured num_classes (8: 7 classes + the
        # ignore_index slot), one more than class_name (7 names), so bound the loop by class_name instead.
        for i in range(len(self.cfg.class_name)):
            item = [self.cfg.class_name[i], '--']
            for j in range(len(metrics)):
                item.append(np.round(metrics[j][i].cpu().numpy(), 4))
            result_table.add_row(item)

        total = list(total_metrics.values())
        total = [np.round(v, 4) for v in total]
        total.insert(0, 'total')
        result_table.add_row(total)

        # metrics are already synced across ranks by compute(); print/write once
        if not self.trainer.is_global_zero:
            return
        print(result_table)

        file_name = self.cfg.exp_name + "/train_metric.txt"
        f = open(file_name,"a")
        f.write('epoch:{}/{} {}\n'.format(self.current_epoch, self.cfg.epoch, mode))
        f.write(str(result_table)+'\n')
        f.close()

    def _log_gate_stats(self):
        # GEE_Head (mode='gated'/'sum') exposes _gate_means: 1 = fully exploit (class-prior attention),
        # 0 = fully explore (scene attention). A gate stuck near 0 or 1 at every stage means it is not
        # actually routing anything -- worth watching for, since the loss curve alone would not show it.
        # prog_bar=True so it is visible live on the tqdm bar, not only in TensorBoard after the epoch.
        for i, g in enumerate(getattr(self.net.seghead, '_gate_means', [])):
            if g is not None:
                self.log(f'gate_stage{i}', g, on_step=False, on_epoch=True, prog_bar=True)

    def _print_progress(self, batch_idx, all_loss):
        # A plain print() every N steps, independent of the tqdm bar: readable with `tail` on a log
        # file from a background (nohup) run, and not lost once a long epoch's bar is overwritten.
        every = getattr(self.cfg, 'print_every_n_steps', 100)
        if every <= 0 or batch_idx % every != 0:
            return
        total = getattr(self.trainer, 'num_training_batches', '?')
        losses = ' '.join(f'{k}={float(v):.3f}' for k, v in all_loss.items() if k != 'total_loss')
        gates = getattr(self.net.seghead, '_gate_means', [])
        gate_str = 'gate=[' + ','.join(f'{float(g):.3f}' for g in gates if g is not None) + ']' if any(g is not None for g in gates) else ''
        mem = f'mem={torch.cuda.memory_allocated()/1e9:.2f}GB' if torch.cuda.is_available() else ''
        print(f"[ep {self.current_epoch}][batch {batch_idx}/{total}] "
              f"total_loss={float(all_loss['total_loss']):.3f} ({losses}) {gate_str} {mem}", flush=True)

    def training_step(self, batch, batch_idx):
        image, mask = batch[0], batch[1]
        preds = self(image)
        all_loss = self.loss(preds, mask)
        self._log_gate_stats()
        self._print_progress(batch_idx, all_loss)

        pred = preds[0].argmax(dim=1)

        self._cm_update('tr', pred, mask)

        for loss_name in all_loss:
            self.log(f'train_{loss_name}', all_loss[loss_name], on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=image.shape[0])
        return all_loss['total_loss']

    def on_train_epoch_end(self):
        metrics, tr_oa = self._cm_metrics('tr')

        log = {'tr_oa': tr_oa,
               'tr_prec': np.mean([item.cpu() for item in metrics[0][self.eval_label_id_left: self.eval_label_id_right]]),
               'tr_recall': np.mean([item.cpu() for item in metrics[1][self.eval_label_id_left: self.eval_label_id_right]]),
               'tr_f1': np.mean([item.cpu() for item in metrics[2][self.eval_label_id_left: self.eval_label_id_right]]),
               'tr_miou': np.mean([item.cpu() for item in metrics[3][self.eval_label_id_left: self.eval_label_id_right]])}
        
        # self.output(metrics, log, 'train')
        
        for key, value in zip(log.keys(), log.values()):
            self.log(key, value, on_step=False,on_epoch=True,prog_bar=False)

        # Runs after this epoch's validation (Lightning validates before firing on_train_epoch_end),
        # so this covers train+val combined. Printed directly (visible in the Kaggle cell output even
        # without TensorBoard) since an OOM crash gives no log at all otherwise.
        # - allocated: bytes actually held by live tensors.
        # - reserved: bytes the caching allocator has carved out of the device (>= allocated); an OOM
        #   happens when the allocator can't find/grow a reserved block, so this -- not allocated -- is
        #   the number that predicts an OOM. A reserved/allocated gap that grows epoch over epoch is
        #   fragmentation, which PYTORCH_ALLOC_CONF=expandable_segments:True is meant to reduce.
        # - free/total: the whole device's headroom (torch.cuda.mem_get_info), the actual OOM boundary.
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.max_memory_allocated() / 1e9
            reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info())
            print(f"[epoch {self.current_epoch}] GPU memory: allocated={alloc_gb:.2f}GB "
                  f"reserved={reserved_gb:.2f}GB | device free={free_gb:.2f}GB / total={total_gb:.2f}GB")
            self.log('gpu_mem_alloc_gb', alloc_gb, on_step=False, on_epoch=True, prog_bar=False)
            self.log('gpu_mem_reserved_gb', reserved_gb, on_step=False, on_epoch=True, prog_bar=False)
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _tile_starts(size, crop, stride):
        # Window origins along one axis. The last window is shifted back so it ends exactly at `size`
        # (it then overlaps its neighbour more than `stride` implies) instead of running past the image.
        if size <= crop:
            return [0]
        starts = list(range(0, size - crop + 1, stride))
        if starts[-1] != size - crop:
            starts.append(size - crop)
        return starts

    def _val_forward(self, image, mask):
        """Returns (main logits at full image size, loss dict).

        cfg.val_sliding = dict(crop, stride): the image is cut into crop x crop tiles every `stride`
        pixels, each tile goes through the network on its own (same input size as training), and the
        tile logits are summed into a full-size buffer that is divided by how many tiles covered each
        pixel. stride == crop -> no overlap (1024 = 2 x 512 tiles per axis); stride < crop -> overlapping
        pixels get the average of the tile predictions. The loss is the mean of the per-tile losses.
        """
        sw = getattr(self.cfg, 'val_sliding', None)
        if not sw:
            preds = self(image)
            return preds[0], self.loss(preds, mask)

        crop, stride = sw['crop'], sw['stride']
        B, _, H, W = image.shape
        logits, count, loss_sum, n_tiles = None, image.new_zeros(1, 1, H, W), None, 0
        for y in self._tile_starts(H, crop, stride):
            for x in self._tile_starts(W, crop, stride):
                preds = self(image[:, :, y:y + crop, x:x + crop])
                tile_loss = self.loss(preds, mask[:, y:y + crop, x:x + crop])
                if logits is None:
                    logits = preds[0].new_zeros(B, preds[0].shape[1], H, W)
                    loss_sum = {k: 0. for k in tile_loss}
                logits[:, :, y:y + crop, x:x + crop] += preds[0]
                count[:, :, y:y + crop, x:x + crop] += 1
                for k, v in tile_loss.items():
                    loss_sum[k] = loss_sum[k] + v
                n_tiles += 1
        return logits / count, {k: v / n_tiles for k, v in loss_sum.items()}

    def validation_step(self, batch, batch_idx):
        image, mask = batch[0], batch[1]
        logits, all_loss = self._val_forward(image, mask)

        pred = logits.argmax(dim=1)

        self._cm_update('val', pred, mask)

        for loss_name in all_loss:
            self.log(f'val_{loss_name}', all_loss[loss_name], on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=image.shape[0])
        return all_loss['total_loss']

    def on_validation_epoch_end(self):
        metrics, val_oa = self._cm_metrics('val')

        log = {'val_oa': val_oa,
               'val_prec': np.mean([item.cpu() for item in metrics[0][self.eval_label_id_left: self.eval_label_id_right]]),
               'val_recall': np.mean([item.cpu() for item in metrics[1][self.eval_label_id_left: self.eval_label_id_right]]),
               'val_f1': np.mean([item.cpu() for item in metrics[2][self.eval_label_id_left: self.eval_label_id_right]]),
               'val_miou': np.mean([item.cpu() for item in metrics[3][self.eval_label_id_left: self.eval_label_id_right]])}
        
        self.output(metrics, log, 'val')

        # one JSON line per validation, read by tools/run_ablation.py to build the comparison table
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            n_cls = len(self.cfg.class_name)
            os.makedirs(self.cfg.exp_name, exist_ok=True)
            with open(self.cfg.exp_name + "/val_metrics.jsonl", "a") as f:
                f.write(json.dumps(dict(
                    epoch=self.current_epoch, time=time.time(), params_M=self.params_M,
                    val_miou=float(log['val_miou']), val_oa=float(log['val_oa']),
                    iou=[float(v) for v in metrics[3][:n_cls].cpu()])) + "\n")

        for key, value in zip(log.keys(), log.values()):
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False)

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)
    if args.set:
        cfg.merge_from_dict(parse_cfg_overrides(args.set))
    print(cfg)
    model = myTrain(cfg)

    
    lr_monitor=LearningRateMonitor(logging_interval = cfg.logging_interval)

    ckpt_cb = ModelCheckpoint(dirpath = cfg.exp_name,
                              filename = '{epoch:d}',
                              monitor = cfg.monitor,
                              mode = 'max',
                              save_top_k = cfg.save_top_k,
                              save_last = getattr(cfg, 'save_last', False))
    
    # refresh_rate=1 (one tqdm redraw per batch) floods a Jupyter/Kaggle cell: without a real TTY, tqdm
    # can't overwrite the same line with \r, so every redraw becomes a new appended line -- thousands of
    # them over a ~2500-batch epoch, which bogs down the browser (the kernel keeps training fine, only
    # the page rendering suffers). A higher rate keeps the bar useful without spamming the page/log file;
    # the custom per-N-step print in training_step (print_every_n_steps) is unaffected by this.
    pbar = TQDMProgressBar(refresh_rate=getattr(cfg, 'progress_bar_refresh_rate', 50))

    callbacks = [ckpt_cb, pbar, lr_monitor]

    logger = TensorBoardLogger(save_dir = "",
                               name = cfg.exp_name,
                               default_hp_metric = False)
    
    
    # optional config keys (defaults keep the original behaviour): precision='16-mixed',
    # limit_train_batches / limit_val_batches for quick timing runs, accumulate_grad_batches to
    # reach a larger EFFECTIVE batch size without the peak memory of a bigger real batch (e.g.
    # batch_size=1 + accumulate_grad_batches=4 trains like batch_size=4 but only ever holds one
    # sample's activations at a time -- the OOM at batch_size=2 never needed to happen this way)
    # cfg.gpus with more than one id -> DDP, one process per GPU. loader.batch_size is then PER GPU, so
    # the effective batch is batch_size * len(gpus) (8 x 2 = 16, the SCSM/LOGCAN++ setting).
    # find_unused_parameters: the RepViT layers after the last out_index take no part in the loss.
    multi_gpu = len(cfg.gpus) > 1
    trainer = Trainer(max_epochs = cfg.epoch,
                      strategy = DDPStrategy(find_unused_parameters=True,
                                             # a hung collective aborts after this instead of the 30 min default
                                             timeout=timedelta(minutes=getattr(cfg, 'ddp_timeout_min', 10))) if multi_gpu else 'auto',
                      sync_batchnorm = multi_gpu and getattr(cfg, 'sync_bn', False),   # per-GPU batch 8 is enough for BN
                      check_val_every_n_epoch = getattr(cfg, 'check_val_every_n_epoch', 1),
                      precision = getattr(cfg, 'precision', 32),
                      limit_train_batches = getattr(cfg, 'limit_train_batches', 1.0),
                      limit_val_batches = getattr(cfg, 'limit_val_batches', 1.0),
                      accumulate_grad_batches = getattr(cfg, 'accumulate_grad_batches', 1),
                      callbacks = callbacks,
                      logger = logger,
                      enable_model_summary = True,
                      accelerator = 'auto',
                      devices = cfg.gpus,
                      num_sanity_val_steps = 2,
                      benchmark = True)
    
    trainer.fit(model, ckpt_path=cfg.resume_ckpt_path)