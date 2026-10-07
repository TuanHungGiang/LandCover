import torch
import torch.optim as optim
import math

class lambdax:
    def __init__(self, cfg):
        self.cfg = cfg
    @staticmethod
    def lambda_epoch(self, epoch):
        return math.pow(1 - epoch / self.cfg.max_epoch, self.cfg.poly_exp)


def get_optimizer(cfg, net):
    # Fine-tuning uses a lower LR for the pretrained backbone and the base LR for the new decoder.
    # Build the groups locally so Kaggle does not need catalyst just for differential learning rates.
    if cfg.lr_mode == 'multi':
        backbone_params, other_params = [], []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith('backbone.') else other_params).append(param)
        net_params = [
            dict(params=backbone_params, lr=cfg.backbone_lr,
                 weight_decay=getattr(cfg, 'backbone_weight_decay', cfg.weight_decay)),
            dict(params=other_params, lr=cfg.lr, weight_decay=cfg.weight_decay),
        ]
    else:
        net_params = net.parameters()

    if cfg.type == "AdamW":
        optimizer = optim.AdamW(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # optimizer = Lookahead(optimizer)

    elif cfg.type == "SGD":
        from catalyst.contrib.nn import Lookahead
        optimizer = optim.SGD(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum,
                              nesterov=False)
        optimizer = Lookahead(optimizer)
    else:
        raise KeyError("The optimizer type ( %s ) doesn't exist!!!" % cfg.type)

    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.type == 'Poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / cfg.max_epoch, cfg.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif cfg.type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=1e-6)
    elif cfg.type == 'WarmupCosine':
        warmup = max(0, int(getattr(cfg, 'warmup_epochs', 0)))
        eta_min = float(getattr(cfg, 'eta_min', 1e-6))

        def make_lambda(base_lr):
            eta_ratio = min(1.0, eta_min / base_lr)

            def lr_lambda(epoch):
                if warmup and epoch < warmup:
                    return float(epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, cfg.max_epoch - warmup)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
                return eta_ratio + (1.0 - eta_ratio) * cosine
            return lr_lambda

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[make_lambda(group['lr']) for group in optimizer.param_groups])
    else:
        raise KeyError("The scheduler type ( %s ) doesn't exist!!!" % cfg.type)

    return scheduler

def build_optimizer(cfg, net):
    optimizer = get_optimizer(cfg.optimizer, net)
    scheduler = get_scheduler(cfg.scheduler, optimizer)
    # if cfg.type == 'Poly':
    #     lambda1 = lambda epoch: math.pow(1 - epoch / cfg.max_epoch, cfg.poly_exp)
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # elif cfg.type == 'CosineAnnealingLR':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=1e-6)
    # else:
    #     raise KeyError("The scheduler type ( %s ) doesn't exist!!!" % cfg.type)

    return optimizer, scheduler
