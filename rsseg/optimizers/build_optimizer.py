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
    # catalyst is only needed by the 'multi' lr mode and the SGD+Lookahead branch, so import it lazily
    if cfg.lr_mode == 'multi':
        from catalyst import utils
        layerwise_params = {"backbone.*": dict(lr=cfg.backbone_lr, weight_decay=cfg.backbone_weight_decay)}
        net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
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


    

    
    