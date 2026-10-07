import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Macro soft Dice over classes present in the target, with ignored pixels masked out."""

    def __init__(self, ignore_index=255, smooth=1.0, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = float(smooth)

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        valid = target != self.ignore_index
        if not torch.any(valid):
            return pred.sum() * 0.0

        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(pred.dtype)
        mask = valid.unsqueeze(1).to(pred.dtype)
        probs = F.softmax(pred, dim=1) * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        present = one_hot.sum(dims) > 0
        return 1.0 - dice[present].mean()


class CEDiceLoss(nn.Module):
    """Cross entropy with a macro Dice term on the main segmentation output."""

    def __init__(self, ignore_index=255, reduction='mean', class_weight=None,
                 dice_weight=0.5, label_smoothing=0.0, smooth=1.0, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            'class_weight',
            None if class_weight is None else torch.tensor(list(class_weight), dtype=torch.float32),
        )
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)

    def forward(self, pred, target):
        ce = F.cross_entropy(
            pred, target, weight=self.class_weight, ignore_index=self.ignore_index,
            reduction=self.reduction, label_smoothing=self.label_smoothing,
        )
        return ce + self.dice_weight * self.dice(pred, target)


# Backward-compatible name used by older configs.
Dice_Loss = DiceLoss
