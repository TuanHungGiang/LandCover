import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', class_weight=None, **kwargs):
        """class_weight: optional list with one weight per class (e.g. [0.7, 1, 1, 1, 1, 1, 1] makes the catch-all
        class 0 count less); applied to every CE term of the model (main + deep supervision)."""
        super(CELoss, self).__init__()

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weight = None if class_weight is None else torch.tensor(list(class_weight), dtype=torch.float32)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, pred, target):
        w = None if self.class_weight is None else self.class_weight.to(pred.device)
        return F.cross_entropy(pred, target, weight=w, ignore_index=self.ignore_index, reduction=self.reduction)
