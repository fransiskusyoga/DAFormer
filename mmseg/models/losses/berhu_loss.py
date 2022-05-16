

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class BerHuLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BerHuLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, label, is_vector=None):
        if not is_vector:
            n, c, h, w = pred.size()
            assert c == 1
            pred = pred.squeeze()
            label = label.squeeze()
        # label = label.squeeze().to(self.device)
        adiff = torch.abs(pred - label)
        batch_max = 0.2 * torch.max(adiff).item()
        t1_mask = adiff.le(batch_max).float()
        t2_mask = adiff.gt(batch_max).float()
        t1 = adiff * t1_mask
        t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
        t2 = t2 * t2_mask
        return self.loss_weight*(torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)