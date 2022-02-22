import torch
from torch import nn


class MultilabelFocalLoss(nn.Module):
    def __init__(
        self, n_classes, weight=None, gamma=2.0, threshold=0.5, reduction="mean"
    ):
        super(MultilabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.n_classes = n_classes
        self.threshold = threshold

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= self.threshold, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        if self.reduction == "mean":
            loss = self.n_classes * loss.mean()
        elif self.reduction == "sum":
            loss = self.n_classes * loss.sum()
        else:
            loss = self.n_classes * loss.mean()

        return loss
