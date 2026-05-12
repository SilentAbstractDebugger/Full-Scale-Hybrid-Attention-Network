"""
loss/loss.py
------------
DiceLoss        : 1 - soft Dice coefficient
FocalLoss       : focal cross-entropy for class imbalance
SegmentationLoss: Dice + 0.5 * Focal  (combined)
DeepSupervisionLoss: 0.5*L_main + 0.3*L_aux1 + 0.2*L_aux2
All functions accept raw logits (sigmoid applied internally).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss. smooth avoids division by zero."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(logits.size(0),   -1)
        targets = targets.view(targets.size(0), -1)
        inter   = (probs * targets).sum(1)
        denom   = probs.sum(1) + targets.sum(1)
        dice    = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    Downweights easy negatives, focuses on hard vessel pixels.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(logits, targets,
                                                     reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class SegmentationLoss(nn.Module):
    """
    Combined loss:  L = Dice + focal_weight * Focal
    Default focal_weight = 0.5
    """
    def __init__(self, dice_smooth=1.0, focal_alpha=0.25,
                 focal_gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.dice         = DiceLoss(dice_smooth)
        self.focal        = FocalLoss(focal_alpha, focal_gamma)
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        return (self.dice(logits, targets)
                + self.focal_weight * self.focal(logits, targets))


class DeepSupervisionLoss(nn.Module):
    """
    Weighted sum over three output heads:
        L_total = 0.5 * L_main + 0.3 * L_aux1 + 0.2 * L_aux2
    """
    def __init__(self, seg_loss=None, weights=(0.5, 0.3, 0.2)):
        super().__init__()
        self.seg_loss = seg_loss or SegmentationLoss()
        self.w_main, self.w_aux1, self.w_aux2 = weights

    def forward(self, outputs, targets):
        main, aux1, aux2 = outputs
        return (self.w_main * self.seg_loss(main,  targets)
              + self.w_aux1 * self.seg_loss(aux1,  targets)
              + self.w_aux2 * self.seg_loss(aux2,  targets))
