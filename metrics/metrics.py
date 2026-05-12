"""
metrics/metrics.py
------------------
Evaluation metrics for binary medical image segmentation.

Metrics
-------
dice_score            : 2|X∩Y| / (|X|+|Y|)      higher is better
iou_score             : |X∩Y| / |X∪Y|            higher is better
hausdorff_distance_95 : 95th-percentile surface   lower  is better
sensitivity_specificity: TP/(TP+FN), TN/(TN+FP)
compute_metrics       : batch wrapper returning all metrics

All functions operate on binary numpy arrays of shape (H, W).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, target, smooth=1e-6):
    """Dice / F1 coefficient."""
    p = pred.astype(np.float32).flatten()
    t = target.astype(np.float32).flatten()
    inter = (p * t).sum()
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(pred, target, smooth=1e-6):
    """Intersection over Union (Jaccard index)."""
    p = pred.astype(np.float32).flatten()
    t = target.astype(np.float32).flatten()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + smooth) / (union + smooth))


def _surface_distances(pred_b, target_b):
    """Symmetric surface-to-surface distances via EDT."""
    d_p2t = distance_transform_edt(~target_b)[pred_b]
    d_t2p = distance_transform_edt(~pred_b)[target_b]
    if d_p2t.size == 0 and d_t2p.size == 0: return np.array([0.0])
    if d_p2t.size == 0: return d_t2p
    if d_t2p.size == 0: return d_p2t
    return np.concatenate([d_p2t, d_t2p])


def hausdorff_distance_95(pred, target):
    """
    95th-percentile Hausdorff Distance in pixels.
    Returns 0.0 if both masks are empty;
    returns diagonal length as penalty if one is empty.
    """
    pb = pred.astype(bool); tb = target.astype(bool)
    if not pb.any() and not tb.any(): return 0.0
    if not pb.any() or  not tb.any():
        return float(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2))
    return float(np.percentile(_surface_distances(pb, tb), 95))


def sensitivity_specificity(pred, target):
    """
    Sensitivity = TP / (TP + FN)  — vessel recall
    Specificity = TN / (TN + FP)  — background recall
    """
    pb = pred.astype(bool); tb = target.astype(bool)
    TP = ( pb &  tb).sum(); TN = (~pb & ~tb).sum()
    FP = ( pb & ~tb).sum(); FN = (~pb &  tb).sum()
    sens = float(TP / (TP + FN + 1e-8))
    spec = float(TN / (TN + FP + 1e-8))
    return sens, spec


def compute_metrics(preds, targets):
    """
    Compute all metrics for a batch.
    preds, targets : (N, H, W) binary numpy arrays.
    Returns dict: dice, iou, hd95, sensitivity, specificity
    """
    dices, ious, hd95s, senss, specs = [], [], [], [], []
    for i in range(preds.shape[0]):
        dices.append(dice_score(preds[i], targets[i]))
        ious.append(iou_score(preds[i], targets[i]))
        hd95s.append(hausdorff_distance_95(preds[i], targets[i]))
        s, sp = sensitivity_specificity(preds[i], targets[i])
        senss.append(s); specs.append(sp)
    return {
        "dice":        float(np.mean(dices)),
        "iou":         float(np.mean(ious)),
        "hd95":        float(np.mean(hd95s)),
        "sensitivity": float(np.mean(senss)),
        "specificity": float(np.mean(specs)),
    }
