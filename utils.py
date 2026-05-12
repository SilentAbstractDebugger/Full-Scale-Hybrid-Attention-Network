"""
utils.py
--------
Shared utilities:
  get_device          : auto-detect CUDA / MPS / CPU
  save_checkpoint     : save model + optimiser state
  load_checkpoint     : restore model + optimiser state
  save_visualisations : save original / gt / pred PNGs
  AverageMeter        : running mean tracker
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch, torch.nn as nn
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d


def save_checkpoint(model, optimizer, epoch, val_dice, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "val_dice": val_dice,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, filepath)
    print(f"  ✓ Checkpoint saved → {filepath}  (Dice={val_dice:.4f})")


def load_checkpoint(model, filepath, optimizer=None,
                    device=torch.device("cpu")):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  ✓ Loaded {filepath}  (epoch {ckpt.get('epoch','?')},"
          f" Dice={ckpt.get('val_dice',0):.4f})")
    return ckpt


def denormalize(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) uint8."""
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * _STD + _MEAN, 0, 1)
    return (img * 255).astype(np.uint8)


def save_visualisations(img_t, gt_t, logit_t, save_dir,
                        prefix="", threshold=0.5):
    """
    Save three PNGs to save_dir:
      {prefix}original.png   – denormalised RGB image
      {prefix}gt.png         – ground-truth binary mask  (0 or 255)
      {prefix}pred.png       – predicted binary mask     (0 or 255)
    Prediction pipeline: sigmoid -> threshold -> {0,255}
    """
    d = Path(save_dir); d.mkdir(parents=True, exist_ok=True)

    # Original
    Image.fromarray(denormalize(img_t)).save(d / f"{prefix}original.png")

    # Ground truth
    gt = (gt_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(gt, mode="L").save(d / f"{prefix}gt.png")

    # Prediction
    pred = (torch.sigmoid(logit_t.squeeze()) > threshold
            ).cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(pred, mode="L").save(d / f"{prefix}pred.png")


class AverageMeter:
    """Tracks running mean — used for epoch-level statistics."""
    def __init__(self, name="m"):
        self.name = name; self.reset()
    def reset(self):
        self.sum = self.count = self.avg = 0.0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n
        self.avg  = self.sum / self.count if self.count else 0.0
    def __repr__(self): return f"{self.name}={self.avg:.4f}"
