"""
train.py
--------
Training loop for U-Net and FSHA-Net on the DRIVE dataset.

Improvements over naive baseline
---------------------------------
  1. LR scheduler  : ReduceLROnPlateau on val Dice (halves LR after 7 stale epochs)
  2. Grad clipping : prevents exploding gradients on small datasets
  3. Early stopping: stops after --patience epochs with no Dice improvement
  4. HD95 tracked  : Hausdorff Distance logged at every validation step
  5. Weight decay  : 1e-5 L2 regularisation via Adam

Usage
-----
    python train.py --model unet    --epochs 50 --batch_size 4
    python train.py --model fshanet --epochs 50 --batch_size 4
"""

import argparse
from pathlib import Path
import torch, torch.nn as nn
from data_loader import get_dataloaders
from models.unet import UNet
from models.fsha_net import FSHANet
from loss.loss import SegmentationLoss, DeepSupervisionLoss
from metrics.metrics import dice_score, iou_score, hausdorff_distance_95
from utils import get_device, save_checkpoint, save_visualisations, AverageMeter


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net or FSHA-Net on DRIVE")
    p.add_argument("--model",       default="fshanet", choices=["unet","fshanet"])
    p.add_argument("--data_root",   default="D:/pillai/DRIVE")
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--save_dir",    default="outputs")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--patience",    type=int,   default=15,
                   help="Early stopping: epochs with no val-Dice improvement")
    p.add_argument("--clip_grad",   type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")
    return p.parse_args()


def build_model(name):
    if name == "unet":    return UNet(in_channels=3, out_channels=1, base_filters=64)
    if name == "fshanet": return FSHANet(in_ch=3, out_ch=1, base_f=64)
    raise ValueError(name)


def train_one_epoch(model, loader, optimizer, criterion,
                    device, model_name, clip_grad):
    model.train()
    m = AverageMeter("loss")
    for i, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out  = model(imgs)
        loss = criterion(out, masks)   # works for both UNet and FSHANet

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        m.update(loss.item(), imgs.size(0))

        if (i+1) % max(1, len(loader)//4) == 0:
            print(f"    [{i+1}/{len(loader)}]  loss={loss.item():.4f}"
                  f"  avg={m.avg:.4f}")
    return m.avg


def validate(model, loader, criterion, device, model_name):
    """Returns (val_loss, dice, iou, hd95)."""
    model.eval()
    lm = AverageMeter("loss")
    dm = AverageMeter("dice")
    im = AverageMeter("iou")
    hm = AverageMeter("hd95")

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)   # eval mode always returns single logit

            loss = criterion.seg_loss(logits, targets) if hasattr(criterion, "seg_loss") else criterion(logits, targets)
            lm.update(loss.item(), imgs.size(0))

            preds = (torch.sigmoid(logits) > 0.5).float()
            for b in range(imgs.size(0)):
                p = preds[b,0].cpu().numpy().astype(int)
                t = masks[b,0].cpu().numpy().astype(int)
                dm.update(dice_score(p, t))
                im.update(iou_score(p, t))
                hm.update(hausdorff_distance_95(p, t))

    return lm.avg, dm.avg, im.avg, hm.avg


def train(args):
    print("="*65)
    print(f"  Model : {args.model.upper()}   Data: {args.data_root}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}"
          f"  LR: {args.lr}  Patience: {args.patience}")
    print("="*65)

    device = get_device()
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers)

    model = build_model(args.model).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    seg_loss  = SegmentationLoss()
    criterion = (DeepSupervisionLoss(seg_loss=seg_loss)
                 if args.model == "fshanet" else seg_loss)

    # Adam with weight decay for mild regularisation
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Halve LR after 7 epochs of no val-Dice improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7)

    best_dice  = 0.0
    no_improve = 0
    ckpt_path  = Path(args.ckpt_dir) / f"{args.model}_best.pth"
    vis_dir    = Path(args.save_dir) / args.model

    for epoch in range(1, args.epochs+1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}]  LR={lr_now:.2e}")

        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  criterion, device, args.model, args.clip_grad)
        vl_loss, vl_dice, vl_iou, vl_hd95 = validate(
            model, val_loader, criterion, device, args.model)

        print(f"  Train={tr_loss:.4f} | Val={vl_loss:.4f} | "
              f"Dice={vl_dice:.4f} | IoU={vl_iou:.4f} | HD95={vl_hd95:.2f}px")

        scheduler.step(vl_dice)

        if vl_dice > best_dice:
            best_dice  = vl_dice
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, vl_dice, str(ckpt_path))
            # Save visualisation at best epoch
            model.eval()
            with torch.no_grad():
                si, sm = next(iter(val_loader))
                si, sm = si.to(device), sm.to(device)
                sl = model(si)
                save_visualisations(si[0], sm[0], sl[0],
                                    str(vis_dir), f"epoch{epoch:03d}_")
            model.train()
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\n⚠  Early stopping at epoch {epoch}.")
                break

    print(f"\n✅ Done.  Best Dice={best_dice:.4f}  →  {ckpt_path}")


if __name__ == "__main__":
    train(parse_args())
