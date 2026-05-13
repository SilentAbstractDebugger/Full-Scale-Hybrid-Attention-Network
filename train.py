import argparse
import csv
from pathlib import Path

import torch
from data_loader import get_dataloaders
from models.unet import UNet
from models.fsha_net import FSHANet
from loss.loss import SegmentationLoss, DeepSupervisionLoss
from metrics.metrics import (dice_score, iou_score,
                              hausdorff_distance_95, sensitivity_specificity)
from utils import get_device, save_checkpoint, save_visualisations, AverageMeter


# =============================================================================
#  CSV LOGGER  -- saves into  curves/  inside the repo
# =============================================================================

class CSVLogger:
    """
    Appends one row per epoch to
        curves/<model_name>_train_log.csv
    inside the project folder (same directory as train.py).

    Columns:
        epoch, train_loss, val_loss, val_dice, val_iou,
        val_hd95, val_sensitivity, val_specificity, lr
    """

    HEADER = [
        "epoch", "train_loss", "val_loss",
        "val_dice", "val_iou", "val_hd95",
        "val_sensitivity", "val_specificity", "lr",
    ]

    def __init__(self, model_name: str):
        # Always lives inside the repo under  curves/
        script_dir = Path(__file__).parent
        log_dir    = script_dir / "curves"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.path = log_dir / f"{model_name}_train_log.csv"
        # Write header only once (fresh file)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
        print(f"  [CSVLogger] -> {self.path}")

    def log(self, epoch, train_loss, val_loss,
            val_dice, val_iou, val_hd95,
            val_sensitivity, val_specificity, lr):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_dice:.6f}",
                f"{val_iou:.6f}",
                f"{val_hd95:.4f}",
                f"{val_sensitivity:.6f}",
                f"{val_specificity:.6f}",
                f"{lr:.2e}",
            ])


# =============================================================================
#  ARGUMENT PARSER
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net or FSHA-Net on DRIVE")
    p.add_argument("--model",       default="fshanet", choices=["unet", "fshanet"])
    p.add_argument("--data_root",   default="D:/pillai/DRIVE")
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--save_dir",    default="output")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--clip_grad",   type=float, default=1.0)
    return p.parse_args()


# =============================================================================
#  MODEL BUILDER
# =============================================================================

def build_model(name):
    if name == "unet":    return UNet(in_channels=3, out_channels=1, base_filters=64)
    if name == "fshanet": return FSHANet(in_ch=3, out_ch=1, base_f=64)
    raise ValueError(name)


# =============================================================================
#  TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad):
    model.train()
    meter = AverageMeter("loss")

    for i, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out  = model(imgs)
        loss = criterion(out, masks)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        meter.update(loss.item(), imgs.size(0))

        if (i + 1) % max(1, len(loader) // 4) == 0:
            print(f"    [{i+1}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  avg={meter.avg:.4f}")

    return meter.avg


# =============================================================================
#  VALIDATE -- returns ALL 6 metrics
# =============================================================================

def validate(model, loader, criterion, device):
    """
    Returns:
        val_loss, val_dice, val_iou, val_hd95,
        val_sensitivity, val_specificity
    """
    model.eval()
    lm  = AverageMeter("loss")
    dm  = AverageMeter("dice")
    im  = AverageMeter("iou")
    hm  = AverageMeter("hd95")
    snm = AverageMeter("sens")
    spm = AverageMeter("spec")

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)          # eval mode -> single logit

            if hasattr(criterion, "seg_loss"):
                loss = criterion.seg_loss(logits, masks)
            else:
                loss = criterion(logits, masks)
            lm.update(loss.item(), imgs.size(0))

            preds = (torch.sigmoid(logits) > 0.5).float()

            for b in range(imgs.size(0)):
                p = preds[b, 0].cpu().numpy().astype(int)
                t = masks[b, 0].cpu().numpy().astype(int)
                dm.update(dice_score(p, t))
                im.update(iou_score(p, t))
                hm.update(hausdorff_distance_95(p, t))
                sn, sp = sensitivity_specificity(p, t)
                snm.update(sn)
                spm.update(sp)

    return (lm.avg, dm.avg, im.avg,
            hm.avg, snm.avg, spm.avg)


# =============================================================================
#  MAIN TRAINING LOOP
# =============================================================================

def train(args):
    print("=" * 65)
    print(f"  Model : {args.model.upper()}   Data: {args.data_root}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}"
          f"  LR: {args.lr}  Patience: {args.patience}")
    print("=" * 65)

    device = get_device()
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers)

    model = build_model(args.model).to(device)
    print(f"  Params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    seg_loss  = SegmentationLoss()
    criterion = (DeepSupervisionLoss(seg_loss=seg_loss)
                 if args.model == "fshanet" else seg_loss)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, verbose=False)

    best_dice  = 0.0
    no_improve = 0
    ckpt_path  = Path(args.ckpt_dir) / f"{args.model}_best.pth"
    vis_dir    = Path(args.save_dir) / args.model

    # Create CSV logger -- writes to curves/<model>_train_log.csv
    logger = CSVLogger(args.model)

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}]  LR={lr_now:.2e}")

        # train
        tr_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, device, args.clip_grad)

        # validate -- all 6 metrics
        vl_loss, vl_dice, vl_iou, \
        vl_hd95, vl_sens, vl_spec = validate(
            model, val_loader, criterion, device)

        print(f"  Train={tr_loss:.4f} | Val={vl_loss:.4f} | "
              f"Dice={vl_dice:.4f} | IoU={vl_iou:.4f} | "
              f"HD95={vl_hd95:.2f}px | "
              f"Sens={vl_sens:.4f} | Spec={vl_spec:.4f}")

        # Log everything to CSV
        logger.log(epoch, tr_loss, vl_loss,
                   vl_dice, vl_iou, vl_hd95,
                   vl_sens, vl_spec, lr_now)

        scheduler.step(vl_dice)

        # checkpoint + visualisation
        if vl_dice > best_dice:
            best_dice  = vl_dice
            no_improve = 0
            save_checkpoint(model, optimizer, epoch,
                            vl_dice, str(ckpt_path))
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
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    print(f"\nDone.  Best Dice={best_dice:.4f}  ->  {ckpt_path}")
    print(f"   Log -> curves/{args.model}_train_log.csv")
    print(f"   Run:  python plot_curves.py")


if __name__ == "__main__":
    train(parse_args())
