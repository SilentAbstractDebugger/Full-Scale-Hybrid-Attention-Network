import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from data_loader import get_dataloaders
from models.unet import UNet
from models.fsha_net import FSHANet
from metrics.metrics import (dice_score, iou_score,
                              hausdorff_distance_95,
                              sensitivity_specificity)
from utils import get_device, load_checkpoint, save_visualisations, AverageMeter


# =============================================================================
#  ARGUMENT PARSER
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="both",
                   choices=["unet", "fshanet", "both"])
    p.add_argument("--data_root",   default="D:/pillai/DRIVE")
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--unet_ckpt",   default="checkpoints/unet_best.pth")
    p.add_argument("--fsha_ckpt",   default="checkpoints/fshanet_best.pth")
    p.add_argument("--save_dir",    default="output")
    p.add_argument("--save_vis",    action="store_true", default=True)
    return p.parse_args()


# =============================================================================
#  EVALUATE ONE MODEL  -- saves per-sample CSV
# =============================================================================

def evaluate_model(model, model_name, val_loader,
                   device, save_dir, save_vis=True):
    model.eval()

    dm  = AverageMeter("dice")
    im  = AverageMeter("iou")
    hm  = AverageMeter("hd95")
    snm = AverageMeter("sens")
    spm = AverageMeter("spec")

    vis_root = Path(save_dir) / model_name

    # -- CSV setup (saved inside repo under curves/) -------------------------
    script_dir = Path(__file__).parent
    log_dir    = script_dir / "curves"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = log_dir / f"{model_name}_eval_log.csv"

    csv_header = ["sample", "dice", "iou", "hd95", "sensitivity", "specificity"]
    csv_rows   = []
    # -------------------------------------------------------------------------

    print(f"\n{'─'*58}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'─'*58}")

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(val_loader):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            preds  = (torch.sigmoid(logits) > 0.5).float()

            p_np = preds[0, 0].cpu().numpy().astype(int)
            t_np = masks[0, 0].cpu().numpy().astype(int)

            d        = dice_score(p_np, t_np)
            iou      = iou_score(p_np, t_np)
            hd       = hausdorff_distance_95(p_np, t_np)
            sn, sp   = sensitivity_specificity(p_np, t_np)

            dm.update(d);  im.update(iou)
            hm.update(hd); snm.update(sn); spm.update(sp)

            # accumulate for CSV
            csv_rows.append({
                "sample":      idx + 1,
                "dice":        f"{d:.6f}",
                "iou":         f"{iou:.6f}",
                "hd95":        f"{hd:.4f}",
                "sensitivity": f"{sn:.6f}",
                "specificity": f"{sp:.6f}",
            })

            print(f"  [{idx+1:2d}] Dice={d:.4f}  IoU={iou:.4f}  "
                  f"HD95={hd:6.2f}px  Sens={sn:.4f}  Spec={sp:.4f}")

            if save_vis:
                save_visualisations(imgs[0], masks[0], logits[0],
                                    str(vis_root), f"eval_{idx+1:03d}_")

    # Write per-sample CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  [CSVLogger] Eval log -> {csv_path}")

    res = {
        "dice":        dm.avg,
        "iou":         im.avg,
        "hd95":        hm.avg,
        "sensitivity": snm.avg,
        "specificity": spm.avg,
    }

    print(f"\n  -- {model_name.upper()} SUMMARY --")
    for k, v in res.items():
        unit = " px" if k == "hd95" else ""
        print(f"  {k:<14}: {v:.4f}{unit}")

    return res


# =============================================================================
#  COMPARISON TABLE
# =============================================================================

def print_comparison(results):
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON TABLE")
    print("=" * 65)
    hdr = (f"  {'Model':<12} {'Dice':>7} {'IoU':>7} "
           f"{'HD95':>7} {'Sens':>7} {'Spec':>7}")
    print(hdr)
    print(f"  {'-'*58}")
    for name, r in results.items():
        print(f"  {name:<12} {r['dice']:>7.4f} {r['iou']:>7.4f} "
              f"{r['hd95']:>7.2f} {r['sensitivity']:>7.4f} "
              f"{r['specificity']:>7.4f}")
    print("=" * 65)

    if len(results) == 2:
        names = list(results.keys())
        bd = names[0] if results[names[0]]["dice"] >= results[names[1]]["dice"] \
             else names[1]
        bh = names[0] if results[names[0]]["hd95"] <= results[names[1]]["hd95"] \
             else names[1]
        print(f"\n  > {bd.upper()} achieves higher Dice / IoU.")
        print(f"  > {bh.upper()} has tighter boundary predictions (lower HD95).")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    args   = parse_args()
    device = get_device()
    _, val_loader = get_dataloaders(
        args.data_root, args.img_size, 1, args.num_workers)

    results = {}

    if args.model in ("unet", "both"):
        m = UNet(3, 1, 64).to(device)
        load_checkpoint(m, args.unet_ckpt, device=device)
        results["unet"] = evaluate_model(
            m, "unet", val_loader, device,
            args.save_dir, args.save_vis)

    if args.model in ("fshanet", "both"):
        m = FSHANet(3, 1, 64).to(device)
        load_checkpoint(m, args.fsha_ckpt, device=device)
        results["fshanet"] = evaluate_model(
            m, "fshanet", val_loader, device,
            args.save_dir, args.save_vis)

    if len(results) > 1:
        print_comparison(results)

    print("\nDone. Run  python plot_curves.py  to generate all plots.")


if __name__ == "__main__":
    main()
