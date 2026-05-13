import csv
import sys
from pathlib import Path
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


CURVES_DIR = Path(__file__).parent / "curves"

# Colours and display labels
COLORS = {
    "unet":    "#E07B39",   # warm orange
    "fshanet": "#2E86AB",   # steel blue
}
LABELS = {
    "unet":    "U-Net (baseline)",
    "fshanet": "FSHA-Net (ours)",
}

DPI = 200   # resolution of saved PNGs


ABLATION_DATA = {
    #  variant name             Dice    IoU    HD95   Sens   Spec
    "U-Net (baseline)":       [0.7089, 0.5738, 3.57, 0.8618, 0.9742],
    "FS only\n(no attention)": [0.730,  0.590,  3.10, 0.865,  0.978],
    "FS + Attention\n(no DS)": [0.745,  0.600,  2.80, 0.866,  0.981],
    "FSHA-Net\n(full model)":  [0.7553, 0.6071, 2.57, 0.8681, 0.9820],
}
# NOTE: Replace the placeholder rows (rows 2 and 3) with real values
#       once you retrain the ablation variants.


def read_train_log(model_name):
    """
    Returns dict of lists, or None if file not found.
    Keys: epoch, train_loss, val_loss, val_dice, val_iou,
          val_hd95, val_sensitivity, val_specificity, lr
    """
    path = CURVES_DIR / f"{model_name}_train_log.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found.")
        return None
    data = {k: [] for k in [
        "epoch", "train_loss", "val_loss", "val_dice", "val_iou",
        "val_hd95", "val_sensitivity", "val_specificity", "lr"]}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in data:
                data[k].append(float(row[k]))
    print(f"  [loaded train] {path.name}  ({len(data['epoch'])} epochs)")
    return data



def read_eval_log(model_name):
    """
    Returns dict of lists, or None if file not found.
    Keys: sample, dice, iou, hd95, sensitivity, specificity
    """
    path = CURVES_DIR / f"{model_name}_eval_log.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found.")
        return None
    data = {k: [] for k in
            ["sample", "dice", "iou", "hd95", "sensitivity", "specificity"]}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in data:
                data[k].append(float(row[k]))
    print(f"  [loaded eval]  {path.name}  ({len(data['sample'])} samples)")
    return data



def lr_drop_epochs(lr_list):
    drops = []
    for i in range(1, len(lr_list)):
        if lr_list[i] < lr_list[i-1] - 1e-12:
            drops.append(i + 1)
    return drops


def annotate_best(ax, epochs, values, color, higher_better=True):
    if higher_better:
        best_val = max(values)
        best_ep  = epochs[values.index(best_val)]
    else:
        best_val = min(values)
        best_ep  = epochs[values.index(best_val)]
    ax.scatter(best_ep, best_val, color=color,
               s=70, zorder=6, marker="*", linewidths=0)
    ax.annotate(f" {best_val:.4f}",
                xy=(best_ep, best_val),
                fontsize=7, color=color, va="bottom")

 
def plot_line(ax, train_logs, key, ylabel, title,
              higher_better=True, mark_lr=True, ylim=None):
    for name, data in train_logs.items():
        if data is None:
            continue
        epochs = data["epoch"]
        values = data[key]
        ax.plot(epochs, values, color=COLORS[name], linewidth=1.8,
                label=LABELS[name], zorder=3)
        annotate_best(ax, epochs, values, COLORS[name], higher_better)
        if mark_lr:
            for ep in lr_drop_epochs(data["lr"]):
                if ep <= epochs[-1]:
                    ax.axvline(ep, color=COLORS[name], linestyle="--",
                               linewidth=0.8, alpha=0.4, zorder=2)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=8)


def save(fig, filename):
    out = CURVES_DIR / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.name}")



def make_training_figures(train_logs):
    any_data = any(v is not None for v in train_logs.values())
    if not any_data:
        print("  [skip] No training logs found -- skipping training figures.")
        return

    # -- Fig 1: Loss curves (train + val side by side) ------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.suptitle("Loss Curves", fontsize=11, fontweight="bold")
    plot_line(ax1, train_logs, "train_loss",
              "Training Loss", "Training Loss vs. Epoch",
              higher_better=False)
    plot_line(ax2, train_logs, "val_loss",
              "Validation Loss", "Validation Loss vs. Epoch",
              higher_better=False)
    fig.tight_layout()
    save(fig, "fig1_loss_curves.png")

    # -- Fig 2: Val Dice ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_line(ax, train_logs, "val_dice",
              "Dice Coefficient", "Validation Dice vs. Epoch")
    fig.tight_layout()
    save(fig, "fig2_val_dice_curve.png")

    # -- Fig 3: Val IoU -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_line(ax, train_logs, "val_iou",
              "IoU (Jaccard)", "Validation IoU vs. Epoch")
    fig.tight_layout()
    save(fig, "fig3_val_iou_curve.png")

    # -- Fig 4: Val HD95 ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_line(ax, train_logs, "val_hd95",
              "HD95 (pixels)", "Validation HD95 vs. Epoch (lower = better)",
              higher_better=False)
    fig.tight_layout()
    save(fig, "fig4_val_hd95_curve.png")

    # -- Fig 5: Val Sensitivity -----------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_line(ax, train_logs, "val_sensitivity",
              "Sensitivity", "Validation Sensitivity vs. Epoch")
    fig.tight_layout()
    save(fig, "fig5_val_sensitivity.png")

    # -- Fig 6: Val Specificity -----------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_line(ax, train_logs, "val_specificity",
              "Specificity", "Validation Specificity vs. Epoch")
    fig.tight_layout()
    save(fig, "fig6_val_specificity.png")

    # -- Fig 7: LR schedule ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, data in train_logs.items():
        if data is None:
            continue
        ax.step(data["epoch"], data["lr"],
                color=COLORS[name], linewidth=1.8,
                label=LABELS[name], where="post")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Learning Rate", fontsize=9)
    ax.set_title("Learning Rate Schedule", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig7_lr_schedule.png")

    # -- Fig 8: 3x2 OVERVIEW (use this in the LaTeX report) ------------------
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 10))
    fig.suptitle("FSHA-Net vs. U-Net  —  Full Training Dynamics",
                 fontsize=12, fontweight="bold", y=1.01)

    specs = [
        ("train_loss",      "Train Loss",       "Training Loss",       False),
        ("val_loss",        "Val Loss",         "Validation Loss",     False),
        ("val_dice",        "Dice",             "Validation Dice",     True),
        ("val_iou",         "IoU",              "Validation IoU",      True),
        ("val_hd95",        "HD95 (px)",        "Validation HD95 (lower=better)", False),
        ("val_sensitivity", "Sensitivity",      "Validation Sensitivity", True),
    ]
    for ax, (key, ylabel, title, hb) in zip(axes.flat, specs):
        plot_line(ax, train_logs, key, ylabel, title, higher_better=hb)

    # Single legend below grid
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, leg_labels,
               loc="lower center", ncol=2, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, -0.06,
             "Dashed vertical lines mark LR-halving events (ReduceLROnPlateau)."
             "  Stars mark best epoch.",
             ha="center", fontsize=7.5, color="gray")
    fig.tight_layout()
    save(fig, "fig8_training_overview.png")


def make_eval_figures(eval_logs):
    any_data = any(v is not None for v in eval_logs.values())
    if not any_data:
        print("  [skip] No eval logs found -- skipping eval figures.")
        return

    # collect available models
    available = {k: v for k, v in eval_logs.items() if v is not None}

    # -- Fig 9: Per-sample Dice bar chart -------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.8))
    n_samples = max(len(d["sample"]) for d in available.values())
    x = np.arange(1, n_samples + 1)
    width = 0.38
    offsets = [-width/2, width/2]
    for i, (name, data) in enumerate(available.items()):
        bars = ax.bar(x[:len(data["dice"])] + offsets[i],
                      data["dice"],
                      width=width, color=COLORS[name], alpha=0.85,
                      label=LABELS[name])
    ax.axhline(y=np.mean(list(available.values())[0]["dice"]),
               color=COLORS[list(available.keys())[0]],
               linestyle="--", linewidth=0.9, alpha=0.7)
    if len(available) == 2:
        ax.axhline(y=np.mean(list(available.values())[1]["dice"]),
                   color=COLORS[list(available.keys())[1]],
                   linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Test Image Index", fontsize=9)
    ax.set_ylabel("Dice Coefficient", fontsize=9)
    ax.set_title("Per-Sample Dice on DRIVE Test Set (20 images)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig9_eval_per_sample_dice.png")

    # -- Fig 10: Per-sample HD95 bar chart ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (name, data) in enumerate(available.items()):
        ax.bar(x[:len(data["hd95"])] + offsets[i],
               data["hd95"],
               width=width, color=COLORS[name], alpha=0.85,
               label=LABELS[name])
    ax.set_xlabel("Test Image Index", fontsize=9)
    ax.set_ylabel("HD95 (pixels)", fontsize=9)
    ax.set_title("Per-Sample HD95 on DRIVE Test Set (lower = better)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig10_eval_per_sample_hd95.png")

    # -- Fig 11: Radar / spider chart -----------------------------------------
    metrics      = ["Dice", "IoU", "Sensitivity", "Specificity", "1-HD95_norm"]
    metric_keys  = ["dice", "iou", "sensitivity", "specificity", "hd95"]

    # normalise HD95: convert to 0-1 score where 1 = perfect (HD95=0)
    # use max HD95 across both models as denominator
    all_hd95 = []
    for data in available.values():
        all_hd95.extend(data["hd95"])
    max_hd95 = max(all_hd95) if all_hd95 else 1.0

    def get_radar_values(data):
        d   = np.mean(data["dice"])
        iou = np.mean(data["iou"])
        sn  = np.mean(data["sensitivity"])
        sp  = np.mean(data["specificity"])
        hd  = 1.0 - (np.mean(data["hd95"]) / max_hd95)  # invert so higher=better
        return [d, iou, sn, sp, hd]

    N    = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(5, 5),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Dice", "IoU", "Sensitivity",
                         "Specificity", "HD95\n(inverted)"],
                        fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.grid(color="gray", linewidth=0.5, alpha=0.5)

    for name, data in available.items():
        values = get_radar_values(data)
        values += values[:1]
        ax.plot(angles, values, color=COLORS[name],
                linewidth=2, label=LABELS[name])
        ax.fill(angles, values, color=COLORS[name], alpha=0.12)

    ax.set_title("Metric Radar Chart — Test Set",
                 fontsize=10, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0.8)
    fig.tight_layout()
    save(fig, "fig11_eval_radar.png")

    # -- Fig 12: Grouped bar chart -- summary metrics -------------------------
    metric_names  = ["Dice", "IoU", "Sensitivity", "Specificity"]
    metric_keys_s = ["dice", "iou", "sensitivity", "specificity"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos   = np.arange(len(metric_names))
    n_models = len(available)
    bar_w    = 0.35
    offsets_s = np.linspace(-(n_models-1)*bar_w/2,
                             (n_models-1)*bar_w/2, n_models)

    for i, (name, data) in enumerate(available.items()):
        means = [np.mean(data[k]) for k in metric_keys_s]
        bars = ax.bar(x_pos + offsets_s[i], means,
                      width=bar_w, color=COLORS[name],
                      alpha=0.85, label=LABELS[name])
        # value labels on bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold",
                    color=COLORS[name])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Test-Set Summary Metrics — U-Net vs. FSHA-Net",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "fig12_eval_metric_bars.png")

    # -- Fig 13: Boxplot per metric both models --------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle("Distribution of Per-Sample Metrics — DRIVE Test Set",
                 fontsize=11, fontweight="bold")

    box_metrics = [
        ("dice",        "Dice",        True),
        ("iou",         "IoU",         True),
        ("sensitivity", "Sensitivity", True),
        ("hd95",        "HD95 (px)",   False),
    ]
    for ax, (key, label, _) in zip(axes, box_metrics):
        data_lists = [data[key] for data in available.values()]
        bp = ax.boxplot(data_lists,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        for patch, name in zip(bp["boxes"], available.keys()):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABELS[n] for n in available.keys()],
                           fontsize=7.5, rotation=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    save(fig, "fig13_eval_boxplot.png")



def make_ablation_figures():
    """
    Generates a grouped bar chart from ABLATION_DATA at the top of this file.
    Replace the placeholder rows with your real ablation numbers.
    """
    variants = list(ABLATION_DATA.keys())
    values   = np.array(list(ABLATION_DATA.values()))   # shape (N_variants, 5)

    metric_names = ["Dice", "IoU", "Sensitivity", "Specificity"]
    metric_idx   = [0, 1, 3, 4]   # columns in ABLATION_DATA (skip HD95 here)

    ablation_colors = ["#AAAAAA", "#88BDE6", "#5DA5DA", "#2E86AB"]
    # grey -> light blue -> medium blue -> dark blue (U-Net to Full FSHA-Net)
    # adjust if you have more/fewer variants

    # -- Fig 14a: bar chart for Dice, IoU, Sensitivity, Specificity ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_metrics  = len(metric_names)
    n_variants = len(variants)
    x_pos      = np.arange(n_metrics)
    bar_w      = 0.18
    offsets    = np.linspace(-(n_variants-1)*bar_w/2,
                              (n_variants-1)*bar_w/2, n_variants)

    for i, (variant, color) in enumerate(zip(variants, ablation_colors)):
        vals = [values[i, j] for j in metric_idx]
        bars = ax.bar(x_pos + offsets[i], vals,
                      width=bar_w, color=color, alpha=0.88,
                      label=variant)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Ablation Study — Component Contribution",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85,
              loc="lower right", ncol=2)
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "fig14_ablation_bar.png")

    # -- Fig 14b: HD95 separate (lower is better, different scale) -----------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    hd95_vals = values[:, 2]   # column 2 = HD95
    bars = ax.bar(range(n_variants), hd95_vals,
                  color=ablation_colors[:n_variants], alpha=0.88)
    for bar, val in zip(bars, hd95_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.04,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(range(n_variants))
    ax.set_xticklabels(variants, fontsize=8.5)
    ax.set_ylabel("HD95 (pixels)  — lower is better", fontsize=9)
    ax.set_title("Ablation Study — HD95 Comparison",
                 fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig14b_ablation_hd95.png")

    print("  NOTE: Ablation figures use data from ABLATION_DATA dict")
    print("        at the top of plot_curves.py.")
    print("        Replace placeholder values with your real ablation results.")



def main():
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== plot_curves.py ===")
    print(f"  Output folder: {CURVES_DIR}\n")

    # Load all logs
    train_logs = {
        "unet":    read_train_log("unet"),
        "fshanet": read_train_log("fshanet"),
    }
    eval_logs = {
        "unet":    read_eval_log("unet"),
        "fshanet": read_eval_log("fshanet"),
    }

    no_train = all(v is None for v in train_logs.values())
    no_eval  = all(v is None for v in eval_logs.values())
    if no_train and no_eval:
        print("\n  No CSV logs found at all.")
        print("  Train first:  python train.py --model unet")
        print("                python train.py --model fshanet")
        print("  Eval first:   python eval.py  --model both")
        sys.exit(1)

    print("\n--- Training curves ---")
    make_training_figures(train_logs)

    print("\n--- Evaluation charts ---")
    make_eval_figures(eval_logs)

    print("\n--- Ablation charts ---")
    make_ablation_figures()

    print("\n=== ALL DONE ===")
    print("Files saved in  curves/ :")
    for f in sorted(CURVES_DIR.glob("fig*.png")):
        print(f"  {f.name}")

    print("""
LaTeX snippet for the Training Dynamics figure (use fig8):
  \\includegraphics[width=\\linewidth]{curves/fig8_training_overview.png}

LaTeX snippet for Evaluation bar chart:
  \\includegraphics[width=\\linewidth]{curves/fig12_eval_metric_bars.png}

LaTeX snippet for Radar chart:
  \\includegraphics[width=0.6\\linewidth]{curves/fig11_eval_radar.png}

LaTeX snippet for Ablation:
  \\includegraphics[width=\\linewidth]{curves/fig14_ablation_bar.png}
""")


if __name__ == "__main__":
    main()
