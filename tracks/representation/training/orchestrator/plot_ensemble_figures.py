import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ..shared.soft_label_utils import masked_class_data, _soft_average_precision, _soft_roc_auc, _soft_roc_curve, _soft_pr_curve


def compute_ensemble_curves(probs, y_true, class_list, cfg) -> dict:
    """Compute all ROC/PR curve data without any filesystem I/O.

    Returns a dict with keys:
        y_micro, p_micro  — concatenated masked arrays for micro-average curves
        per_class         — list of {cls, y_c, probs_c} for classes with data
        colors            — colormap array, one row per class
    """
    micro_true = []
    micro_probs = []
    per_class = []

    for i, cls in enumerate(class_list):
        _, y_c, probs_c = masked_class_data(y_true, probs, i, cfg)
        if y_c.size > 0:
            micro_true.append(y_c)
            micro_probs.append(probs_c)
            per_class.append({"cls": cls, "y_c": y_c, "probs_c": probs_c})

    if micro_true:
        y_micro = np.concatenate(micro_true)
        p_micro = np.concatenate(micro_probs)
    else:
        y_micro = np.array([], dtype=int)
        p_micro = np.array([], dtype=np.float32)

    colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(class_list)))

    return {
        "y_micro": y_micro,
        "p_micro": p_micro,
        "per_class": per_class,
        "colors": colors,
        "class_list": list(class_list),
    }


def write_ensemble_figures(curves: dict, figures_dir: str) -> None:
    """Write ROC and PR curve figures to disk. Receives pre-computed curve data."""
    y_micro = curves["y_micro"]
    p_micro = curves["p_micro"]
    per_class = curves["per_class"]
    colors = curves["colors"]
    class_list = curves["class_list"]

    roc_dir = os.path.join(figures_dir, "ROC")
    pr_dir = os.path.join(figures_dir, "PR")
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    def _save_empty_plot(path, title):
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, title, ha='center', va='center')
        plt.axis('off')
        plt.savefig(path)
        plt.close()

    # 1. ROC Curves
    if y_micro.size > 0:
        plt.figure(figsize=(10, 8))
        fpr_micro, tpr_micro = _soft_roc_curve(y_micro, p_micro)
        auc_micro = _soft_roc_auc(y_micro, p_micro)
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC={auc_micro:.2f})', linestyle=':', linewidth=3, color='deeppink')

        for i, entry in enumerate(per_class):
            fpr, tpr = _soft_roc_curve(entry["y_c"], entry["probs_c"])
            auc_c = _soft_roc_auc(entry["y_c"], entry["probs_c"])
            cls_idx = class_list.index(entry["cls"])
            plt.plot(fpr, tpr, color=colors[cls_idx], label=f'{entry["cls"]} (AUC={auc_c:.2f})', linewidth=2, alpha=0.8)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(roc_dir, "ROC_Ensemble_Total.png"))
        plt.close()
    else:
        _save_empty_plot(os.path.join(roc_dir, "ROC_Ensemble_Total.png"), "No valid ROC data")

    # 2. PR Curves
    if y_micro.size > 0:
        plt.figure(figsize=(10, 8))
        ap_micro = _soft_average_precision(y_micro, p_micro)
        prec_m, rec_m = _soft_pr_curve(y_micro, p_micro)
        plt.plot(rec_m, prec_m, label=f'Micro-average (AP={ap_micro:.2f})', linestyle=':', linewidth=3, color='navy')

        for i, entry in enumerate(per_class):
            ap_c = _soft_average_precision(entry["y_c"], entry["probs_c"])
            prec_c, rec_c = _soft_pr_curve(entry["y_c"], entry["probs_c"])
            cls_idx = class_list.index(entry["cls"])
            plt.plot(rec_c, prec_c, color=colors[cls_idx], label=f'{entry["cls"]} (AP={ap_c:.2f})', linewidth=2, alpha=0.8)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Ensemble PR Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(pr_dir, "PR_Ensemble_Total.png"))
        plt.close()
    else:
        _save_empty_plot(os.path.join(pr_dir, "PR_Ensemble_Total.png"), "No valid PR data")

    # 3. Per Class Separate Plots
    for entry in per_class:
        cls = entry["cls"]
        clean_cls = str(cls).replace(' ', '_').replace('/', '_')
        y_c = entry["y_c"]
        probs_c = entry["probs_c"]

        fpr, tpr = _soft_roc_curve(y_c, probs_c)
        roc_auc = _soft_roc_auc(y_c, probs_c)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC - {cls}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(roc_dir, f"ROC_{clean_cls}.png"))
        plt.close()

        ap = _soft_average_precision(y_c, probs_c)
        prec, rec = _soft_pr_curve(y_c, probs_c)
        plt.figure()
        plt.plot(rec, prec, label=f'AP={ap:.2f}')
        plt.title(f"PR - {cls}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(pr_dir, f"PR_{clean_cls}.png"))
        plt.close()
