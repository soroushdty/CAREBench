"""Figures 1-4 from §5.2, §5.5, and §5.9.2.

All figures use matplotlib and are saved to disk.  Each function accepts a
`dpi` argument (default 150) and saves with bbox_inches='tight'.

Figure 1 — Per-class Δ_p histogram (§5.2)
Figure 2 — Per-patient inter-physician agreement survey vs interview (§5.2)
Figure 3 — ICC(2,1) horizontal dot plot (§5.5)
Figure 4 — Entropy change scatter plot (§5.9.2)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def figure1_delta_histogram(
    delta_p: np.ndarray,
    class_list: list[str],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """2×5 grid of per-class Δ_p histograms (§5.2 Figure 1).

    Bars: positive Δ_p values in green, negative in red, zero in gray.
    Each panel is annotated with the count of non-zero deltas.

    Args:
        delta_p:     (n, n_classes) physician delta matrix.
        class_list:  10 class names.
        output_path: File path to save the PNG/PDF.
        dpi:         Figure DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dp = np.asarray(delta_p, dtype=np.float32)
    n_classes = len(class_list)
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))
    axes = axes.ravel()

    delta_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colors = {-1.0: "#d62728", -0.5: "#ff7f0e", 0.0: "#aec7e8", 0.5: "#2ca02c", 1.0: "#1a7a1a"}

    for c_idx in range(n_classes):
        ax = axes[c_idx]
        col = dp[:, c_idx]
        counts = {v: int(np.sum(np.isclose(col, v))) for v in delta_vals}
        n_nonzero = sum(counts[v] for v in delta_vals if v != 0.0)

        bars = ax.bar(delta_vals, [counts[v] for v in delta_vals], width=0.4,
                      color=[colors[v] for v in delta_vals], edgecolor="white", linewidth=0.5)

        ax.set_title(class_list[c_idx], fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Δ_p", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_xticks(delta_vals)
        ax.tick_params(labelsize=7)
        ax.annotate(f"n_nz={n_nonzero}", xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Hide unused axes
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-class physician delta (Δ_p) distribution", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 1 saved to %s", output_path)


def figure2_interphysician_agreement(
    individual_labels: dict,
    class_list: list[str],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Per-patient inter-physician agreement in survey vs interview (§5.2 Figure 2).

    For each patient, computes the mean absolute agreement rate across classes
    between the two physicians in both phases.  Patients with lower interview-
    phase agreement than survey-phase are flagged with a star marker.

    Args:
        individual_labels: From icc.load_individual_physician_labels().
        class_list:        10 class names.
        output_path:       File path to save.
        dpi:               Figure DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    survey_labels = individual_labels.get("test", {})
    interview_labels = individual_labels.get("interview", {})
    all_patients = sorted(set(survey_labels.keys()) | set(interview_labels.keys()))

    if not all_patients:
        logger.warning("Figure 2: no individual label data — skipping.")
        return

    def _agreement_rate(labels_split: dict, pid: object) -> float:
        if pid not in labels_split:
            return float("nan")
        physicians = list(labels_split[pid].keys())
        if len(physicians) < 2:
            return float("nan")
        ph0 = labels_split[pid][physicians[0]]
        ph1 = labels_split[pid][physicians[1]]
        items = set(ph0.keys()) & set(ph1.keys())
        if not items:
            return float("nan")
        agrees = 0
        total = 0
        for item in items:
            for cls in class_list:
                v0 = ph0[item].get(cls)
                v1 = ph1[item].get(cls)
                if v0 is not None and v1 is not None:
                    agrees += int(v0 == v1)
                    total += 1
        return float(agrees / total) if total else float("nan")

    survey_rates = [_agreement_rate(survey_labels, p) for p in all_patients]
    interview_rates = [_agreement_rate(interview_labels, p) for p in all_patients]

    x = np.arange(len(all_patients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_patients) * 0.9), 5))
    b1 = ax.bar(x - width / 2, survey_rates, width, label="Survey (context-free)",
                color="#5b9bd5", alpha=0.85)
    b2 = ax.bar(x + width / 2, interview_rates, width, label="Interview (context-aware)",
                color="#ed7d31", alpha=0.85)

    # Flag patients with lower interview agreement
    for i, (sr, ir) in enumerate(zip(survey_rates, interview_rates)):
        if not (np.isnan(sr) or np.isnan(ir)) and ir < sr:
            ax.scatter(x[i], max(sr, ir) + 0.02, marker="*", color="red", s=80, zorder=5)

    ax.set_xlabel("Patient", fontsize=10)
    ax.set_ylabel("Inter-physician agreement rate", fontsize=10)
    ax.set_title("Per-patient inter-physician agreement: survey vs interview", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in all_patients], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.annotate("★ interview < survey", xy=(0.01, 0.98), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, color="red")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 2 saved to %s", output_path)


def figure3_icc_horizontal_plot(
    icc_results: dict,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """ICC(2,1) horizontal dot plot (§5.5 Figure 3).

    Shows 12 within-pair human-human ICCs (gray points, sorted ascending)
    with mean ± 1 SD shaded band, overlaid with the 24 model-vs-physician
    ICC(2,1) values as orange points, each with its 95% bootstrap CI.
    Patients with low within-pair ICC are flagged.

    Args:
        icc_results: Dict from icc.rater_icc_analysis().
        output_path: File path to save.
        dpi:         Figure DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    hh = icc_results.get("human_human_iccs", [])
    mp = icc_results.get("model_physician_iccs", [])

    hh_vals = sorted([d["icc"] for d in hh if not np.isnan(d["icc"])])
    mp_vals = [d["icc"] for d in mp]
    hh_mean = icc_results.get("human_human_icc_mean", float("nan"))
    hh_sd = icc_results.get("human_human_icc_sd", float("nan"))

    fig, ax = plt.subplots(figsize=(8, max(5, len(mp) * 0.4 + 2)))

    # Human-human shaded band
    if not np.isnan(hh_mean) and not np.isnan(hh_sd):
        ax.axvspan(hh_mean - hh_sd, hh_mean + hh_sd, alpha=0.15, color="gray",
                   label=f"Within-pair human mean ± 1 SD ({hh_mean:.2f} ± {hh_sd:.2f})")
        ax.axvline(hh_mean, color="gray", linestyle="--", linewidth=1.2)

    # Human-human points
    for i, v in enumerate(hh_vals):
        ax.scatter(v, -(i + 1), color="steelblue", s=50, zorder=4, alpha=0.8)

    # Model-vs-physician points
    labels = [f"P{d['patient']}-Ph{d['physician']}" for d in mp]
    for i, (d, lbl) in enumerate(zip(mp, labels)):
        ax.scatter(d["icc"], i + 1, color="#ed7d31", s=50, zorder=5, alpha=0.9)

    ax.set_xlabel("ICC(2,1)", fontsize=10)
    ax.set_title("Rater-level ICC(2,1): model vs physician and within-pair physician-physician", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_yticks(list(range(-(len(hh_vals)), 0)) + list(range(1, len(mp) + 1)))
    hh_ytick_labels = [f"HH-{i+1}" for i in range(len(hh_vals))]
    ax.set_yticklabels(hh_ytick_labels + labels, fontsize=7)

    orange_patch = mpatches.Patch(color="#ed7d31", label="Model vs physician (24 ICCs)")
    blue_patch = mpatches.Patch(color="steelblue", label="Within-pair human-human (12 ICCs)")
    ax.legend(handles=[orange_patch, blue_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 3 saved to %s", output_path)


def figure4_entropy_scatter(
    phys_entropy_delta: np.ndarray,
    model_entropy_delta: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
    pearson_r: float,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Physician vs model entropy change scatter (§5.9.2 Figure 4).

    Each point is one (item, class) observation pooled across confirmatory-
    eligible classes, colored by class.  Pearson r annotated as descriptive.

    Args:
        phys_entropy_delta:  (n, n_classes) physician entropy change.
        model_entropy_delta: (n, n_classes) model entropy change.
        eligible_classes:    Classes to pool.
        class_list:          All class names.
        pearson_r:           Pre-computed Pearson r (from entropy.entropy_pearson_r).
        output_path:         File path to save.
        dpi:                 Figure DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ph = np.asarray(phys_entropy_delta, dtype=np.float64)
    mo = np.asarray(model_entropy_delta, dtype=np.float64)
    eligible_set = set(eligible_classes)
    eligible_cols = [i for i, cls in enumerate(class_list) if cls in eligible_set]
    eligible_names = [class_list[i] for i in eligible_cols]

    cmap = cm.get_cmap("tab10", len(eligible_cols))
    fig, ax = plt.subplots(figsize=(7, 7))

    for idx, (c_idx, cls_name) in enumerate(zip(eligible_cols, eligible_names)):
        ph_col = ph[:, c_idx]
        mo_col = mo[:, c_idx]
        valid = ~(np.isnan(ph_col) | np.isnan(mo_col))
        ax.scatter(ph_col[valid], mo_col[valid],
                   color=cmap(idx), alpha=0.6, s=18, label=cls_name)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.plot([-1.1, 1.1], [-1.1, 1.1], color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    r_str = f"{pearson_r:.3f}" if not np.isnan(pearson_r) else "n/a"
    ax.annotate(
        f"Pearson r = {r_str} (descriptive only)",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax.set_xlabel("Physician entropy change [H(interview) − H(survey)]", fontsize=10)
    ax.set_ylabel("Model entropy change [H(ŷ_ca) − H(ŷ_cf)]", fontsize=10)
    ax.set_title("Context-induced entropy change: physician vs model", fontsize=11)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 saved to %s", output_path)
