"""Per-stage/per-split outputs for missing, patient, class, text, and label artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..logic.divisions import numeric_binary, resolve_class_cols, trim_raw_item
from ..utils.io_utils import output_enabled, save_plot, skip, write_csv


# ---------------------------------------------------------------------------
# Private helpers called from write_stage_split_outputs
# ---------------------------------------------------------------------------

def _write_missingness_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    stage: str,
    split: str,
) -> None:
    if output_enabled(summary_cfg, "missing_csv", default=True):
        missing_count = df.isna().sum().rename_axis("column").reset_index(name="missing_count")
        missing_pct = ((df.isna().sum() / max(len(df), 1)) * 100.0).rename_axis("column").reset_index(
            name="missing_percentage"
        )
        missing = missing_count.merge(missing_pct, on="column", how="left")
        write_csv(result, strict, missing, out_dir / "missing.csv")


def _write_distribution_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    df: pd.DataFrame,
    class_cols: list[str],
    out_dir: Path,
    stage: str,
    split: str,
    figsize: tuple[float, float],
    dpi: int,
    bar_top_n: int,
    heatmap_max_labels: int,
    item_col: str,
    text_length_bins: int,
    patient_col: str | None,
) -> None:
    if output_enabled(summary_cfg, "patient_distribution_csv", default=True):
        if patient_col and patient_col in df.columns:
            patient_counts = df[patient_col].value_counts(dropna=False).rename_axis(patient_col).reset_index(
                name="count"
            )
            patient_counts["percentage"] = (patient_counts["count"] / max(len(df), 1)) * 100.0
            write_csv(result, strict, patient_counts, out_dir / "patient_distribution.csv")
        else:
            skip(result, f"{stage}/{split}/patient_distribution.csv (missing patient column)")

    class_dist_csv_on = output_enabled(summary_cfg, "class_distribution_csv", default=True)
    class_dist_plot_on = output_enabled(summary_cfg, "class_distribution_plot", default=True)
    if class_dist_csv_on or class_dist_plot_on:
        if class_cols:
            binary = numeric_binary(df, class_cols)
            dist = binary.sum(axis=0).rename_axis("class").reset_index(name="count")
            dist["percentage"] = (dist["count"] / max(len(binary), 1)) * 100.0
            dist = dist.sort_values("count", ascending=False).reset_index(drop=True)
            if class_dist_csv_on:
                write_csv(result, strict, dist, out_dir / "class_distribution.csv")
            if class_dist_plot_on:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                top = dist.head(bar_top_n)
                ax.bar(top["class"], top["percentage"])
                ax.set_title(f"Class distribution (%) - {stage}/{split}")
                ax.set_xlabel("Class")
                ax.set_ylabel("Percentage")
                ax.tick_params(axis="x", rotation=90)
                save_plot(result, strict, fig, out_dir / "class_distribution.png")
        else:
            skip(result, f"{stage}/{split}/class_distribution (missing class columns)")

    class_cooc_csv_on = output_enabled(summary_cfg, "class_cooccurrence_csv", default=True)
    class_cooc_plot_on = output_enabled(summary_cfg, "class_cooccurrence_plot", default=True)
    if class_cooc_csv_on or class_cooc_plot_on:
        if class_cols:
            labels = class_cols[:heatmap_max_labels]
            binary = numeric_binary(df, labels)
            cooc = binary.T.dot(binary).reset_index().rename(columns={"index": "class"})
            if class_cooc_csv_on:
                write_csv(result, strict, cooc, out_dir / "class_cooccurrence.csv")
            if class_cooc_plot_on:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                image = ax.imshow(binary.T.dot(binary).to_numpy(), aspect="auto")
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
                ax.set_title(f"Class co-occurrence - {stage}/{split}")
                fig.colorbar(image, ax=ax)
                save_plot(result, strict, fig, out_dir / "class_cooccurrence.png")
        else:
            skip(result, f"{stage}/{split}/class_cooccurrence (missing class columns)")

    if output_enabled(summary_cfg, "text_length_csv", default=True):
        if item_col in df.columns:
            lengths = df[item_col].map(lambda v: 0 if pd.isna(v) else len(str(v).split()))
            stats = {
                "total_rows": int(len(lengths)),
                "median_word_count": float(lengths.median()) if len(lengths) else 0.0,
                "min_word_count": int(lengths.min()) if len(lengths) else 0,
                "max_word_count": int(lengths.max()) if len(lengths) else 0,
                "mean_word_count": float(lengths.mean()) if len(lengths) else 0.0,
                "std_word_count": float(lengths.std(ddof=0)) if len(lengths) else 0.0,
            }
            n_bins = max(1, text_length_bins)
            min_len = int(lengths.min()) if len(lengths) else 0
            max_len = int(lengths.max()) if len(lengths) else 0
            if min_len == max_len:
                bins = [min_len - 0.5, min_len + 0.5]
                n_bins = 1
            else:
                step = (max_len - min_len) / n_bins
                bins = [min_len + i * step for i in range(n_bins + 1)]
            binned = pd.cut(lengths, bins=bins, include_lowest=True)
            bin_counts = binned.value_counts(sort=False).rename_axis("bin_interval").reset_index(name="count")
            bin_counts["bin_label"] = bin_counts["bin_interval"].astype(str)
            bin_counts["percentage"] = (bin_counts["count"] / max(len(lengths), 1)) * 100.0
            dist = bin_counts[["bin_label", "count", "percentage"]].copy()
            for key, value in stats.items():
                dist[key] = value
            write_csv(result, strict, dist, out_dir / "text_length.csv")
        else:
            skip(result, f"{stage}/{split}/text_length_distribution (missing item column)")


def _write_patient_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    df: pd.DataFrame,
    class_cols: list[str],
    out_dir: Path,
    stage: str,
    split: str,
    patient_col: str | None,
) -> None:
    if output_enabled(summary_cfg, "per_patient_class_count_csv", default=True):
        if patient_col and patient_col in df.columns and class_cols:
            numeric = df[class_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            grouped = pd.concat([df[[patient_col]], numeric], axis=1).groupby(patient_col, dropna=False)
            out = grouped.size().rename("row_count").to_frame().reset_index()
            sums = grouped[class_cols].sum().reset_index()
            out = out.merge(sums, on=patient_col, how="left")
            write_csv(result, strict, out, out_dir / "per_patient_class_count.csv")
        else:
            skip(result, f"{stage}/{split}/per_patient_class_count.csv (missing columns)")

    if output_enabled(summary_cfg, "label_cardinality_csv", default=True):
        if class_cols:
            binary = numeric_binary(df, class_cols)
            labels_per_row = binary.sum(axis=1)
            stats = {
                "total_rows": int(len(labels_per_row)),
                "mean_labels_per_row": float(labels_per_row.mean()) if len(labels_per_row) else 0.0,
                "pct_rows_0_labels": float((labels_per_row == 0).mean() * 100.0) if len(labels_per_row) else 0.0,
                "pct_rows_1_label": float((labels_per_row == 1).mean() * 100.0) if len(labels_per_row) else 0.0,
                "pct_rows_2_labels": float((labels_per_row == 2).mean() * 100.0) if len(labels_per_row) else 0.0,
                "pct_rows_3plus_labels": float((labels_per_row >= 3).mean() * 100.0)
                if len(labels_per_row)
                else 0.0,
            }
            dist = labels_per_row.value_counts().sort_index().rename_axis("labels_per_row").reset_index(name="count")
            dist["percentage_rows"] = (dist["count"] / max(len(labels_per_row), 1)) * 100.0
            for key, value in stats.items():
                dist[key] = value
            write_csv(result, strict, dist, out_dir / "label_cardinality.csv")
        else:
            skip(result, f"{stage}/{split}/label_cardinality (missing class columns)")

    rows_per_patient_on = output_enabled(summary_cfg, "rows_per_patient_csv", default=True)
    labels_per_patient_on = output_enabled(summary_cfg, "labels_per_patient_summary_csv", default=True)
    if rows_per_patient_on or labels_per_patient_on:
        if patient_col and patient_col in df.columns:
            row_counts = df[patient_col].value_counts(dropna=False).rename_axis(patient_col).reset_index(
                name="row_count"
            )
            if rows_per_patient_on:
                write_csv(result, strict, row_counts, out_dir / "rows_per_patient.csv")
            if class_cols and labels_per_patient_on:
                binary = numeric_binary(df, class_cols)
                per_patient = pd.concat([df[[patient_col]], binary], axis=1).groupby(patient_col, dropna=False).sum()
                per_patient["total_positive_labels"] = per_patient.sum(axis=1)
                per_patient_df = per_patient.reset_index()
                write_csv(result, strict, per_patient_df, out_dir / "labels_per_patient_summary.csv")
        else:
            skip(result, f"{stage}/{split}/per_patient_burden (missing patient column)")


def _write_duplicate_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    stage: str,
    split: str,
    item_col: str,
    patient_col: str | None,
    physician_col: str | None,
) -> None:
    if not (
        output_enabled(summary_cfg, "duplicate_profile_csv", default=True)
        or output_enabled(summary_cfg, "duplicate_samples_csv", default=True)
    ):
        return
    hashable_cols = [
        c
        for c in df.columns
        if not (df[c].dtype == object and df[c].map(lambda v: isinstance(v, list)).any())
    ]
    dup_df = df[hashable_cols] if hashable_cols else df
    metrics: list[dict[str, Any]] = [{"metric": "exact_duplicate_rows", "count": int(dup_df.duplicated().sum())}]
    if patient_col and patient_col in df.columns and item_col in df.columns:
        duplicate_patient_item = int((df.groupby([patient_col, item_col], dropna=False).size() > 1).sum())
        metrics.append({"metric": "duplicate_patient_item_groups", "count": duplicate_patient_item})
    if (
        physician_col
        and physician_col in df.columns
        and patient_col
        and patient_col in df.columns
        and item_col in df.columns
    ):
        duplicate_triplet = int(df.duplicated(subset=[patient_col, item_col, physician_col]).sum())
        metrics.append({"metric": "duplicate_patient_item_physician_rows", "count": duplicate_triplet})
    if item_col in df.columns:
        duplicate_item = int((df.groupby(item_col, dropna=False).size() > 1).sum())
        metrics.append({"metric": "duplicate_item_groups", "count": duplicate_item})
    if output_enabled(summary_cfg, "duplicate_profile_csv", default=True):
        write_csv(result, strict, pd.DataFrame(metrics), out_dir / "duplicate_profile.csv")
    # Always write duplicate_samples.csv; empty body when no duplicates exist.
    sample = dup_df[dup_df.duplicated(keep=False)].head(200)
    if output_enabled(summary_cfg, "duplicate_samples_csv", default=True):
        write_csv(result, strict, sample, out_dir / "duplicate_samples.csv")


def _write_label_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    df: pd.DataFrame,
    class_cols: list[str],
    out_dir: Path,
    stage: str,
    split: str,
    patient_col: str | None,
) -> None:
    if output_enabled(summary_cfg, "per_label_patient_positive_buckets_csv", default=True):
        if class_cols:
            binary = numeric_binary(df, class_cols)
            rows: list[dict[str, Any]] = []
            for label in class_cols:
                positives = int(binary[label].sum())
                row: dict[str, Any] = {
                    "class": label,
                    "total_positives": positives,
                    "prevalence_percentage": (positives / max(len(binary), 1)) * 100.0,
                }
                if patient_col and patient_col in df.columns:
                    patient_counts = (
                        pd.concat([df[[patient_col]], binary[[label]]], axis=1)
                        .groupby(patient_col, dropna=False)[label]
                        .sum()
                    )
                    row["patients_zero_positives"] = int((patient_counts == 0).sum())
                    row["patients_exactly_1_positive"] = int((patient_counts == 1).sum())
                    row["patients_exactly_2_positives"] = int((patient_counts == 2).sum())
                    row["patients_exactly_3_positives"] = int((patient_counts == 3).sum())
                    row["patients_exactly_4_positives"] = int((patient_counts == 4).sum())
                    row["patients_lt5_positives"] = int((patient_counts < 5).sum())
                rows.append(row)
            write_csv(result, strict, pd.DataFrame(rows), out_dir / "per_label_patient_positive_buckets.csv")
        else:
            skip(result, f"{stage}/{split}/per_label_patient_positive_buckets.csv (missing class columns)")

    if output_enabled(summary_cfg, "fractional_label_audit_csv", default=True) or output_enabled(
        summary_cfg, "fractional_values_csv", default=True
    ):
        if class_cols:
            numeric = df[class_cols].apply(pd.to_numeric, errors="coerce")
            rows2: list[dict[str, Any]] = []
            fractional_rows: list[pd.DataFrame] = []
            for label in class_cols:
                values = numeric[label].dropna()
                fractional_mask = (values != 0) & (values != 1)
                rows2.append(
                    {
                        "class": label,
                        "exact_0_count": int((values == 0).sum()),
                        "exact_1_count": int((values == 1).sum()),
                        "fractional_count": int(fractional_mask.sum()),
                        "label_type": "fractional" if bool(fractional_mask.any()) else "binary",
                    }
                )
                if fractional_mask.any():
                    vc = values[fractional_mask].value_counts().rename_axis("value").reset_index(name="count")
                    vc.insert(0, "class", label)
                    fractional_rows.append(vc)
            audit = pd.DataFrame(rows2)
            if output_enabled(summary_cfg, "fractional_label_audit_csv", default=True):
                write_csv(result, strict, audit, out_dir / "fractional_label_audit.csv")
            if fractional_rows and output_enabled(summary_cfg, "fractional_values_csv", default=True):
                frac = pd.concat(fractional_rows, axis=0, ignore_index=True)
                frac["class_total_fractional_count"] = (
                    frac.groupby("class", dropna=False)["count"].transform("sum").astype(int)
                )
                frac["value_percentage_within_class_fractional"] = (
                    frac["count"] / frac["class_total_fractional_count"].replace(0, pd.NA)
                ) * 100.0
                frac["global_rank_by_count"] = frac["count"].rank(method="dense", ascending=False).astype(int)
                write_csv(result, strict, frac, out_dir / "fractional_values.csv")
        else:
            skip(result, f"{stage}/{split}/fractional_label_audit (missing class columns)")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def write_stage_split_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    stage_data: dict[str, dict[str, pd.DataFrame]],
    selected_stages: list[str],
    summary_dir: Path,
    item_col: str,
    patient_col: str | None,
    physician_col: str | None,
    metadata: dict[str, Any],
    figsize: tuple[float, float],
    dpi: int,
    bar_top_n: int,
    heatmap_max_labels: int,
    text_length_bins: int = 20,
) -> None:
    for stage in selected_stages:
        for split, df in stage_data[stage].items():
            out_dir = (
                summary_dir / "final"
                if stage == "final"
                else summary_dir / "preprocessing_stages" / stage / split
            )
            class_cols = resolve_class_cols(df, metadata, stage, split)

            _write_missingness_outputs(
                result=result, strict=strict, summary_cfg=summary_cfg,
                df=df, out_dir=out_dir, stage=stage, split=split,
            )
            _write_distribution_outputs(
                result=result, strict=strict, summary_cfg=summary_cfg,
                df=df, class_cols=class_cols, out_dir=out_dir, stage=stage, split=split,
                figsize=figsize, dpi=dpi, bar_top_n=bar_top_n, heatmap_max_labels=heatmap_max_labels,
                item_col=item_col, text_length_bins=text_length_bins, patient_col=patient_col,
            )
            _write_patient_outputs(
                result=result, strict=strict, summary_cfg=summary_cfg,
                df=df, class_cols=class_cols, out_dir=out_dir, stage=stage, split=split,
                patient_col=patient_col,
            )
            _write_duplicate_outputs(
                result=result, strict=strict, summary_cfg=summary_cfg,
                df=df, out_dir=out_dir, stage=stage, split=split,
                item_col=item_col, patient_col=patient_col, physician_col=physician_col,
            )
            _write_label_outputs(
                result=result, strict=strict, summary_cfg=summary_cfg,
                df=df, class_cols=class_cols, out_dir=out_dir, stage=stage, split=split,
                patient_col=patient_col,
            )


# ---------------------------------------------------------------------------
# Item frequency outputs
# ---------------------------------------------------------------------------

def write_item_frequency_outputs(
    *,
    result: dict[str, Any],
    strict: bool,
    summary_cfg: dict[str, Any],
    summary_dir: Path,
    split_names: list[str],
    stage_data: dict[str, dict[str, pd.DataFrame]],
    item_col: str,
) -> None:
    if not (
        output_enabled(summary_cfg, "item_frequency_raw_csv", default=True)
        or output_enabled(summary_cfg, "item_frequency_standardized_csv", default=True)
    ):
        return

    for split in split_names:
        raw_df = stage_data.get("raw", {}).get(split)
        if raw_df is not None and output_enabled(
            summary_cfg,
            "item_frequency_raw_csv",
            default=True,
        ):
            if item_col in raw_df.columns:
                s = raw_df[item_col].map(trim_raw_item)
                counts = s.dropna().astype(str).value_counts().rename_axis("item").reset_index(name="count")
                counts["percentage"] = (counts["count"] / max(len(raw_df), 1)) * 100.0
                write_csv(
                    result,
                    strict,
                    counts,
                    summary_dir / "preprocessing_stages" / "raw" / split / "item_frequency_raw.csv",
                )
            else:
                skip(result, f"raw/{split}/item_frequency_raw.csv (missing item column)")

        std_df = stage_data.get("standardized", {}).get(split)
        if std_df is not None and output_enabled(
            summary_cfg,
            "item_frequency_standardized_csv",
            default=True,
        ):
            if item_col in std_df.columns:
                s = std_df[item_col].map(lambda v: None if pd.isna(v) else str(v).strip())
                counts = s.dropna().astype(str).value_counts().rename_axis("item").reset_index(name="count")
                counts["percentage"] = (counts["count"] / max(len(std_df), 1)) * 100.0
                write_csv(
                    result,
                    strict,
                    counts,
                    summary_dir / "preprocessing_stages" / "standardized" / split / "item_frequency_standardized.csv",
                )
            else:
                skip(result, f"standardized/{split}/item_frequency_standardized.csv (missing item column)")
