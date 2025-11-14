#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make figures from state features/labels (with seaborn) and embed clear per-figure explanations.

Generated figures:
  1) WE by condition ............................................. we_by_condition.png
  2) Occupancy entropy H(p) by condition .......................... Hocc_by_condition.png
  3) Mean occupancy heatmap (p_i averaged per subject×condition) .. occupancy_heatmap.png
  4) State entropies bar H_i (sorted) ............................. state_entropy_bar.png
  5) QC: counts by state×condition (raw window counts) ............ counts_state_condition.png
  6) QC (supplement): row-normalized % counts by condition×state .. counts_state_condition_rowpct.png
  7) Scatter: WE vs H(p) .......................................... WE_vs_Hocc.png

Notation (consistent across the project):
  - p_i        : state occupancy for state i (per subject×condition)
  - H(p)       : Shannon entropy of the occupancy vector p = (p_1..p_K)
  - H_i        : state entropy for state i (centroid/medoid-based)
  - WE         : weighted entropy = sum_i p_i * H_i

CLI (examples, adjust paths as needed):
  python scripts/07_plot_states_features.py \
      --features "data/processed/.../states_features/features_subject_condition_k5.csv" \
      --state-entropies "data/processed/.../states_features/state_entropies_k5.npy" \
      --stack "data/processed/.../stack_...tau...npz" \
      --state-labels "data/processed/.../states_features/state_labels_k5.npy" \
      --out-dir "data/processed/.../states_features"

This script is robust to missing optional sources:
  - If --stack or --state-labels is not given, the two QC heatmaps are skipped.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------- Small utilities ---------------------------- #

def ensure_outdir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)  # avoid crash if rerun


def pick_condition_column(df: pd.DataFrame) -> str:
    """Choose the most descriptive condition column name in the features CSV."""
    # Prefer human-readable names if present
    if "condition_name" in df.columns:
        return "condition_name"  # best option for plotting
    # Fallback to numeric code if necessary
    if "condition" in df.columns:
        return "condition"
    # As a last resort, try to infer from any column containing "cond"
    for c in df.columns:
        if "cond" in c.lower():
            return c
    raise ValueError("No condition column found in features CSV.")


def annotate(ax: plt.Axes, text: str) -> None:
    """Place an explanatory textbox inside the plot (top-left, consistent styling)."""
    # Use axes coordinates so placement is resolution independent
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.6")
    )  # concise but readable explanation per figure


def load_features_csv(features_csv: str) -> pd.DataFrame:
    """Load the features CSV produced by script 06 and basic-check required columns."""
    df = pd.read_csv(features_csv)                    # read tidy features
    required_any = {"WE", "H_occ", "H_occ_norm"}     # at least these should exist
    if not required_any.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns: {required_any - set(df.columns)}")
    return df


def load_state_entropies(state_entropies_npy: str) -> np.ndarray:
    """Load state entropies H_i (length K)."""
    return np.load(state_entropies_npy)  # produced by 06


def load_stack_and_labels(stack_npz: str, labels_npy: str) -> tuple[pd.Series, pd.Series]:
    """
    Return per-window condition names and per-window state labels.

    This function tries, in order:
      1) 'event_labels_all' string array in stack (best)
      2) numeric y_cond + name map from 'conditions' in stack meta
    """
    data = np.load(stack_npz, allow_pickle=True)     # load NPZ with possible arrays and dicts
    state_labels = np.load(labels_npy)               # shape (N_windows,)
    N = int(state_labels.shape[0])                   # number of windows for alignment

    # Try direct per-window string labels
    for key in ("event_labels_all", "event_labels", "events", "labels"):
        if key in data and len(np.asarray(data[key])) == N:
            cond_series = pd.Series(np.asarray(data[key]), dtype="object")  # human-readable
            return cond_series, pd.Series(state_labels.astype(int), dtype=int)

    # Fallback: map numeric codes to names via 'conditions'
    if "y_cond" in data and len(np.asarray(data["y_cond"])) == N:
        y = np.asarray(data["y_cond"]).astype(int)   # numeric condition codes
        # Resolve names list
        cond_names = None
        if "conditions" in data:
            try:
                cond_names = list(np.asarray(data["conditions"]).tolist())
            except Exception:
                cond_names = None
        if cond_names is None:
            cond_series = pd.Series(y, dtype=int)    # numeric labels when names not available
        else:
            cond_series = pd.Series([cond_names[idx] for idx in y], dtype="object")
        return cond_series, pd.Series(state_labels.astype(int), dtype=int)

    # If nothing matched, raise a clear error
    raise RuntimeError("Could not derive per-window condition names from the provided stack.")


# ------------------------------- Plotters -------------------------------- #

def plot_we_by_condition(df: pd.DataFrame, out_path: str, cond_col: str) -> None:
    """Plot WE by condition as a violin + jitter; each dot is one subject×condition."""
    plt.figure(figsize=(9, 6))                       # set figure size
    ax = plt.gca()                                   # current axis
    sns.violinplot(data=df, x=cond_col, y="WE", inner=None, cut=0)  # smooth violin outlines
    sns.stripplot(data=df, x=cond_col, y="WE", dodge=False, size=4, alpha=0.7)  # dots per row
    # Add a small horizontal line for the mean per condition
    grp = df.groupby(cond_col)["WE"].mean()
    for i, m in enumerate(grp):
        ax.hlines(m, i - 0.28, i + 0.28, lw=2)
    ax.set_xlabel("")                                 # cleaner x-label; condition names already on ticks
    ax.set_ylabel("WE")                               # keep notation used elsewhere
    ax.set_title("WE by condition")
    # Explanatory text box describing what the figure shows
    annotate(ax, "Each dot = one subject×condition (equal weight per subject).\n"
                 "WE = Σ_i p_i · H_i; higher values indicate states with higher centroid entropy weighted by occupancy.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hocc_by_condition(df: pd.DataFrame, out_path: str, cond_col: str) -> None:
    """Plot occupancy entropy H(p) by condition as a violin + jitter."""
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    # Prefer normalized if present; fallback to raw H_occ
    ycol = "H_occ_norm" if "H_occ_norm" in df.columns else "H_occ"
    label = "H(p) normalized" if ycol == "H_occ_norm" else "H(p)"
    sns.violinplot(data=df, x=cond_col, y=ycol, inner=None, cut=0)
    sns.stripplot(data=df, x=cond_col, y=ycol, dodge=False, size=4, alpha=0.7)
    grp = df.groupby(cond_col)[ycol].mean()
    for i, m in enumerate(grp):
        ax.hlines(m, i - 0.28, i + 0.28, lw=2)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.set_title("Occupancy entropy by condition")
    annotate(ax, "H(p) = −Σ_i p_i log p_i computed per subject×condition.\n"
                 "Each dot = one subject×condition; equal weight per subject.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mean_occupancy_heatmap(df: pd.DataFrame, out_path: str, cond_col: str) -> None:
    """Plot mean p_i per condition as a heatmap (averaged across subjects; equal weight per row)."""
    # Extract all p_i columns in order
    p_cols = [c for c in df.columns if c.startswith("p_")]
    plot_df = (df.groupby(cond_col)[p_cols]
                 .mean(numeric_only=True)
                 .reindex(sorted(df[cond_col].unique(), key=lambda x: str(x))))  # stable order
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    sns.heatmap(plot_df, annot=False, cmap="viridis", cbar_kws={"label": "Mean occupancy p_i"})
    ax.set_xlabel("State components (p_i)")
    ax.set_ylabel("Condition")
    ax.set_title("Mean occupancy by condition")
    annotate(ax, "Mean of p_i across subject×condition rows (equal weight per subject).\n"
                 "Rows = conditions, columns = states p_0..p_{K−1}.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_state_entropy_bar(state_entropies_npy: str, out_path: str) -> None:
    """Bar plot of state entropies H_i sorted high→low."""
    H = np.load(state_entropies_npy).astype(float)   # array length K
    order = np.argsort(H)[::-1]                      # descending order
    H_sorted = H[order]
    names = [f"state {i}" for i in order]            # label with original indices

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    sns.barplot(x=names, y=H_sorted)
    ax.set_xlabel("State index (sorted by H_i)")
    ax.set_ylabel("State entropy H_i")
    ax.set_title("State entropies (sorted high→low)")
    annotate(ax, "H_i = entropy of centroid/medoid connectivity values for state i.\n"
                 "Sorted by H_i to highlight the most complex state patterns.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_counts_state_condition(stack_npz: str, labels_npy: str, out_path: str) -> None:
    """QC heatmap: raw window counts per condition×state."""
    cond_s, labels_s = load_stack_and_labels(stack_npz, labels_npy)  # per-window labels
    # Build contingency table of raw counts
    table = (pd.DataFrame({"condition": cond_s, "state": labels_s})
               .groupby(["condition", "state"]).size().unstack(fill_value=0))
    # Sorted display order for conditions for stability
    table = table.reindex(index=sorted(table.index, key=lambda x: str(x)))
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    sns.heatmap(table, annot=False, cmap="magma", cbar_kws={"label": "# windows"})
    ax.set_xlabel("State")
    ax.set_ylabel("Condition")
    ax.set_title("Counts of windows by state × condition (QC)")
    annotate(ax, "QC figure — raw window counts.\n"
                 "Not suitable for inference when datasets are unbalanced.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_counts_state_condition_rowpct(stack_npz: str, labels_npy: str, out_path: str) -> None:
    """QC (supplement): row-normalized percentages per condition×state."""
    cond_s, labels_s = load_stack_and_labels(stack_npz, labels_npy)  # per-window labels
    # Build counts table
    table = (pd.DataFrame({"condition": cond_s, "state": labels_s})
               .groupby(["condition", "state"]).size().unstack(fill_value=0))
    table = table.reindex(index=sorted(table.index, key=lambda x: str(x)))
    # Row-normalize to percentages
    rowpct = table.div(table.sum(axis=1), axis=0).fillna(0.0) * 100.0
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    sns.heatmap(rowpct, annot=False, cmap="viridis", cbar_kws={"label": "% of windows per condition"})
    ax.set_xlabel("State")
    ax.set_ylabel("Condition")
    ax.set_title("Counts (row-normalized) by state × condition (QC — supplement)")
    annotate(ax, "Supplementary plot to the raw counts heatmap.\n"
                 "Each row is normalized to 100%; reveals pattern independent of window totals.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_we_vs_hocc(df: pd.DataFrame, out_path: str, cond_col: str) -> None:
    """Scatter of WE vs H(p) normalized, colored by condition."""
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ycol = "H_occ_norm" if "H_occ_norm" in df.columns else "H_occ"
    label = "H(p) normalized" if ycol == "H_occ_norm" else "H(p)"
    sns.scatterplot(data=df, x=ycol, y="WE", hue=cond_col, s=40, alpha=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("WE")
    ax.set_title("WE vs H(p)")
    annotate(ax, "Each dot = one subject×condition.\n"
                 "Shows relation between occupancy entropy H(p) and weighted entropy WE.")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------- Main ---------------------------------- #

def main() -> int:
    """Parse CLI, load data, and render all figures with embedded explanations."""
    ap = argparse.ArgumentParser(description="Make figures from state features/labels.")
    ap.add_argument("--features", required=True, help="CSV from 06 (e.g., features_subject_condition_k5.csv)")
    ap.add_argument("--state-entropies", required=True, help="*.npy with H_i from 06 (state_entropies_k5.npy)")
    ap.add_argument("--stack", default=None, help="NPZ stack with per-window condition info (for QC counts)")
    ap.add_argument("--state-labels", default=None, help="*.npy per-window state labels (for QC counts)")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures")
    args = ap.parse_args()

    # Prepare output directory
    ensure_outdir(args.out_dir)

    # Load features from 06
    feat_df = load_features_csv(args.features)
    cond_col = pick_condition_column(feat_df)

    # Use a readable seaborn theme
    sns.set_context("talk", font_scale=0.9)          # modest scaling for readability
    sns.set_style("whitegrid")                       # clean background helpful for scientific plots

    # 1) WE by condition
    plot_we_by_condition(feat_df, os.path.join(args.out_dir, "we_by_condition.png"), cond_col)

    # 2) H(p) by condition
    plot_hocc_by_condition(feat_df, os.path.join(args.out_dir, "Hocc_by_condition.png"), cond_col)

    # 3) Mean occupancy heatmap
    plot_mean_occupancy_heatmap(feat_df, os.path.join(args.out_dir, "occupancy_heatmap.png"), cond_col)

    # 4) State entropy bar (H_i)
    plot_state_entropy_bar(args.state_entropies, os.path.join(args.out_dir, "state_entropy_bar.png"))

    # 5) QC counts by state×condition (raw) — only if stack + state-labels are provided
    if args.stack and args.state_labels:
        plot_counts_state_condition(args.stack, args.state_labels,
                                    os.path.join(args.out_dir, "counts_state_condition.png"))
        # 6) QC supplement: row-normalized version (%)
        plot_counts_state_condition_rowpct(args.stack, args.state_labels,
                                           os.path.join(args.out_dir, "counts_state_condition_rowpct.png"))
    else:
        print("[info] Skipping QC heatmaps: both --stack and --state-labels are required.")

    # 7) Scatter: WE vs H(p)
    plot_we_vs_hocc(feat_df, os.path.join(args.out_dir, "WE_vs_Hocc.png"), cond_col)

    print(f"[done] Saved plots in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
