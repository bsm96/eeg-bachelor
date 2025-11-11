#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make figures from state features/labels:
- WE by condition
- Mean occupancy heatmap (p_i)
- H(p) by condition
- State entropy bar (H_i)
- QC heatmap: counts by state×condition (all windows)
- WE vs H(p)

Uses 'condition_name' from features CSV when present.
Falls back to names reconstructed from stack:
  * event_labels_all per window, or
  * y_cond mapped via 'conditions' name list.
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- utils ----------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_features(path_csv):
    """Read features CSV and detect occupancy columns and condition label column."""
    df = pd.read_csv(path_csv)
    p_cols = [c for c in df.columns if c.startswith("p_")]
    if not p_cols:
        raise ValueError("No occupancy columns found (expected 'p_0', 'p_1', ...).")
    K = len(p_cols)
    cond_col = "condition_name" if "condition_name" in df.columns else "condition"
    return df, p_cols, K, cond_col

def load_state_entropies(path_npy):
    """Load H_i vector (state entropies)."""
    H = np.load(path_npy)
    if H.ndim != 1:
        raise ValueError("state_entropies file must be a 1D vector.")
    return H

def load_stack_and_labels(stack_npz, labels_npy):
    """Return per-window condition labels (strings) and state labels (ints) for QC plot."""
    data = np.load(stack_npz, allow_pickle=True)
    N = None
    # Determine N from labels file
    state_labels = np.load(labels_npy)
    N = state_labels.shape[0]

    # Prefer per-window string labels if present
    for k in ["event_labels_all", "event_labels", "events", "labels"]:
        if k in data:
            s = np.asarray(data[k])
            if s.shape[0] == N:
                cond_series = pd.Series(s).astype("object")
                return cond_series, pd.Series(state_labels.astype(int))

    # Else use numeric codes and map to names if available
    cond_codes = None
    for k in ["y_cond", "cond", "y"]:
        if k in data:
            arr = np.asarray(data[k])
            if arr.ndim == 1 and arr.shape[0] == N and arr.dtype.kind in ("i","u","f"):
                cond_codes = arr.astype(int)
                break

    if cond_codes is None:
        raise ValueError("Conditions not found for QC (no event_labels_all and no y_cond).")

    # Global name list (conditions)
    cond_names = None
    for k in ["conditions", "condition_names", "cond_names", "event_names"]:
        if k in data:
            arr = np.asarray(data[k])
            if arr.ndim == 1 and arr.dtype.kind in ("U","S","O"):
                cond_names = [str(x) for x in arr.tolist()]
                break

    if cond_names is not None:
        uniq = sorted(pd.unique(cond_codes))
        if len(cond_names) == len(uniq):
            map_code2name = {c: nm for c, nm in zip(uniq, cond_names)}
            cond_series = pd.Series(cond_codes).map(map_code2name).astype("object")
        else:
            cond_series = pd.Series(cond_codes).map(lambda x: f"cond_{int(x)}").astype("object")
    else:
        cond_series = pd.Series(cond_codes).map(lambda x: f"cond_{int(x)}").astype("object")

    return cond_series, pd.Series(state_labels.astype(int))

def cond_order(series):
    """Stable order based on first occurrence."""
    return list(pd.Index(series).astype("object").unique())

def cmap_for_conditions(conditions):
    """Simple categorical colormap."""
    base = plt.get_cmap("tab10")
    uniq = list(pd.Index(conditions).astype("object").unique())
    return {c: base(i % 10) for i, c in enumerate(uniq)}

# ---------- plots ----------

def plot_we_by_condition(df, out_path, cond_col):
    order = cond_order(df[cond_col])
    colors = cmap_for_conditions(df[cond_col])
    data = [df.loc[df[cond_col] == c, "WE"].values for c in order]
    plt.figure(figsize=(6,4))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[order[i]]); b.set_alpha(0.5)
    for i, vals in enumerate(data, start=1):
        x = np.random.normal(i, 0.05, size=len(vals))
        plt.scatter(x, vals, s=12, alpha=0.7, color=colors[order[i-1]])
    plt.xticks(range(1, len(order)+1), order); plt.ylabel("WE"); plt.title("WE by condition")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_hocc_by_condition(df, out_path, cond_col):
    order = cond_order(df[cond_col])
    colors = cmap_for_conditions(df[cond_col])
    colh = "H_occ_norm" if "H_occ_norm" in df.columns else "H_occ"
    data = [df.loc[df[cond_col] == c, colh].values for c in order]
    plt.figure(figsize=(6,4))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[order[i]]); b.set_alpha(0.5)
    for i, vals in enumerate(data, start=1):
        x = np.random.normal(i, 0.05, size=len(vals))
        plt.scatter(x, vals, s=12, alpha=0.7, color=colors[order[i-1]])
    plt.xticks(range(1, len(order)+1), order)
    plt.ylabel("H(p) normalized" if colh == "H_occ_norm" else "H(p)")
    plt.title("Occupancy entropy by condition")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_occupancy_heatmap(df, p_cols, K, out_path, cond_col):
    mean_by_cond = df.groupby(cond_col)[p_cols].mean()
    order = list(mean_by_cond.index); M = mean_by_cond.values
    plt.figure(figsize=(1.5 + 0.8*K, 2 + 0.3*len(order)))
    im = plt.imshow(M, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean occupancy p_i")
    plt.yticks(range(len(order)), order); plt.xticks(range(K), [f"p_{i}" for i in range(K)])
    plt.title("Mean occupancy by condition")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_state_entropy_bar(H, out_path):
    order = np.argsort(H)[::-1]; Hs = H[order]; labels = [f"state {i}" for i in order]
    plt.figure(figsize=(0.8*len(Hs) + 2, 4))
    plt.bar(range(len(Hs)), Hs)
    plt.xticks(range(len(Hs)), labels); plt.ylabel("State entropy H_i")
    plt.title("State entropies (sorted high→low)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_counts_state_condition(stack_npz, labels_npy, out_path, cond_names_from_features):
    cond_s, labels_s = load_stack_and_labels(stack_npz, labels_npy)
    # Keep row order stable using first occurrence in cond_s
    table = pd.DataFrame({"condition": cond_s, "state": labels_s}) \
              .groupby(["condition","state"]).size().unstack(fill_value=0)
    order = list(table.index); K = table.shape[1]; M = table.values
    plt.figure(figsize=(1.5 + 0.8*K, 2 + 0.3*len(order)))
    im = plt.imshow(M, aspect="auto", cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="# windows")
    plt.yticks(range(len(order)), order); plt.xticks(range(K), [f"state {i}" for i in range(K)])
    plt.title("Counts of windows by state × condition")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_we_vs_hocc(df, out_path, cond_col):
    colors = cmap_for_conditions(df[cond_col])
    colh = "H_occ_norm" if "H_occ_norm" in df.columns else "H_occ"
    plt.figure(figsize=(6,4))
    for cond, sub in df.groupby(cond_col):
        plt.scatter(sub[colh], sub["WE"], s=18, alpha=0.7, label=str(cond), color=colors[cond])
    plt.xlabel("H(p) normalized" if colh == "H_occ_norm" else "H(p)"); plt.ylabel("WE")
    plt.title("WE vs H(p)"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Make plots from state features (auto-uses condition_name when present).")
    ap.add_argument("--features", required=True, type=str, help="features_subject_condition_kK.csv from 06.")
    ap.add_argument("--state-entropies", required=True, type=str, help="state_entropies_kK.npy from 06.")
    ap.add_argument("--stack", required=True, type=str, help="Stack .npz (for per-window QC).")
    ap.add_argument("--state-labels", required=True, type=str, help="state_labels_kK.npy from 06.")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for PNGs.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    feat_df, p_cols, K, cond_col = load_features(args.features)
    H = load_state_entropies(args.state_entropies)

    # 1) WE by condition
    plot_we_by_condition(feat_df, os.path.join(args.out_dir, "we_by_condition.png"), cond_col)

    # 2) Mean occupancy heatmap
    plot_occupancy_heatmap(feat_df, p_cols, K, os.path.join(args.out_dir, "occupancy_heatmap.png"), cond_col)

    # 3) H(p) by condition
    plot_hocc_by_condition(feat_df, os.path.join(args.out_dir, "Hocc_by_condition.png"), cond_col)

    # 4) State entropy bar
    plot_state_entropy_bar(H, os.path.join(args.out_dir, "state_entropy_bar.png"))

    # 5) QC counts by state×condition
    plot_counts_state_condition(
        args.stack, args.state_labels, os.path.join(args.out_dir, "counts_state_condition.png"),
        cond_names_from_features=None
    )

    # 6) Scatter: WE vs H(p)
    plot_we_vs_hocc(feat_df, os.path.join(args.out_dir, "WE_vs_Hocc.png"), cond_col)

    print(f"[done] Saved plots in {args.out_dir}")

if __name__ == "__main__":
    raise SystemExit(main())
