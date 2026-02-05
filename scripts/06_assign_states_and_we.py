#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign each window to the nearest K-Medoids state (Manhattan) and compute:
- Occupancy p_i per subject*condition
- State entropies H_i (pooled windows histogram by default)
- WE = sum_i p_i * H_i
- Occupancy entropy H(p) and normalized H(p)
- Robust condition names derived from stack:
  * per-window names:  event_labels_all (preferred)
  * numeric codes:     y_cond (preferred among numeric keys)
  * global name list:  conditions (list[str]) or similar

Optional override: --cond-map "0=Resting,1=Familiar voice,2=Medical staff"
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

# -------------------------
# small helpers
# -------------------------

def is_numeric_vector(x, n=None):
    """Return True if x is a 1D numeric array (optionally with length n)."""
    x = np.asarray(x)
    ok_dtype = x.dtype.kind in ("i", "u", "f")
    ok_len = (n is None) or (x.ndim == 1 and x.shape[0] == n)
    return ok_dtype and ok_len and x.ndim == 1

def _first_key(d, keys):
    """Return first present key from a numpy .npz-like dict."""
    for k in keys:
        if k in d:
            return k
    return None

def load_stack(path):
    """Load stack .npz and extract features + any condition metadata present."""
    data = np.load(path, allow_pickle=True)

    # Feature matrix (N, P)
    X = data[_first_key(data, ["X", "features"])]
    X = np.asarray(X)
    N = X.shape[0]

    # Subjects (N,) or fallback 0..N-1
    subj_key = _first_key(data, ["groups", "subjects", "subject", "subj_ids"])
    subjects = np.asarray(data[subj_key]) if subj_key else np.arange(N)

    # Per-window string labels (preferred if available)
    lbl_key = _first_key(data, ["event_labels_all", "event_labels", "events", "labels"])
    event_labels = np.asarray(data[lbl_key]) if lbl_key is not None else None
    if event_labels is not None and event_labels.shape[0] != N:
        event_labels = None  # guard if shape does not match windows

    # Numeric condition codes (prefer y_cond; never confuse with 'conditions' list[str])
    num_keys = ["y_cond", "cond", "y"]
    cond_codes = None
    for k in num_keys:
        if k in data and is_numeric_vector(data[k], n=N):
            cond_codes = np.asarray(data[k]).astype(int)
            break

    # Global list of condition names (list[str] with length ~= #unique codes)
    name_list_key = _first_key(data, ["conditions", "condition_names", "cond_names", "event_names"])
    cond_name_list = None
    if name_list_key is not None:
        arr = np.asarray(data[name_list_key])
        if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
            cond_name_list = [str(s) for s in arr.tolist()]

    return X, subjects, cond_codes, event_labels, cond_name_list

def load_medoids(path):
    """Load medoids (K, P) from .npy."""
    M = np.load(path)
    if M.ndim != 2:
        raise ValueError("Medoids array must be 2D with shape (K, P).")
    return M, int(M.shape[0])

def parse_cond_map(s):
    """Parse '0=Resting,1=Familiar voice,2=Medical staff' into {0:'Resting',...}."""
    mapping = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("=", 1)
        mapping[int(left.strip())] = right.strip()
    return mapping

# -------------------------
# condition-name logic
# -------------------------

def derive_condition_names(cond_codes, event_labels, cond_name_list, cond_map):
    """
    Build parallel vectors for each window:
      - cond_code: numeric code
      - cond_name: human-readable name
    Priority:
      1) explicit --cond-map if codes are available
      2) per-window names (event_labels_all) -> factorize to codes
      3) numeric codes + a global name list (map sorted unique codes -> names)
      4) numeric codes only -> synthesize names 'cond_<code>'
    """
    # 1) explicit mapping wins when codes exist
    if cond_map and cond_codes is not None:
        codes = np.asarray(cond_codes).astype(int)
        names = pd.Series(codes).map(lambda x: cond_map.get(int(x), f"cond_{int(x)}")).to_numpy()
        return codes, names.astype(object)

    # 2) per-window string labels present -> use names directly, factorize to codes
    if event_labels is not None:
        name_series = pd.Series(event_labels).astype("object")
        codes, _ = pd.factorize(name_series, sort=True)  # stable order by label
        return pd.Series(codes, dtype=int).to_numpy(), name_series.to_numpy()

    # 3) numeric codes + a global name list
    if cond_codes is not None and cond_name_list is not None:
        codes = np.asarray(cond_codes).astype(int)
        uniq_codes = list(np.sort(pd.unique(codes)))
        # Map codes by sorted order; if mismatch, user can pass --cond-map
        if len(cond_name_list) == len(uniq_codes):
            mapping = {c: str(nm) for c, nm in zip(uniq_codes, cond_name_list)}
            names = pd.Series(codes).map(mapping).astype("object").to_numpy()
            return codes, names

    # 4) numeric only -> give synthetic names
    if cond_codes is not None:
        codes = np.asarray(cond_codes).astype(int)
        names = pd.Series(codes).map(lambda x: f"cond_{int(x)}").astype("object").to_numpy()
        return codes, names

    # Fallback: all else fails -> single condition
    codes = np.zeros(len(event_labels) if event_labels is not None else 1, dtype=int)
    names = np.array(["cond_0"], dtype=object).repeat(codes.shape[0])
    return codes, names

# -------------------------
# core computations
# -------------------------

def assign_states(X, medoids):
    """Assign each row of X to the nearest medoid using Manhattan distance."""
    D = pairwise_distances(X, medoids, metric="manhattan")  # (N, K) L1 distances
    return D.argmin(axis=1).astype(int)                     # (N,) state ids

def state_entropies_from_hist(X, labels, K, bins=50):
    """Compute H_i from pooled windows per state using a global histogram range."""
    vmin, vmax = float(X.min()), float(X.max())
    if vmin == vmax:
        vmax = vmin + 1e-6                                  # safeguard for constant data
    H = np.zeros(K, float)
    for i in range(K):
        xi = X[labels == i]
        if xi.size == 0:
            H[i] = 0.0
            continue
        hist, _ = np.histogram(xi.ravel(), bins=bins, range=(vmin, vmax))
        p = hist.astype(float)
        H[i] = 0.0 if p.sum() == 0 else float(entropy(p / p.sum()))
    return H

def centroid_entropy_from_medoid(v):
    x = v.astype(float)
    P = x.size
    vmin, vmax = float(x.min()), float(x.max())
    if vmin == vmax:
        vmax = vmin + 1e-6
    bins = int(np.sqrt(P))
    hist, _ = np.histogram(x, bins=bins, range=(vmin, vmax))
    p = hist.astype(float)
    return 0.0 if p.sum() == 0 else float(entropy(p / p.sum()))


def compute_features(subjects, cond_code, labels, medoids, X, mode, bins):
    """Aggregate per subject×condition: p_i, WE, H(p); return table + H_i."""
    K = medoids.shape[0]
    df = pd.DataFrame({"subject": subjects, "condition": cond_code, "state": labels})

    # Counts per subject*condition*state
    counts = (
        df.groupby(["subject", "condition", "state"])
          .size().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
    )
    n_tot = counts.sum(axis=1).to_numpy()                   # total windows per row
    P_occ = (counts.div(n_tot, axis=0)).fillna(0.0)         # occupancy fractions

    # H_i per state
    if mode == "state-hist":
        H_vec = state_entropies_from_hist(X, labels, K, bins=bins)
    else:
        H_vec = np.array([centroid_entropy_from_medoid(medoids[i]) for i in range(K)], float)

    # WE and occupancy entropy
    WE = (P_occ.to_numpy() * H_vec[None, :]).sum(axis=1)    # WE = sum_i p_i * H_i
    P_mat = P_occ.to_numpy(copy=True)
    H_occ = entropy(P_mat, axis=1)                          # H(p)
    H_occ_norm = H_occ / np.log(K)                          # normalized H(p) ∈ [0,1]

    # Assemble tidy frame
    res = P_occ.copy()
    res.columns = [f"p_{i}" for i in range(K)]
    res["WE"] = WE
    res["H_occ"] = H_occ
    res["H_occ_norm"] = H_occ_norm
    res["n_windows"] = n_tot
    res = res.reset_index()
    return res, H_vec

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Assign states and compute occupancy/WE/H(p) with robust condition names.")
    ap.add_argument("--stack", required=True, type=str, help="Stack .npz (expects X, y_cond/event_labels_all, conditions).")
    ap.add_argument("--medoids", required=True, type=str, help="Medoids .npy from learn_brain_states.py.")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory.")
    ap.add_argument("--entropy-mode", choices=["state-hist", "medoid"], default="medoid",
                    help="How to compute H_i (default: medoid/centroid entropy).")
    ap.add_argument("--hist-bins", type=int, default=50, help="Histogram bins for state-hist.")
    ap.add_argument("--cond-map", type=str, default="",
                    help='Optional code->name map, e.g. "0=Resting,1=Familiar voice,2=Medical staff".')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data + medoids
    X, subjects, cond_codes, event_labels, cond_name_list = load_stack(args.stack)
    medoids, K = load_medoids(args.medoids)

    # Build per-window codes + names (order-agnostic and robust)
    cond_map = parse_cond_map(args.cond_map) if args.cond_map else None
    cond_code, cond_name = derive_condition_names(cond_codes, event_labels, cond_name_list, cond_map)

    # Assign and aggregate
    labels = assign_states(X, medoids)
    np.save(os.path.join(args.out_dir, f"state_labels_k{K}.npy"), labels)

    feat_df, H_vec = compute_features(
        subjects=subjects, cond_code=cond_code, labels=labels,
        medoids=medoids, X=X, mode=args.entropy_mode, bins=args.hist_bins,
    )

    # Add readable names at aggregated level via code->name mapping
    pairs = pd.DataFrame({"code": cond_code, "name": cond_name}).drop_duplicates()
    code2name = dict(zip(pairs["code"].astype(int), pairs["name"].astype(str)))
    feat_df["condition_name"] = feat_df["condition"].map(code2name)

    # Save tidy CSV + compact NPZ + H_i vector + meta
    feat_csv = os.path.join(args.out_dir, f"features_subject_condition_k{K}.csv")
    feat_npz = os.path.join(args.out_dir, f"features_subject_condition_k{K}.npz")
    feat_df.to_csv(feat_csv, index=False)
    np.save(os.path.join(args.out_dir, f"state_entropies_k{K}.npy"), H_vec)
    np.savez_compressed(
        feat_npz,
        subjects=feat_df["subject"].to_numpy(),
        conditions=feat_df["condition"].to_numpy(),
        condition_name=feat_df["condition_name"].to_numpy(),
        P=feat_df[[c for c in feat_df.columns if c.startswith("p_")]].to_numpy(),
        WE=feat_df["WE"].to_numpy(),
        H_occ=feat_df["H_occ"].to_numpy(),
        H_occ_norm=feat_df["H_occ_norm"].to_numpy(),
        n_windows=feat_df["n_windows"].to_numpy(),
        H=H_vec,
    )
    meta = {
        "stack": os.path.abspath(args.stack),
        "medoids": os.path.abspath(args.medoids),
        "out_dir": os.path.abspath(args.out_dir),
        "K": int(K), "N": int(X.shape[0]), "P": int(X.shape[1]),
        "entropy_mode": args.entropy_mode, "hist_bins": int(args.hist_bins),
        "condition_names_source": (
            "cond_map" if cond_map else
            ("event_labels_all" if event_labels is not None else
             ("conditions(list)" if cond_name_list is not None else
              ("y_cond" if cond_codes is not None else "fallback")))
        ),
    }
    with open(os.path.join(args.out_dir, f"features_k{K}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Saved labels, features (p_i, WE, H(p)), and state entropies ({args.entropy_mode}).")

if __name__ == "__main__":
    raise SystemExit(main())