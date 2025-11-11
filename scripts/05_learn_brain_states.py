#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learn_brain_states.py
---------------------
Unsupervised learning of EEG "brain states" from vectorized wSMI windows.

Design choices
- K-Medoids with Manhattan distance (robust L1 for high-dimensional connectivity).
- Elbow selection of k via Kneedle (auto) or manual selection (select) or fixed k.
- Balanced subsampling by subject×condition to avoid domination by frequent conditions.
- Saves medoids, fitted model, metadata, and an elbow plot for reproducibility.

Example:
python scripts/learn_brain_states.py `
  --stack "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd_k3_tau8ms_stack.npz" `
  --out-dir "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\states" `
  --k auto `
  --k-range 2 10 `
  --cap-per-subject-per-cond 3 `
  --n-init 1000 `
  --random-state 42

Dependencies:
pip install numpy matplotlib joblib scikit-learn-extra kneed
"""
import os  # env-vars must be set before numpy/sklearn are imported
# Force one BLAS/OpenMP thread per worker process to avoid oversubscription.
# This keeps CPU usage efficient when joblib runs many parallel restarts.
os.environ.setdefault("OMP_NUM_THREADS", "1")       # OpenMP-backed libs (general)
os.environ.setdefault("MKL_NUM_THREADS", "1")       # Intel MKL (NumPy/SciPy on MKL)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # OpenBLAS (NumPy/SciPy on OB)
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")   # numexpr (if used via pandas/etc.)

import argparse  # CLI parsing
import json      # save metadata as JSON
import sys       # exit codes
import time      # timestamp in metadata
import numpy as np
import joblib       # save/load sklearn objects
import matplotlib.pyplot as plt  # elbow plot
from sklearn.utils import check_random_state  # reproducible RNG
from sklearn_extra.cluster import KMedoids   # K-Medoids implementation
from kneed import KneeLocator                # Kneedle elbow detection
from joblib import Parallel, delayed        # parallel processing


# ---------- utilities ----------

def _get(arr, keys, default=None):
    """Pick the first existing key from a loaded .npz dict-like object."""
    for k in keys:
        if k in arr:
            return arr[k]
    return default


def load_stack(path):
    """Load the stack .npz and return (X, subjects, conditions)."""
    data = np.load(path, allow_pickle=True)                   # load npz
    X = _get(data, ["X", "features"])                         # feature matrix (N, P)
    if X is None:
        raise ValueError("Feature matrix 'X' not found in stack file.")  # fail fast
    subjects = _get(data, ["groups", "subjects", "subject", "subj_ids"]) # subject IDs
    if subjects is None:
        subjects = np.arange(X.shape[0])                      # fallback to simple ids
    conditions = _get(
        data,
        ["y_cond", "conditions", "cond", "y", "labels", "event_labels", "events"],
    )                                                          # condition labels
    if conditions is None:
        conditions = np.zeros(X.shape[0], dtype=int)          # neutral single class
    subjects = np.asarray(subjects)                           # ensure ndarray
    conditions = np.asarray(conditions)                       # ensure ndarray
    return X, subjects, conditions


def index_by_subject_condition(subjects, conditions):
    """Create an index list for each (subject, condition) pair."""
    pairs = {}                                                # map (s,c) -> list of idx
    for i, (s, c) in enumerate(zip(subjects, conditions)):    # iterate rows
        key = (s, c)                                          # subject×condition key
        if key not in pairs:                                  # init list if first time
            pairs[key] = []
        pairs[key].append(i)                                  # collect index
    return pairs


def balanced_cap_indices(pairs, cap, random_state):
    """Subsample up to 'cap' per (subject, condition); return a sorted index array."""
    if cap is None or cap <= 0:                               # no cap -> use all
        all_idx = np.concatenate([np.asarray(v, int) for v in pairs.values()])
        return np.sort(all_idx)                               # stable order
    rng = check_random_state(random_state)                    # RNG
    idx_keep = []                                             # collected indices
    for v in pairs.values():                                  # loop each (s,c) bucket
        v = np.asarray(v, int)                                # as ndarray
        if v.size <= cap:                                     # small bucket -> keep all
            idx_keep.append(v)
        else:
            idx_keep.append(rng.choice(v, size=cap, replace=False))  # uniform sample
    idx_keep = np.concatenate(idx_keep)                       # concat selections
    return np.sort(idx_keep)                                  # sorted for reproducibility


def fit_best_kmedoids(X, k, n_init, random_state, n_jobs):
    """Parallel restarts; keep best-inertia model."""
    def _fit_once(seed):  # single restart
        mdl = KMedoids(n_clusters=k, metric="manhattan", # (manhattan = L1 distance) using manhattan just like the article Della Bella et al., 2025
                       init="random", max_iter=300,
                       random_state=seed).fit(X)
        return mdl.inertia_, mdl
    seeds = [(random_state or 0) + r for r in range(n_init)]  # reproducible seeds
    results = Parallel(n_jobs=n_jobs, backend="loky")(        # use all cores
        delayed(_fit_once)(s) for s in seeds
    )
    best_inertia, best_model = min(results, key=lambda t: t[0])  # pick best
    return best_model, float(best_inertia)



def inertia_curve(X, k_min, k_max, n_init, random_state, n_jobs):
    """Compute inertia (within-cluster L1 dissimilarity) for each k in range."""
    ks, ws = [], []                                           # lists for plot
    for k in range(k_min, k_max + 1):                         # iterate k values
        mdl, w = fit_best_kmedoids(X, k, n_init, random_state, n_jobs=n_jobs)  # best model this k
        ks.append(k)                                          # store k
        ws.append(w)                                          # store inertia
        print(f"[elbow] k={k:2d} inertia={w:.3f}")            # progress log
    return np.asarray(ks), np.asarray(ws)                     # arrays for downstream


def pick_k_via_kneedle(ks, ws):
    """Select k using Kneedle (convex & decreasing). Fallback to max-distance method."""
    try:
        kl = KneeLocator(ks, ws, curve="convex", direction="decreasing")  # detect knee
        if kl.knee is not None:                                           # success case
            return int(kl.knee)                                           # selected k
    except Exception:
        pass                                                               # robust fallback
    # fallback: point with maximum distance to the line between endpoints
    x1, y1 = ks[0], ws[0]                                                 # start point
    x2, y2 = ks[-1], ws[-1]                                               # end point
    num = np.abs((y2 - y1) * ks - (x2 - x1) * ws + (x2 * y1 - y2 * x1))   # area term
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)                        # line length
    d = num / (den + 1e-12)                                               # distances
    return int(ks[np.argmax(d)])                                          # argmax index


def save_elbow_plot(path, ks, ws, k_star):
    """Save elbow plot with the chosen k marked."""
    plt.figure(figsize=(6, 4))                                            # small figure
    plt.plot(ks, ws, marker="o")                                          # inertia curve
    plt.axvline(k_star, linestyle="--", alpha=0.7)                        # chosen k
    plt.title(f"Elbow (K-Medoids, Manhattan) — suggested k={k_star}")     # title
    plt.xlabel("k")                                                       # x-label
    plt.ylabel("Within-cluster dissimilarity (inertia)")                  # y-label
    plt.tight_layout()                                                    # tidy layout
    plt.savefig(path, dpi=150)                                            # write file
    plt.close()                                                           # free figure


# ---------- main ----------

def main():
    # CLI definition with minimal, readable flags
    ap = argparse.ArgumentParser(description="Learn unsupervised brain states (K-Medoids, Manhattan).")
    ap.add_argument("--stack", required=True, type=str, help="Path to stack .npz (contains X, subjects/groups, conditions).")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for model/medoids/metadata/plot.")
    ap.add_argument("--k", default="auto", help="'auto' | 'select' | <int> (e.g., 5).")
    ap.add_argument("--k-range", nargs=2, type=int, default=[2, 10], help="Range for k when using 'auto'/'select'.")
    ap.add_argument("--cap-per-subject-per-cond", type=int, default=0, help="Max windows per subject×condition for training (0 disables).")
    ap.add_argument("--n-init", type=int, default=64, help="Random restarts per k (set high, e.g., 1000 or 10000 for robustness).")
    ap.add_argument("--random-state", type=int, default=42, help="Base random seed.")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 = all cores).")
    args = ap.parse_args()                                                # parse flags

    os.makedirs(args.out_dir, exist_ok=True)                              # ensure output dir

    X_all, subjects, conditions = load_stack(args.stack)                  # load data
    print(f"[data] X_all={X_all.shape}, subjects={np.unique(subjects).size}, conditions={np.unique(conditions).size}")

    pairs = index_by_subject_condition(subjects, conditions)              # build (s,c) buckets
    idx_train = balanced_cap_indices(pairs, args.cap_per_subject_per_cond, args.random_state)  # balanced subsample
    X_tr = X_all[idx_train]                                              # training subset
    print(f"[train] using {X_tr.shape[0]} windows for clustering (cap-per-subject-per-cond={args.cap_per_subject_per_cond})")

    k_min, k_max = map(int, args.k_range)                                # unpack k-range
    ks, ws = inertia_curve(X_tr, k_min, k_max, args.n_init, args.random_state, args.n_jobs)  # elbow sweep

    # choose k according to mode
    plot_path = os.path.join(args.out_dir, "elbow.png")                  # plot destination
    if str(args.k).lower() == "select":                                  # interactive choice
        k_auto = pick_k_via_kneedle(ks, ws)                               # suggested k
        save_elbow_plot(plot_path, ks, ws, k_auto)                        # save elbow plot
        print(f"[elbow] Plot saved → {plot_path}. Suggested k={k_auto}.")  # info
        try:
            k_star = int(input(f"Enter k (press Enter for {k_auto}): ") or k_auto)  # user input
        except Exception:
            k_star = k_auto                                               # fallback to suggested
    elif str(args.k).lower() == "auto":                                   # automatic elbow
        k_star = pick_k_via_kneedle(ks, ws)                               # choose via Kneedle
        save_elbow_plot(plot_path, ks, ws, k_star)                        # save plot
        print(f"[elbow] Auto-selected k={k_star}. Plot saved → {plot_path}")  # log
    else:
        k_star = int(args.k)                                              # fixed k
        save_elbow_plot(plot_path, ks, ws, k_star)                        # still save plot
        print(f"[elbow] Using user-specified k={k_star}. Plot saved → {plot_path}")  # log

    # final fit at chosen k on the same training subset (multiple restarts)
    best_model, best_inertia = fit_best_kmedoids(X_tr, k_star, args.n_init, args.random_state, args.n_jobs)
    medoid_idx = best_model.medoid_indices_                               # medoid indices within X_tr
    medoids = X_tr[medoid_idx]                                            # medoid feature vectors

    # save artifacts for downstream reassignment (step 2)
    np.save(os.path.join(args.out_dir, f"states_k{k_star}_manhattan_medoids.npy"), medoids)       # (k, P)
    np.save(os.path.join(args.out_dir, f"states_k{k_star}_train_medoid_idx.npy"), medoid_idx)     # (k,)
    np.save(os.path.join(args.out_dir, f"states_k{k_star}_train_indices.npy"), idx_train)         # (N_train,)
    joblib.dump(best_model, os.path.join(args.out_dir, f"states_k{k_star}_kmedoids.joblib"))      # serialized model

    # write metadata for provenance
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stack": os.path.abspath(args.stack),
        "out_dir": os.path.abspath(args.out_dir),
        "k": int(k_star),
        "k_range": [int(k_min), int(k_max)],
        "cap_per_subject_per_cond": int(args.cap_per_subject_per_cond),
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "N_all": int(X_all.shape[0]),
        "N_train": int(X_tr.shape[0]),
        "P": int(X_all.shape[1]),
        "metric": "manhattan",
        "model": "KMedoids (scikit-learn-extra)",
        "best_inertia": float(best_inertia),
        "inertia_curve": {int(k): float(w) for k, w in zip(ks, ws)},
        "elbow_plot": os.path.abspath(plot_path),
        "notes": "Training subset selected by subject×condition cap; reassign ALL windows in step 2.",
    }
    with open(os.path.join(args.out_dir, f"states_k{int(k_star)}_meta.json"), "w") as f:  # open file
        json.dump(meta, f, indent=2)                                                     # dump JSON

    print(f"[done] Saved medoids/model/metadata for k={k_star} in {args.out_dir}")       # final log
    return 0                                                                             # OK exit code


if __name__ == "__main__":
    sys.exit(main())  # run CLI
