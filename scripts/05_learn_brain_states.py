#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learn_brain_states.py
---------------------
Unsupervised learning of EEG "brain states" from vectorized wSMI windows.

Design choices
- Algorithms: K-Means or K-Medoids selectable via --algo.
- Metric: default Manhattan; choose Euclidean with --metric euclidean.
- When running `--algo kmeans --metric manhattan`, a k-medians (L1) solver is used
  under the hood, so the prototypes are true L1-centers but exposed as "k-means"
  prototypes for downstream code.
- Elbow selection of k via Kneedle (auto) or manual selection (select) or fixed k.
- Balanced subsampling by subject*condition to avoid domination by frequent conditions.
- Saves prototypes (centroids/medoids), fitted model, metadata, and an elbow plot.

Example:
python scripts\05_learn_brain_states.py `
  --stack "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd_k3_tau8ms_stack.npz" `
  --out-dir "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\states_features_k_means" `
  --k auto `
  --k-range 2 10 `
  --algo kmeans `
  --metric manhattan `
  --cap-per-subject-per-cond 300 `
  --n-init 1000 `
  --random-state 42

Example 2:
python scripts\05_learn_brain_states.py `
  --stack "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd_k3_tau8ms_stack.npz" `
  --out-dir "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\states_features_k_means" `
  --k auto `
  --k-range 2 10 `
  --algo kmeans `
  --metric manhattan `
  --n-init 1000 `

Dependencies:
pip install numpy matplotlib joblib scikit-learn kneed tqdm
Optional (only if --algo kmedoids):
pip install scikit-learn-extra
"""
import os  # Environment variables must be set before numpy/sklearn are imported

# Force one BLAS/OpenMP thread per worker process to avoid oversubscription.
# This keeps CPU usage efficient when joblib runs many parallel restarts.
os.environ.setdefault("OMP_NUM_THREADS", "1")       # OpenMP-backed libraries
os.environ.setdefault("MKL_NUM_THREADS", "1")       # Intel MKL
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # OpenBLAS
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")   # numexpr (if used)

import argparse  # Command-line argument parsing
import json      # JSON metadata writing
import sys       # Exit codes
import time      # Timestamp for metadata

import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm                              # Progress bar

from sklearn.utils import check_random_state       # Reproducible RNG
from sklearn.cluster import KMeans                 # K-Means implementation
from sklearn.metrics import pairwise_distances     # Distance computations
from joblib import Parallel, delayed               # Parallel restarts
from kneed import KneeLocator                      # Kneedle elbow detection


# -------------------------
# Utilities to load the stacked dataset
# -------------------------

def _get(arr, keys, default=None):
    """Return the first existing key from a loaded .npz dict-like object."""
    for k in keys:
        if k in arr:
            return arr[k]
    return default


def load_stack(path):
    """Load the stack .npz file and return (X, subjects, conditions).

    X has shape (N, P), where N is number of windows and P number of features.
    subjects and conditions are 1D arrays aligned with the rows of X.
    """
    data = np.load(path, allow_pickle=True)  # load npz archive
    X = _get(data, ["X", "features"])       # feature matrix (N, P)
    if X is None:
        raise ValueError("Feature matrix 'X' not found in stack file.")
    subjects = _get(
        data,
        ["groups", "subjects", "subject", "subj", "subj_ids"],
    )  # subject IDs
    if subjects is None:
        subjects = np.arange(X.shape[0])  # simple fallback ids
    conditions = _get(
        data,
        ["y_cond", "conditions", "cond", "y", "labels", "event_labels", "events"],
    )  # condition/event labels
    if conditions is None:
        conditions = np.zeros(X.shape[0], dtype=int)  # single neutral condition
    subjects = np.asarray(subjects)
    conditions = np.asarray(conditions)
    return X, subjects, conditions


def index_by_subject_condition(subjects, conditions):
    """Create an index list for each (subject, condition) pair.

    Returns a dict mapping (subject, condition) -> list of row indices in X.
    """
    pairs = {}  # map (s, c) -> list of idx
    for i, (s, c) in enumerate(zip(subjects, conditions)):
        key = (s, c)                  # subject×condition key
        if key not in pairs:
            pairs[key] = []           # create new list if first occurrence
        pairs[key].append(i)          # append current index
    return pairs


def balanced_cap_indices(pairs, cap, random_state):
    """Subsample up to 'cap' windows per (subject, condition) pair.

    Returns a sorted 1D numpy array of indices that defines the training subset.
    """
    if cap is None or cap <= 0:
        # No cap: use all indices concatenated across all pairs
        all_idx = np.concatenate([np.asarray(v, int) for v in pairs.values()])
        return np.sort(all_idx)
    rng = check_random_state(random_state)  # RNG instance
    idx_keep = []                           # list of arrays
    for v in pairs.values():
        v = np.asarray(v, int)
        if v.size <= cap:
            # Small bucket: keep all indices
            idx_keep.append(v)
        else:
            # Large bucket: sample without replacement
            idx_keep.append(rng.choice(v, size=cap, replace=False))
    idx_keep = np.concatenate(idx_keep)
    return np.sort(idx_keep)                # sort for reproducibility


# -------------------------
# K-medians helper (used for k-means + Manhattan)
# -------------------------

class _KMediansModel:
    """Minimal model wrapper to mimic sklearn's KMeans attributes.

    Stores cluster_centers_, labels_, inertia_, and random_state.
    """
    def __init__(self, centers, labels, inertia, random_state=None):
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.random_state = random_state


def _run_kmedians(X, k, seed=0, max_iter=300):
    """Simple k-medians (L1) clustering with random initialization.

    - Distances: Manhattan (L1) via sklearn.metrics.pairwise_distances.
    - Assignment: each sample is assigned to its nearest center.
    - Update: new center is the coordinate-wise median of cluster samples.
    - Empty clusters: reinitialized to farthest points from any center.
    """
    rng = check_random_state(seed)
    n_samples, n_features = X.shape

    if k <= 0 or k > n_samples:
        raise ValueError("k must be in [1, n_samples] for k-medians.")

    # Randomly pick distinct initial centers
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centers = X[init_idx].astype(float, copy=True)

    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):
        # Compute Manhattan distances from all points to all centers
        dist = pairwise_distances(X, centers, metric="manhattan")
        # Assign each sample to the closest center
        labels = np.argmin(dist, axis=1)

        new_centers = np.zeros_like(centers)
        # Distance to closest center for each sample (used for empty clusters)
        min_dist = dist[np.arange(n_samples), labels]

        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # Empty cluster: reinitialize to farthest sample overall
                idx_far = np.argmax(min_dist)
                new_centers[j] = X[idx_far]
            else:
                # Coordinate-wise median for all points in this cluster
                new_centers[j] = np.median(X[mask], axis=0)

        # Check for convergence (centers stable)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break

        centers = new_centers

    # Final inertia with Manhattan distances
    dist_final = pairwise_distances(X, centers, metric="manhattan")
    labels = np.argmin(dist_final, axis=1)
    inertia = dist_final[np.arange(n_samples), labels].sum()

    return _KMediansModel(centers=centers, labels=labels, inertia=inertia, random_state=seed)


# -------------------------
# Core model fitting and inertia curve
# -------------------------

def fit_best_model(X, k, n_init, random_state, n_jobs, algo="kmeans", metric="manhattan"):
    """Run multiple restarts and keep the model with lowest inertia.

    Parameters
    ----------
    X : ndarray, shape (N, P)
        Feature matrix of all windows used for training.
    k : int
        Number of clusters (brain states).
    n_init : int
        Number of random restarts.
    random_state : int or None
        Base seed for reproducibility.
    n_jobs : int
        Number of parallel workers for joblib.
    algo : {"kmeans", "kmedoids"}
        Requested algorithm family.
    metric : {"euclidean", "manhattan"}
        Distance metric. For algo="kmeans" and metric="manhattan", a k-medians
        (L1) solver is used internally.

    Returns
    -------
    best_model : estimator-like
        Fitted model with attributes cluster_centers_, labels_, inertia_.
    best_inertia : float
        Lowest inertia across restarts.
    algo_used : str
        Effective algo identifier: "kmeans", "kmeans_l1", or "kmedoids".
    """
    algo = str(algo).lower()
    metric = str(metric).lower()

    if algo not in ("kmeans", "kmedoids"):
        raise ValueError(f"Unknown algo: {algo}")
    if metric not in ("euclidean", "manhattan"):
        raise ValueError(f"Unknown metric: {metric}")

    algo_used = algo  # kmeans or kmedoids

    def _fit_once(seed):
        """Fit a single restart for the chosen algorithm/metric."""
        if algo_used == "kmeans" and metric == "euclidean":
            # Classical K-Means with Euclidean distance
            mdl = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=1,        # restarts handled externally
                max_iter=300,
                random_state=seed,
                verbose=0,
            ).fit(X)
        elif algo_used == "kmeans":
            # L1-version: k-medians with Manhattan distance
            mdl = _run_kmedians(X, k, seed=seed, max_iter=300)
        else:  # algo_used == "kmedoids"
            # Lazy import to avoid hard dependency unless needed
            from sklearn_extra.cluster import KMedoids
            mdl = KMedoids(
                n_clusters=k,
                metric=metric,
                init="random",
                max_iter=300, # max iterations to make sure that it converges, if not it will stop after 300 iterations
                random_state=seed,
            ).fit(X)
        inertia = getattr(mdl, "inertia_", np.inf)
        return inertia, mdl

    # Build seeds for all restarts
    base = 0 if random_state is None else int(random_state)
    seeds = [base + r for r in range(int(n_init))]

    # Parallel runs of _fit_once with progress bar
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_once)(s) for s in tqdm(seeds, desc=f"Fitting k={k}", leave=False)
    )

    # Select model with minimum inertia
    best_inertia, best_model = min(results, key=lambda t: t[0])

    # Encode which variant was actually used
    if algo_used == "kmeans" and metric != "euclidean":
        algo_flag = "kmeans_l1"  # k-medians
    else:
        algo_flag = algo_used

    return best_model, float(best_inertia), algo_flag


def inertia_curve(X, k_min, k_max, n_init, random_state, n_jobs, algo="kmeans", metric="manhattan"):
    """Compute inertia for each k in [k_min, k_max] for the chosen algorithm/metric.

    Returns (ks, ws, algo_used), where ks is an array of k-values, ws is the
    corresponding inertia values, and algo_used is the effective algorithm flag.
    """
    ks = np.arange(int(k_min), int(k_max) + 1, dtype=int)
    ws = []
    algo_used_final = None

    for k in tqdm(ks, desc="Computing elbow curve"):
        model_k, inertia_k, algo_used_k = fit_best_model(
            X=X,
            k=int(k),
            n_init=n_init,
            random_state=random_state,
            n_jobs=n_jobs,
            algo=algo,
            metric=metric,
        )
        ws.append(inertia_k)
        algo_used_final = algo_used_k  # same type across ks

    return ks, np.asarray(ws, float), algo_used_final


def pick_k_via_kneedle(ks, ws):
    """Select k using the Kneedle algorithm on the inertia curve.

    Falls back to the global minimum of inertia if Kneedle does not find a knee.
    """
    ks = np.asarray(ks, float)
    ws = np.asarray(ws, float)

    try:
        kn = KneeLocator(
            ks,
            ws,
            curve="convex",
            direction="decreasing",
        )
        k_star = kn.knee
    except Exception:
        k_star = None

    if k_star is None:
        # Fallback: choose k with smallest inertia
        idx = int(np.argmin(ws))
        k_star = int(ks[idx])
    else:
        k_star = int(round(k_star))

    return k_star


def save_elbow_plot(path, ks, ws, k_star, algo, metric):
    """Save an elbow plot with the selected k highlighted."""
    plt.figure(figsize=(6, 4))
    plt.plot(ks, ws, "o-", linewidth=1.5)
    plt.axvline(k_star, linestyle="--", label=f"selected k = {k_star}")
    plt.xlabel("Number of states (K)")
    plt.ylabel("Inertia")
    plt.title(f"Elbow curve ({algo}, metric={metric})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------------
# CLI
# -------------------------

def main():
    """Command-line entry point for learning wSMI-based brain states."""
    ap = argparse.ArgumentParser(
        description="Learn wSMI-based brain states via K-Means/K-Medoids."
    )
    ap.add_argument(
        "--stack",
        required=True,
        type=str,
        help="Input stack .npz produced by 04_stack_dataset.py.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory for prototypes, model, elbow plot, and metadata.",
    )
    ap.add_argument(
        "--k",
        default="auto",
        help="Number of states K: integer, 'auto' for Kneedle, or 'select' for interactive choice.",
    )
    ap.add_argument(
        "--k-range",
        nargs=2,
        type=int,
        default=[2, 10],
        help="Range [k_min k_max] for elbow search when k is 'auto' or 'select'.",
    )
    ap.add_argument(
        "--algo",
        choices=["kmeans", "kmedoids"],
        default="kmeans",
        help="Clustering algorithm: kmeans (Euclidean or L1 via k-medians) or kmedoids.",
    )
    ap.add_argument(
        "--metric",
        choices=["euclidean", "manhattan"],
        default="manhattan",
        help="Distance metric. For kmeans+manhattan, a k-medians (L1) solver is used internally.",
    )
    ap.add_argument(
        "--cap-per-subject-per-cond",
        type=int,
        default=300,
        help="Maximum number of windows per (subject, condition) used for training.",
    )
    ap.add_argument(
        "--n-init",
        type=int,
        default=1000,
        help="Number of random restarts per k.",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for joblib (use -1 for all cores).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  # ensure output directory exists

    # -------------------------
    # Load dataset and select balanced training subset
    # -------------------------
    print(f"[load] stack: {args.stack}")
    X_all, subjects, conditions = load_stack(args.stack)
    print(f"[load] X_all shape: {X_all.shape}, N={X_all.shape[0]}, P={X_all.shape[1]}")

    pairs = index_by_subject_condition(subjects, conditions)
    idx_train = balanced_cap_indices(pairs, args.cap_per_subject_per_cond, args.random_state)
    X_tr = X_all[idx_train]
    print(
        f"[train] using {X_tr.shape[0]} windows for clustering "
        f"(cap-per-subject-per-cond={args.cap_per_subject_per_cond})"
    )

    # -------------------------
    # Inertia curve and elbow selection
    # -------------------------
    k_min, k_max = map(int, args.k_range)
    ks, ws, algo_used_for_sweep = inertia_curve(
        X_tr,
        k_min,
        k_max,
        args.n_init,
        args.random_state,
        args.n_jobs,
        algo=args.algo,
        metric=args.metric,
    )

    plot_path = os.path.join(args.out_dir, "elbow.png")

    if str(args.k).lower() == "select":
        # Suggest k via Kneedle, allow manual override
        k_auto = pick_k_via_kneedle(ks, ws)
        save_elbow_plot(plot_path, ks, ws, k_auto, algo=algo_used_for_sweep, metric=args.metric)
        print(f"[elbow] Plot saved → {plot_path}. Suggested k={k_auto}.")
        try:
            k_star = int(input(f"Enter k (press Enter for {k_auto}): ") or k_auto)
        except Exception:
            k_star = k_auto
    elif str(args.k).lower() == "auto":
        # Fully automatic Kneedle selection
        k_star = pick_k_via_kneedle(ks, ws)
        save_elbow_plot(plot_path, ks, ws, k_star, algo=algo_used_for_sweep, metric=args.metric)
        print(f"[elbow] Auto-selected k={k_star}. Plot saved → {plot_path}")
    else:
        # Fixed user-specified k
        k_star = int(args.k)
        save_elbow_plot(plot_path, ks, ws, k_star, algo=algo_used_for_sweep, metric=args.metric)
        print(f"[elbow] Using user-specified k={k_star}. Plot saved → {plot_path}")

    # -------------------------
    # Final fit at chosen k on the same training subset
    # -------------------------
    best_model, best_inertia, algo_used_final = fit_best_model(
        X_tr,
        k_star,
        args.n_init,
        args.random_state,
        args.n_jobs,
        algo=args.algo,
        metric=args.metric,
    )

    # Note: When requesting K-Means with Manhattan, a k-medians (L1) solver is used internally.
    #  - this is encoded via algo_used_final == "kmeans_l1".

    # -------------------------
    # Save prototypes and model depending on algorithm
    # -------------------------
    if algo_used_final in ("kmeans", "kmeans_l1"):
        # Centroids from KMeans or centers from k-medians
        prototypes = best_model.cluster_centers_          # shape (k, P)
        metric = "euclidean" if algo_used_final == "kmeans" else "manhattan"
        model_name = (
            "KMeans (scikit-learn)" if algo_used_final == "kmeans" else "K-Medians (L1)"
        )
        algo_tag = "kmeans"

        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}_prototypes.npy"),
            prototypes,
        )
        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_train_indices.npy"),
            idx_train,
        )
        joblib.dump(
            best_model,
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}.joblib"),
        )
        extra_meta = {}
    else:
        # K-Medoids: prototypes are actual training windows
        from sklearn_extra.cluster import KMedoids  # noqa: F401  (ensures dependency noted)

        medoid_idx = best_model.medoid_indices_     # shape (k,)
        prototypes = X_tr[medoid_idx]
        metric = args.metric
        model_name = "KMedoids (scikit-learn-extra)"
        algo_tag = "kmedoids"

        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}_prototypes.npy"),
            prototypes,
        )
        # Backward-compatibility filename used earlier in the project
        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_manhattan_medoids.npy"),
            prototypes,
        )
        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_train_medoid_idx.npy"),
            medoid_idx,
        )
        np.save(
            os.path.join(args.out_dir, f"states_k{k_star}_train_indices.npy"),
            idx_train,
        )
        joblib.dump(
            best_model,
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}.joblib"),
        )
        extra_meta = {
            "train_medoid_idx_path": os.path.abspath(
                os.path.join(args.out_dir, f"states_k{k_star}_train_medoid_idx.npy")
            )
        }

    # -------------------------
    # Metadata for reproducibility
    # -------------------------
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
        "algo_requested": args.algo,
        "algo_used": algo_used_final,
        "metric": metric,
        "model": model_name,
        "best_inertia": float(best_inertia),
        "inertia_curve": {int(k): float(w) for k, w in zip(ks, ws)},
        "elbow_plot": os.path.abspath(plot_path),
        "notes": (
            "Training subset selected by subject×condition cap; "
            "all windows are reassigned to the learned states in step 06."
        ),
        "prototypes_path": os.path.abspath(
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}_prototypes.npy")
        ),
        "model_path": os.path.abspath(
            os.path.join(args.out_dir, f"states_k{k_star}_{algo_tag}.joblib")
        ),
        **extra_meta,
    }

    meta_path = os.path.join(args.out_dir, f"states_k{int(k_star)}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[done] Saved prototypes/model/metadata for k={k_star} "
        f"(algo_requested={args.algo}, algo_used={algo_used_final}) "
        f"in {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())