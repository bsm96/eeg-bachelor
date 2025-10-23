# src/eeg/wsmi/aggregation.py
from __future__ import annotations  # enable future-compatible annotations
from typing import List, Optional, Sequence, Tuple  # clear type hints for API stability
import numpy as np  # numerical arrays
from scipy import stats  # robust trimmed mean (SciPy is widely used alongside MNE in Engemann/C&K-style pipelines)  (flake8): # noqa: E402


def upper_triangle_values(
    M: np.ndarray,                      # square connectivity matrix (n_channels, n_channels) provided by caller
    *,
    include_diagonal: bool = False,     # caller decides whether to include diagonal entries
) -> np.ndarray:
    M = np.ascontiguousarray(M, dtype=float)  # ensure contiguous float array for reliable indexing
    if M.ndim != 2 or M.shape[0] != M.shape[1]:  # validate that the input is a square 2D matrix
        raise ValueError("Input must be a square 2D matrix.")  # explicit API error for malformed input
    k = 0 if include_diagonal else 1  # choose upper-triangle offset based on caller preference
    iu = np.triu_indices_from(M, k=k)  # compute indices for the requested upper triangle
    v = M[iu]  # gather upper-triangle values into a flat vector
    return v  # vector of pairwise values (possibly including diagonal if requested)


def reduce_pairs_epoch(
    W_epoch: np.ndarray,                # connectivity matrix for a single epoch provided by caller
    *,
    reducer: str = "median",            # pairwise reducer: "median" or "mean", chosen by caller
    include_diagonal: bool = False,     # whether to include diagonal entries in reduction
) -> float:
    v = upper_triangle_values(W_epoch, include_diagonal=include_diagonal)  # extract pairwise values once
    v = v[np.isfinite(v)]  # drop NaN/inf values to avoid bias in reducers
    if v.size == 0:  # guard against empty vector after filtering
        return float("nan")  # signal that this epoch produced no valid scalar
    if reducer == "median":  # branch for robust median across pairs
        return float(np.nanmedian(v))  # compute median while ignoring any residual NaNs
    if reducer == "mean":  # branch for arithmetic mean across pairs
        return float(np.nanmean(v))  # compute mean while ignoring any residual NaNs
    raise ValueError(f"Unknown reducer: {reducer!r}")  # explicit error for unsupported reducer names


def trimmed_mean(
    x: Sequence[float],                 # sequence of epoch-level scalars provided by caller
    *,
    proportion_to_cut: float,           # symmetric trim fraction set by caller (no hard-coded ratio here)
) -> float:
    arr = np.asarray(x, dtype=float)  # cast to float array for numerical ops
    arr = arr[np.isfinite(arr)]  # drop NaN/inf values before trimming
    if arr.size == 0:  # handle all-NaN or empty input gracefully
        return float("nan")  # no valid aggregate available
    if not (0.0 <= proportion_to_cut < 0.5):  # validate caller-specified trim fraction
        raise ValueError("proportion_to_cut must be in [0, 0.5).")  # enforce sensible trimming
    tm = stats.trim_mean(arr, proportiontocut=proportion_to_cut)  # SciPy’s robust trimmed mean implementation
    return float(tm)  # return scalar as Python float


def reduce_over_epochs(
    epoch_scalars: Sequence[float],     # one scalar per epoch (output of reduce_pairs_epoch)
    *,
    reducer: str = "mean",              # temporal reducer: "mean", "median", or "trimmed"
    proportion_to_cut: Optional[float] = None,  # trim fraction when reducer == "trimmed"
) -> float:
    vals = np.asarray(epoch_scalars, dtype=float)  # ensure float array for stable statistics
    vals = vals[np.isfinite(vals)]  # discard NaN/inf epoch values
    if vals.size == 0:  # guard when no valid epochs remain
        return float("nan")  # indicate no valid subject-level value
    if reducer == "mean":  # branch for arithmetic mean across epochs
        return float(np.nanmean(vals))  # compute mean ignoring NaNs
    if reducer == "median":  # branch for robust median across epochs
        return float(np.nanmedian(vals))  # compute median ignoring NaNs
    if reducer == "trimmed":  # branch for trimmed mean across epochs
        if proportion_to_cut is None:  # enforce that caller provided a trim fraction
            raise ValueError("proportion_to_cut must be provided when reducer='trimmed'.")  # clear API error
        return trimmed_mean(vals, proportion_to_cut=proportion_to_cut)  # delegate to SciPy-based trimmed mean
    raise ValueError(f"Unknown epoch reducer: {reducer!r}")  # explicit error for unsupported reducer names


def aggregate_wsmi_matrices(
    Ws: Sequence[np.ndarray],           # sequence of epoch-level matrices provided by caller
    *,
    pair_reducer: str = "median",       # reducer across pairs within an epoch ("median" or "mean")
    time_reducer: str = "mean",         # reducer across epochs ("mean", "median", or "trimmed")
    proportion_to_cut: Optional[float] = None,  # trim fraction for "trimmed" time reducer
    include_diagonal: bool = False,     # whether to include diagonal entries during pairwise reduction
) -> Tuple[List[float], float]:
    epoch_scalars: List[float] = []  # container for per-epoch scalar values
    for W_epoch in Ws:  # loop over all epoch matrices in sequence order
        s = reduce_pairs_epoch(  # reduce each epoch to one scalar according to the chosen pair reducer
            W_epoch, reducer=pair_reducer, include_diagonal=include_diagonal
        )
        epoch_scalars.append(s)  # collect the epoch scalar for subsequent temporal aggregation
    subject_scalar = reduce_over_epochs(  # reduce across epochs with the requested temporal reducer
        epoch_scalars, reducer=time_reducer, proportion_to_cut=proportion_to_cut
    )
    return epoch_scalars, subject_scalar  # return both the per-epoch series and the final subject-level scalar


def aggregator_from_string(
    name: str,                              # human-readable strategy name chosen by caller
    *,
    proportion_to_cut: Optional[float] = None,  # trim fraction when strategy implies trimming
) -> Tuple[str, str, Optional[float]]:
    name_norm = name.strip().lower()  # normalize the strategy name for robust matching
    if name_norm == "mean-mean":  # preset: mean over pairs, mean over epochs
        return "mean", "mean", None  # return explicit reducers; no trimming fraction needed
    if name_norm == "median-mean":  # preset: median over pairs, mean over epochs
        return "median", "mean", None  # return explicit reducers; no trimming fraction needed
    if name_norm == "median-median":  # preset: median over pairs, median over epochs
        return "median", "median", None  # return explicit reducers; no trimming fraction needed
    if name_norm == "mean-trim":  # preset: mean over pairs, trimmed mean over epochs
        if proportion_to_cut is None:  # ensure caller supplied a trim fraction
            raise ValueError("proportion_to_cut is required for 'mean-trim'.")  # clear API error
        return "mean", "trimmed", proportion_to_cut  # return reducers and provided trim fraction
    if name_norm == "median-trim":  # preset: median over pairs, trimmed mean over epochs
        if proportion_to_cut is None:  # ensure caller supplied a trim fraction
            raise ValueError("proportion_to_cut is required for 'median-trim'.")  # clear API error
        return "median", "trimmed", proportion_to_cut  # return reducers and provided trim fraction
    raise ValueError(f"Unknown aggregation strategy: {name!r}")  # explicit error for unsupported names
