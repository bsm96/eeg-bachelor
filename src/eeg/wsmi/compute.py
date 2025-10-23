# src/eeg/wsmi/compute.py
from __future__ import annotations  # enable future-compatible annotations for clarity
from typing import Optional, Tuple  # precise type hints for function signatures

import math  # use Python's math (factorial, log) — stable and exact for small k
import numpy as np
from .symbolic import (  # local symbolic utilities (our implementation detail)
    symbols_multi_channel,  # turns multi-channel epoch data into aligned ordinal symbols
    weight_matrix,          # builds the wSMI weight matrix consistent with our symbol coding
    ln_factorial_k,         # provides ln(k!) for normalization chosen by the caller
)


# -------------------------------
# Fast joint histogram utilities
# -------------------------------

def _joint_counts(a: np.ndarray, b: np.ndarray, n_states: int) -> np.ndarray:
    a = a.astype(np.int64, copy=False)  # ensure integer dtype for indexing arithmetic
    b = b.astype(np.int64, copy=False)  # ensure integer dtype for indexing arithmetic
    joint_index = a * n_states + b      # map each (a_t, b_t) to a single flat index deterministically
    counts = np.bincount(               # count occurrences of each joint state efficiently
        joint_index,
        minlength=n_states * n_states,  # ensure a full joint table even if some bins are empty
    )
    return counts.reshape(n_states, n_states)  # return 2D joint histogram aligned with symbol states


# ----------------------------------------
# Pairwise wSMI from two symbol sequences
# ----------------------------------------

def wsmi_pair_from_symbols(
    s_i: np.ndarray,                    # symbol stream of channel i (aligned with s_j)
    s_j: np.ndarray,                    # symbol stream of channel j (aligned with s_i)
    *,
    k: int = 3,                         # embedding dimension (can be overridden by the caller)
    W: Optional[np.ndarray] = None,     # optional weight matrix; if None, canonical weights are used
    normalize: bool = True,             # whether to divide by ln(k!) so values are comparable
    eps: float = 1e-12,                 # small constant to guard safe divisions if needed by caller
) -> float:
    """
    Compute weighted Symbolic Mutual Information (wSMI) between two symbol sequences.
    """
    n_states = int(math.factorial(k))           # number of ordinal states implied by k (use math.factorial)
    W = weight_matrix(k) if W is None else W    # use canonical weights unless caller supplies custom
    joint = _joint_counts(s_i, s_j, n_states)   # build joint histogram over aligned symbol pairs
    total = float(joint.sum())                  # total count of observed joint samples
    if total <= 0.0:                            # guard against empty inputs or invalid spans
        return float("nan")                     # return NaN to signal no valid information

    p_ab = joint / total                        # convert counts to joint probabilities
    p_a = p_ab.sum(axis=1, keepdims=True)       # marginal distribution of channel i
    p_b = p_ab.sum(axis=0, keepdims=True)       # marginal distribution of channel j
    denom = p_a @ p_b                           # outer product gives independent-model probability

    valid = (p_ab > 0.0) & (denom > 0.0) & (W > 0.0)  # include only meaningful, weighted, nonzero entries
    ratio = np.empty_like(p_ab)                 # allocate array to hold ratios safely
    ratio[valid] = p_ab[valid] / denom[valid]   # compute p(a,b)/(p(a)p(b)) on valid bins only

    contrib = np.zeros_like(p_ab)               # accumulator for weighted MI contributions
    contrib[valid] = W[valid] * p_ab[valid] * np.log(ratio[valid])  # weighted log-ratio scaled by joint prob
    mi = float(contrib.sum())                   # sum all contributions to obtain MI in nats

    if normalize:                               # optionally normalize by ln(k!) for comparability
        mi = mi / ln_factorial_k(k)             # divide by ln(k!) as per canonical wSMI normalization
    return mi                                    # deliver the pairwise wSMI value


# -------------------------------------------------
# Full (channels × channels) wSMI for one epoch
# -------------------------------------------------

def wsmi_matrix_from_symbols(
    S: np.ndarray,                  # symbols per channel with aligned time (shape: n_channels × L)
    *,
    k: int = 3,                     # embedding dimension (caller-controlled)
    W: Optional[np.ndarray] = None, # optional weight matrix consistent with the symbol coding
    normalize: bool = True,         # whether pairwise values are normalized by ln(k!)
    diag_value: float = np.nan,     # value placed on diagonal (self-pairs), configurable by caller
) -> np.ndarray:
    """
    Build the symmetric wSMI connectivity matrix for a single epoch from per-channel symbols.
    """
    C, _ = S.shape                                   # unpack number of channels for matrix dimensions
    M = np.empty((C, C), dtype=float)                # allocate output matrix with float dtype
    W_local = weight_matrix(k=k) if W is None else W # determine effective weight matrix once

    for i in range(C):                                # iterate first channel index
        M[i, i] = diag_value                          # set diagonal to caller-provided sentinel or zero
        s_i = S[i]                                    # take symbol stream for channel i (aligned)
        for j in range(i + 1, C):                     # compute only upper triangle to avoid duplicates
            s_j = S[j]                                # take symbol stream for channel j (aligned)
            v = wsmi_pair_from_symbols(               # compute pairwise wSMI with chosen settings
                s_i, s_j, k=k, W=W_local, normalize=normalize
            )
            M[i, j] = v                               # write upper-triangle entry
            M[j, i] = v                               # mirror to lower-triangle to ensure symmetry
    return M                                          # return full matrix for this epoch


# ---------------------------------------------------------
# Convenience: from raw epoch array directly to wSMI matrix
# ---------------------------------------------------------

def compute_wsmi_matrix(
    X: np.ndarray,                 # raw epoch data (n_channels × n_times) already filtered upstream
    *,
    k: int = 3,                    # embedding dimension (caller-controlled)
    tau: int = 8,                  # lag in samples (caller-controlled)
    tie_break: str = "jitter",     # tie policy passed to symbolization
    W: Optional[np.ndarray] = None,# custom weights if the caller wants a non-canonical scheme
    normalize: bool = True,        # normalize by ln(k!) if requested
    diag_value: float = np.nan,    # value to place on the diagonal of the matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level helper: compute per-channel symbols and the resulting wSMI matrix for one epoch.
    """
    X = np.ascontiguousarray(X, dtype=float)         # ensure contiguous float array for stable slicing
    C, _ = X.shape                                   # keep channel count for empty-path handling
    S = symbols_multi_channel(                       # build aligned ordinal symbols per channel
        X, k=k, tau=tau, tie_break=tie_break
    )
    if S.shape[1] == 0:                              # if too few samples for at least one symbol
        return S, np.full((C, C), np.nan, dtype=float)  # return empty symbols and a NaN-filled matrix

    M = wsmi_matrix_from_symbols(                    # compute the full connectivity matrix from symbols
        S, k=k, W=W, normalize=normalize, diag_value=diag_value
    )
    return S, M                                      # return both symbols and corresponding matrix
