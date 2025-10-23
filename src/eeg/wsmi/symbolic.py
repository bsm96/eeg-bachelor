# src/eeg/wsmi/symbolic.py
from __future__ import annotations  # enable future-compatible annotations
import math  # basic math utilities for factorial/log
from itertools import permutations  # generate ordinal patterns (permutations)
from typing import Dict, List, Tuple  # precise type hints for clarity
import numpy as np


# ---------------------------
# Basic helpers / definitions
# ---------------------------

def factorial(n: int) -> int:
    """Compute n! as an integer."""
    return math.factorial(n)  # delegate to Python's math library (exact integer factorial)


def all_permutations(k: int) -> List[Tuple[int, ...]]:
    """Return a stable, canonical list of ordinal patterns for embedding dimension k."""
    return list(permutations(range(k)))  # lexicographic order is stable and sufficient


def perm_to_index_map(k: int) -> Dict[Tuple[int, ...], int]:
    """Map a permutation tuple to a stable integer code in [0, k!-1]."""
    perms = all_permutations(k)  # generate all ordinal patterns for the given k
    return {p: i for i, p in enumerate(perms)}  # build a deterministic lookup for pattern -> code


# ---------------------------
# Ordinal symbolization
# ---------------------------

def ordinal_symbols_1d(
    x: np.ndarray,                 # 1D time series (length provided externally)
    k: int = 3,                    # embedding dimension (default can be overridden externally)
    tau: int = 8,                  # lag in samples between pattern points (default can be overridden externally)
    *,
    tie_break: str = "jitter",     # tie-handling policy set by caller
    jitter_eps: float = 1e-12,     # very small deterministic offset magnitude set by caller
) -> np.ndarray:
    """
    Convert a 1D time series to a sequence of ordinal symbols with embedding (k, tau).
    """
    x = np.ascontiguousarray(x, dtype=float)  # ensure contiguous float array for fast slicing
    T = x.shape[0]  # number of samples provided by caller
    span = (k - 1) * tau  # total span in samples for one ordinal pattern at this (k, tau)
    if T <= span:  # not enough samples for even a single pattern
        return np.empty(0, dtype=np.int32)  # return an empty symbol sequence

    L = T - span  # number of valid pattern positions for this (k, tau)
    emb = np.empty((k, L), dtype=float)  # delayed embedding matrix (rows = pattern positions)
    for m in range(k):  # build each delayed row
        start = m * tau  # row-wise offset determined by tau and row index
        emb[m] = x[start : start + L]  # slice once per row to avoid copies

    if tie_break == "jitter":  # deterministic tiny offsets to avoid equalities within columns
        scale = np.median(np.abs(emb)) or 1.0  # scale jitter relative to data magnitude
        offsets = (np.arange(k, dtype=float)[:, None]) * (jitter_eps * scale)  # monotonic offsets by row
        emb = emb + offsets  # apply offsets once to the embedding matrix
    elif tie_break == "ordinal":  # rely on stable sort for tie-breaking by index order
        pass  # no numeric modification needed
    else:
        raise ValueError("tie_break must be 'jitter' or 'ordinal'")  # explicit guard for API misuse

    order = np.argsort(emb, axis=0, kind="mergesort")  # stable rank of each column (ascending)
    perms = all_permutations(k)  # the canonical ordered list of permutations
    idx_map = {p: i for i, p in enumerate(perms)}  # map permutation -> integer code

    symbols = np.empty(L, dtype=np.int32)  # allocate symbol codes for all columns
    for j in range(L):  # convert each column’s rank pattern to its code
        p = tuple(int(order[:, j][m]) for m in range(k))  # build permutation tuple from ranked row indices
        symbols[j] = idx_map[p]  # assign the stable code for this permutation
    return symbols  # final integer-coded symbol sequence in [0, k!-1]


def symbols_multi_channel(
    X: np.ndarray,              # 2D epoch array (n_channels, n_times), supplied externally
    k: int = 3,                 # embedding dimension (default can be overridden externally)
    tau: int = 8,               # lag in samples (default can be overridden externally)
    **kwargs,                   # forwarded keyword options (e.g., tie_break, jitter_eps)
) -> np.ndarray:
    """
    Compute ordinal symbols per channel; stack into a (n_channels, L) matrix with aligned time.
    """
    X = np.ascontiguousarray(X, dtype=float)  # ensure contiguous float for efficient slicing
    C, T = X.shape  # number of channels and samples provided by caller
    span = (k - 1) * tau  # required span in samples for one pattern
    if T <= span:  # not enough samples to form any symbol
        return np.empty((C, 0), dtype=np.int32)  # return an empty (n_channels, 0) matrix

    out = []  # collect per-channel symbol sequences
    for c in range(C):  # iterate channels to preserve alignment across time
        out.append(ordinal_symbols_1d(X[c], k=k, tau=tau, **kwargs))  # compute symbols per channel
    return np.vstack(out)  # stack rows so all channels share the same L


# ---------------------------
# Weights & normalization
# ---------------------------

def weight_matrix(
    k: int = 3,                      # embedding dimension (default can be overridden externally)
    *,
    diag_weight: float = 0.0,        # weight for identical symbol pairs chosen by caller
    opposite_weight: float = 0.0,    # weight for reversed (mirror) symbol pairs chosen by caller
    others_weight: float = 1.0,      # weight for all other symbol pairs chosen by caller
) -> np.ndarray:
    """
    Build a wSMI weight matrix W of shape (k!, k!) given caller-chosen weights.
    """
    perms = all_permutations(k)  # canonical permutation list matching our symbol coding
    n = len(perms)  # number of distinct ordinal patterns for this k
    W = np.full((n, n), float(others_weight), dtype=float)  # initialize with the generic weight
    for i, p in enumerate(perms):  # fill special cases per row-pattern
        W[i, i] = float(diag_weight)  # set weight for identical pairs according to caller
        rev = tuple(reversed(p))  # compute the time-reversed (mirror) permutation
        j = perms.index(rev)  # find index of the reversed permutation in canonical order
        W[i, j] = float(opposite_weight)  # set weight for opposite pairs according to caller
    return W  # weight matrix to be used in wSMI summation


def ln_factorial_k(k: int) -> float:
    """Return natural log of k! for normalization chosen by caller."""
    return math.log(factorial(k))  # stable log-factorial via math library
