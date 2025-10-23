# src/eeg/wsmi/filters.py

# Beware, notch is only applied if explicitly requested. It is tuned off by default.

from __future__ import annotations  # ensure future-compatible annotations
from typing import Optional, Sequence  # typing utilities for clear signatures

import numpy as np
import mne  # EEG I/O, filtering, epochs
from mne.filter import filter_data, notch_filter  # MNE core filtering utilities


def bandpass_array(
    X: np.ndarray,                 # 2D array: shape (n_channels, n_times), data of a single epoch
    sfreq: float,                  # sampling frequency in Hz (provided externally)
    l_freq: float,                 # high-pass cutoff in Hz (provided externally)
    h_freq: float,                 # low-pass cutoff in Hz (provided externally)
    notch: Optional[Sequence[float]] = None,  # list/tuple of notch frequencies in Hz (optional, provided externally)
    fir_design: str = "firwin",    # FIR design string; configurable from caller (algorithmic choice, not a numeric)
    phase: str = "zero",           # phase mode; configurable from caller (algorithmic choice, not a numeric)
    n_jobs: int = 1,               # parallel jobs; configurable from caller
) -> np.ndarray:
    """
    Apply optional notch filtering followed by band-pass filtering on a (channels × time) array.

    Returns a new float64 array with the same shape as input.
    """
    X = np.ascontiguousarray(X, dtype=float)  # ensure contiguous float array for MNE filtering backend
    if l_freq is None or h_freq is None:  # validate that both cutoffs are provided
        raise ValueError("Both l_freq and h_freq must be provided.")
    if not (0.0 <= float(l_freq) < float(h_freq) < float(sfreq) / 2.0):  # validate band is within Nyquist
        raise ValueError(f"Invalid band: l_freq={l_freq}, h_freq={h_freq}, sfreq={sfreq}.")

    Xf = X.copy()  # operate on a copy to keep input immutable for callers

    if notch:  # optionally remove narrowband interference before band-pass
        Xf = notch_filter(        # apply notch at the specified frequencies (MNE implementation)
            Xf, Fs=float(sfreq),  # pass sampling frequency from caller
            freqs=list(notch),    # convert to list for MNE API
            n_jobs=int(n_jobs),   # allow parallel execution if requested
        )

    Xf = filter_data(             # apply band-pass filtering using MNE’s filter
        Xf,
        sfreq=float(sfreq),       # pass sampling frequency from caller
        l_freq=float(l_freq),     # pass high-pass cutoff from caller
        h_freq=float(h_freq),     # pass low-pass cutoff from caller
        method="fir",             # use FIR; consistent with zero-phase if requested
        fir_design=fir_design,    # configurable FIR design string
        phase=phase,              # zero-phase (filtfilt-like) or other as requested by caller
        n_jobs=int(n_jobs),       # allow parallel execution if requested
    )
    return Xf  # return filtered array with same shape as input


def bandpass_epochs(
    epochs: mne.Epochs,                    # MNE Epochs object to be filtered
    l_freq: float,                         # high-pass cutoff in Hz (provided externally)
    h_freq: float,                         # low-pass cutoff in Hz (provided externally)
    notch: Optional[Sequence[float]] = None,  # list/tuple of notch frequencies in Hz (optional, provided externally)
    picks: str | Sequence[str] = "eeg",    # which channels to filter; configurable by caller
    phase: str = "zero",                   # phase mode; configurable from caller
    n_jobs: int = 1,                       # parallel jobs; configurable from caller
) -> mne.Epochs:
    """
    Apply optional notch and then band-pass filtering directly on a copy of an MNE Epochs object.
    """
    ep = epochs.copy()                               # work on a copy to avoid mutating the input epochs
    if notch:                                        # optionally remove narrowband interference
        ep.notch_filter(                             # MNE-notch on selected channels
            freqs=list(notch),                       # pass external notch list
            picks=picks,                             # keep filtering limited to requested channel types
            n_jobs=int(n_jobs),                      # allow parallel execution if requested
        )
    ep.filter(                                       # apply band-pass on the epochs
        l_freq=float(l_freq),                        # pass high-pass cutoff from caller
        h_freq=float(h_freq),                        # pass low-pass cutoff from caller
        picks=picks,                                 # restrict to requested channels
        n_jobs=int(n_jobs),                          # allow parallel execution if requested
        phase=phase,                                 # phase mode as requested
    )
    return ep  # return the filtered copy of epochs


def preprocess_epoch_array(
    X: np.ndarray,                       # (n_channels, n_times) array of a single epoch
    sfreq: float,                        # sampling frequency in Hz (provided externally)
    l_freq: float,                       # high-pass cutoff in Hz (provided externally)
    h_freq: float,                       # low-pass cutoff in Hz (provided externally)
    use_notch: bool = False,             # whether to apply notch filtering (controlled externally)
    notch_freqs: Optional[Sequence[float]] = None,  # notch frequencies in Hz if enabled (provided externally)
    fir_design: str = "firwin",          # FIR design string; configurable from caller
    phase: str = "zero",                 # phase mode string; configurable from caller
    n_jobs: int = 1,                     # parallel jobs; configurable from caller
) -> np.ndarray:
    """
    One-stop preprocessing for NumPy epochs: optional notch followed by band-pass, ready for wSMI.
    """
    notch = list(notch_freqs) if (use_notch and notch_freqs) else None  # build notch list only when requested
    Xf = bandpass_array(                   # delegate to the array-based band-pass utility
        X=X,
        sfreq=float(sfreq),
        l_freq=float(l_freq),
        h_freq=float(h_freq),
        notch=notch,
        fir_design=fir_design,
        phase=phase,
        n_jobs=int(n_jobs),
    )
    return Xf  # return the preprocessed epoch array
