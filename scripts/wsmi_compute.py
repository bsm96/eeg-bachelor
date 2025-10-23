#!/usr/bin/env python3
from __future__ import annotations  # future-compatible annotations
import argparse  # CLI argument parsing
import json  # write a run_config snapshot alongside outputs
import os  # path handling
from pathlib import Path  # path utilities
from typing import List, Optional  # type hints

import numpy as np
import mne  # EEG I/O, filtering, epochs
from tqdm import tqdm  # progress bar for long epoch loops

# Project imports (our modules)
from eeg.wsmi.filters import bandpass_epochs  # filtering via MNE backend
from eeg.wsmi.compute import compute_wsmi_matrix  # our NumPy wSMI implementation


def _parse_notch_list(s: Optional[str]) -> Optional[List[float]]:
    parser_list = None  # default: no notch
    if s:  # if user provided a CSV string
        parts = [p.strip() for p in s.split(",")]  # split values on commas
        parser_list = [float(p) for p in parts if len(p) > 0]  # cast to float where possible
        if len(parser_list) == 0:  # if parsing produced an empty list
            parser_list = None  # treat as no notch
    return parser_list  # list of notch frequencies or None


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)  # create a Path object from user input
    outdir.mkdir(parents=True, exist_ok=True)  # ensure directory exists recursively
    return outdir  # return as Path for downstream use


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-epoch wSMI matrices from an MNE Epochs file.")
    ap.add_argument("--epochs", required=True, help="Path to an MNE .fif epochs file (preloaded or loadable).")
    ap.add_argument("--out", required=True, help="Output directory to store NPZ and run_config.json.")
    ap.add_argument("--l-freq", type=float, required=True, help="High-pass cutoff in Hz (set externally).")
    ap.add_argument("--h-freq", type=float, required=True, help="Low-pass cutoff in Hz (set externally).")
    ap.add_argument("--use-notch", action="store_true", help="Enable notch filtering if provided frequencies are set.")
    ap.add_argument("--notch-freqs", type=str, default=None, help="Comma-separated notch frequencies in Hz.")
    ap.add_argument("--k", type=int, required=True, help="Embedding dimension (set externally).")
    ap.add_argument("--tau", type=int, required=True, help="Lag in samples (set externally).")
    ap.add_argument("--tie-break", type=str, default="jitter", choices=["jitter", "ordinal"],
                    help="Tie policy for ordinal patterns.")
    ap.add_argument("--normalize", action="store_true", help="Normalize wSMI by ln(k!).")
    ap.add_argument("--skip-filter", action="store_true", help="Skip filtering step if epochs are already filtered.")
    ap.add_argument("--picks", type=str, default="eeg", help="MNE picks selector (e.g., 'eeg').")
    ap.add_argument("--save-symbols", action="store_true", help="Also save per-epoch symbols (memory heavier).")
    args = ap.parse_args()  # parse CLI arguments

    outdir = _ensure_outdir(args.out)  # make sure the output directory exists
    notch_list = _parse_notch_list(args.notch_freqs)  # parse optional notch list from CSV string

    epochs = mne.read_epochs(args.epochs, preload=True, verbose="ERROR")  # load epochs into memory (as in literature stacks)
    if not args.skip_filter:  # apply filtering only when requested by user configuration
        epochs = bandpass_epochs(  # use MNE-based filter wrapper to ensure consistent preprocessing
            epochs=epochs,
            l_freq=float(args.l_freq),
            h_freq=float(args.h_freq),
            notch=(notch_list if args.use_notch else None),
            picks=args.picks,
            n_jobs=1,
        )

    data = epochs.get_data(picks=args.picks)  # extract data as (n_epochs, n_channels, n_times)
    ch_names = epochs.copy().pick(args.picks).ch_names  # collect final channel names for metadata
    sfreq = float(epochs.info["sfreq"])  # store sampling frequency from the loaded file
    n_epochs, n_channels, _ = data.shape  # unpack shape for downstream allocation

    Ws = np.empty((n_epochs, n_channels, n_channels), dtype=float)  # allocate array for all wSMI matrices
    symbols_list: Optional[List[np.ndarray]] = [] if args.save_symbols else None  # optional storage for symbols

    for idx in tqdm(range(n_epochs), desc="wSMI", unit="epoch"):  # show progress during long computations
        X = data[idx]  # select a single epoch as (n_channels, n_times)
        S, M = compute_wsmi_matrix(  # compute symbols and matrix for current epoch
            X,
            k=int(args.k),
            tau=int(args.tau),
            tie_break=args.tie_break,
            normalize=bool(args.normalize),
            diag_value=np.nan,
        )
        Ws[idx] = M  # store the matrix in the preallocated container
        if symbols_list is not None:  # if user asked to save symbols
            symbols_list.append(S)  # append symbol matrix for the current epoch

    wsmi_path = outdir / "wsmi_matrices.npz"  # define an output path for NPZ bundle
    np.savez_compressed(  # write matrices (and optionally symbols) with compression
        wsmi_path,
        Ws=Ws,
        symbols=(np.array(symbols_list, dtype=object) if symbols_list is not None else None),
        ch_names=np.array(ch_names, dtype=object),
    )

    run_config = {  # capture configuration for full reproducibility
        "epochs_path": str(Path(args.epochs).resolve()),
        "output_dir": str(outdir.resolve()),
        "sfreq": sfreq,
        "l_freq": float(args.l_freq),
        "h_freq": float(args.h_freq),
        "use_notch": bool(args.use_notch),
        "notch_freqs": notch_list,
        "k": int(args.k),
        "tau": int(args.tau),
        "tie_break": args.tie_break,
        "normalize": bool(args.normalize),
        "skip_filter": bool(args.skip_filter),
        "picks": args.picks,
        "save_symbols": bool(args.save_symbols),
        "n_epochs": int(n_epochs),
        "n_channels": int(n_channels),
        "library_notes": {
            "numpy": "Also used by C&K, Engemann, Della Bella",
            "mne": "Also used by C&K, Engemann, Della Bella",
            "tqdm": "Additional helper for progress; improves UX on long runs",
        },
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:  # open a JSON file for writing config
        json.dump(run_config, f, indent=2)  # write configuration snapshot with indentation


if __name__ == "__main__":  # standard Python script entry point
    main()  # run the CLI
