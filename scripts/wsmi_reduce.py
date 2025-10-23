#!/usr/bin/env python3
from __future__ import annotations  # future-compatible annotations
import argparse  # CLI parsing
import json  # write a run_config snapshot for reproducibility
from pathlib import Path  # path utilities from the standard library
from typing import Optional  # precise type hints

import numpy as np
import pandas as pd  # tabular I/O

# Our project aggregation utilities (SciPy-based trimmed mean is used inside this module)
from eeg_old.wsmi.aggregation import (  # local module with tested reducers
    aggregate_wsmi_matrices,        # runs pairwise + temporal reduction
    aggregator_from_string,         # maps a strategy string to reducer choices
)


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)  # turn input into a Path object
    outdir.mkdir(parents=True, exist_ok=True)  # create directory tree if it does not exist
    return outdir  # return Path for downstream calls


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduce per-epoch wSMI matrices to scalars (mean/median/trimmed).")
    ap.add_argument("--wsmi-npz", required=True, help="Path to NPZ produced by wsmi_compute.py (contains Ws).")
    ap.add_argument("--out", required=True, help="Output directory for CSV/JSON results.")
    ap.add_argument("--strategy", required=True,
                    help="Aggregation strategy, e.g., 'mean-mean', 'median-mean', 'median-median', 'mean-trim', 'median-trim'.")
    ap.add_argument("--trim-proportion", type=float, default=None,
                    help="Trim fraction per tail (required when strategy implies trimmed mean).")
    ap.add_argument("--include-diagonal", action="store_true",
                    help="Include diagonal entries when reducing over pairs (default excludes diagonal).")
    ap.add_argument("--subject-id", type=str, default=None,
                    help="Optional subject identifier to include in outputs (provided externally).")
    ap.add_argument("--band", type=str, default=None,
                    help="Optional band label to include in outputs (provided externally).")
    args = ap.parse_args()  # parse CLI arguments into a namespace

    outdir = _ensure_outdir(args.out)  # ensure the output directory exists

    npz_path = Path(args.wsmi_npz)  # create a Path handle for the NPZ input
    if not npz_path.exists():  # validate that the input file is present
        raise FileNotFoundError(f"File not found: {npz_path}")  # explicit error for missing input

    with np.load(npz_path, allow_pickle=True) as npz:  # open the NPZ bundle created by wsmi_compute.py
        if "Ws" not in npz.files:  # verify expected array key exists
            raise KeyError("Input NPZ must contain key 'Ws' (array of wSMI matrices).")  # explicit schema error
        Ws = npz["Ws"]  # load 3D array of shape (n_epochs, n_channels, n_channels)
        ch_names = npz.get("ch_names", None)  # load optional channel names metadata
        # symbols may be present but are not needed for aggregation; intentionally ignored here

    pair_reducer, time_reducer, trim_prop = aggregator_from_string(  # resolve reducer choices from strategy name
        args.strategy, proportion_to_cut=args.trim_proportion
    )

    epoch_scalars, subject_scalar = aggregate_wsmi_matrices(  # run aggregation on all epoch matrices
        Ws,
        pair_reducer=pair_reducer,
        time_reducer=time_reducer,
        proportion_to_cut=trim_prop,
        include_diagonal=bool(args.include_diagonal),
    )

    # Build a DataFrame for the per-epoch series to simplify downstream QC/plots
    df_epochs = pd.DataFrame({  # assemble per-epoch results in a tidy table
        "epoch_index": np.arange(len(epoch_scalars), dtype=int),  # explicit epoch indices
        "wsmi_scalar": np.asarray(epoch_scalars, dtype=float),    # per-epoch scalar values
    })
    if args.subject_id is not None:  # optionally annotate subject identifier
        df_epochs["subject_id"] = args.subject_id  # broadcast the provided subject id into the table
    if args.band is not None:  # optionally annotate band label
        df_epochs["band"] = args.band  # broadcast the provided band label into the table

    # Save per-epoch series
    epochs_csv = outdir / "wsmi_epoch_scalars.csv"  # construct an output path for epoch series
    df_epochs.to_csv(epochs_csv, index=False)  # write CSV without index for clean consumption

    # Save a compact JSON summary with the final subject-level scalar
    summary = {  # create a plain dictionary for the subject-level scalar
        "subject_id": args.subject_id,
        "band": args.band,
        "strategy": args.strategy,
        "include_diagonal": bool(args.include_diagonal),
        "trim_proportion": trim_prop,
        "subject_scalar": None if np.isnan(subject_scalar) else float(subject_scalar),
        "n_epochs": int(Ws.shape[0]),
        "n_channels": int(Ws.shape[1]) if Ws.ndim == 3 else None,
    }
    with open(outdir / "wsmi_summary.json", "w", encoding="utf-8") as f:  # open a JSON file for writing
        json.dump(summary, f, indent=2)  # store subject-level result with configuration

    # Save a run configuration snapshot for full reproducibility
    run_config = {  # capture the exact inputs and choices used
        "wsmi_npz": str(npz_path.resolve()),
        "output_dir": str(outdir.resolve()),
        "strategy": args.strategy,
        "resolved_pair_reducer": pair_reducer,
        "resolved_time_reducer": time_reducer,
        "resolved_trim_proportion": trim_prop,
        "include_diagonal": bool(args.include_diagonal),
        "subject_id": args.subject_id,
        "band": args.band,
        "library_notes": {
            "numpy": "Also used by C&K, Engemann, Della Bella",
            "pandas": "Common in C&K/Engemann-style analysis stacks for tabular outputs",
            "scipy": "Used inside eeg.wsmi.aggregation for trimmed mean",
        },
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:  # create config snapshot file
        json.dump(run_config, f, indent=2)  # write human-readable JSON for later auditing


if __name__ == "__main__":  # standard script entry point
    main()  # run the CLI
