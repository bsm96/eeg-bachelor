#!/usr/bin/env python3
from __future__ import annotations  # future-compatible annotations

# --- Make src/ importable without setting PYTHONPATH ---
import sys  # add src/ to sys.path so 'eeg' package is importable
from pathlib import Path  # robust path handling

ROOT = Path(__file__).resolve().parents[1]  # project root folder
SRC = ROOT / "src"  # src/ folder that contains the 'eeg' package
if str(SRC) not in sys.path:  # ensure src/ is visible to this process
    sys.path.insert(0, str(SRC))  # prepend src path

# --- Standard imports ---
import argparse  # CLI parsing
import json  # write run_config and summary
from typing import Optional  # typing helpers

import numpy as np
import pandas as pd  # tabular I/O

# Project imports (our modules)
from eeg.wsmi.aggregation import (  # our reducers; SciPy trimmed mean used inside
    aggregate_wsmi_matrices,
    aggregator_from_string,
)


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)  # cast to Path
    outdir.mkdir(parents=True, exist_ok=True)  # create directory tree if missing
    return outdir  # return Path object


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduce per-epoch wSMI matrices to scalars (mean/median/trimmed).")  # CLI help text
    ap.add_argument("--wsmi-npz", required=True, help="Path to NPZ produced by wsmi_compute.py (contains Ws).")  # NPZ path
    ap.add_argument("--out", required=True, help="Output directory for CSV/JSON results.")  # output dir
    ap.add_argument("--strategy", required=True,
                    help="One of: 'mean-mean', 'median-mean', 'median-median', 'mean-trim', 'median-trim'.")  # strategy
    ap.add_argument("--trim-proportion", type=float, default=None,
                    help="Trim fraction per tail when using a '...-trim' strategy.")  # trim arg
    ap.add_argument("--include-diagonal", action="store_true",
                    help="Include diagonal entries when reducing over pairs.")  # include diag flag
    ap.add_argument("--subject-id", type=str, default=None,
                    help="Optional subject identifier for outputs.")  # subject id
    ap.add_argument("--band", type=str, default=None,
                    help="Optional band label for outputs.")  # band label
    args = ap.parse_args()  # parse CLI

    outdir = _ensure_outdir(args.out)  # ensure output dir exists

    npz_path = Path(args.wsmi_npz)  # convert to Path
    if not npz_path.exists():  # check file presence
        raise FileNotFoundError(f"File not found: {npz_path}")  # explicit error

    with np.load(npz_path, allow_pickle=True) as npz:  # open NPZ bundle
        if "Ws" not in npz.files:  # schema check
            raise KeyError("Input NPZ must contain key 'Ws'.")  # explicit schema error
        Ws = npz["Ws"]  # (n_epochs, n_channels, n_channels)
        ch_names = npz.get("ch_names", None)  # optional metadata (not used further here)

    pair_reducer, time_reducer, trim_prop = aggregator_from_string(  # resolve reducers
        args.strategy, proportion_to_cut=args.trim_proportion
    )

    epoch_scalars, subject_scalar = aggregate_wsmi_matrices(  # reduce matrices → per-epoch + final scalar
        Ws,
        pair_reducer=pair_reducer,
        time_reducer=time_reducer,
        proportion_to_cut=trim_prop,
        include_diagonal=bool(args.include_diagonal),
    )

    # Per-epoch CSV (good for QC)
    df_epochs = pd.DataFrame({  # build tidy table
        "epoch_index": np.arange(len(epoch_scalars), dtype=int),  # epoch indices
        "wsmi_scalar": np.asarray(epoch_scalars, dtype=float),    # scalar per epoch
    })
    if args.subject_id is not None:  # optional subject label
        df_epochs["subject_id"] = args.subject_id  # annotate column
    if args.band is not None:  # optional band label
        df_epochs["band"] = args.band  # annotate column
    df_epochs.to_csv(outdir / "wsmi_epoch_scalars.csv", index=False)  # write CSV

    # Compact JSON summary with final subject-level scalar
    summary = {  # dictionary of summary info
        "subject_id": args.subject_id,
        "band": args.band,
        "strategy": args.strategy,
        "include_diagonal": bool(args.include_diagonal),
        "trim_proportion": trim_prop,
        "subject_scalar": None if np.isnan(subject_scalar) else float(subject_scalar),
        "n_epochs": int(Ws.shape[0]),
        "n_channels": int(Ws.shape[1]) if Ws.ndim == 3 else None,
    }
    with open(outdir / "wsmi_summary.json", "w", encoding="utf-8") as f:  # open JSON file
        json.dump(summary, f, indent=2)  # dump JSON

    # Run-config snapshot (provenance)
    run_config = {  # provenance metadata
        "wsmi_npz": str(npz_path.resolve()),
        "output_dir": str(outdir.resolve()),
        "strategy": args.strategy,
        "resolved_pair_reducer": pair_reducer,
        "resolved_time_reducer": time_reducer,
        "resolved_trim_proportion": trim_prop,
        "include_diagonal": bool(args.include_diagonal),
        "subject_id": args.subject_id,
        "band": args.band,
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:  # open run_config file
        json.dump(run_config, f, indent=2)  # write run_config


if __name__ == "__main__":  # script entry point
    main()  # run CLI
