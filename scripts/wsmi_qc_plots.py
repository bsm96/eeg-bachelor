#!/usr/bin/env python3
from __future__ import annotations  # future-compatible annotations for clarity

# --- Make src/ importable without setting PYTHONPATH ---
import sys  # add src/ to sys.path so 'eeg' package can be imported from a src-layout
from pathlib import Path  # robust path handling across OSes
ROOT = Path(__file__).resolve().parents[1]  # project root directory
SRC = ROOT / "src"  # src directory that contains the 'eeg' package
if str(SRC) not in sys.path:  # ensure src is importable in this process
    sys.path.insert(0, str(SRC))  # prepend src path

# --- Standard library ---
import argparse  # parse CLI arguments
import json  # read summary JSON file
from typing import Optional  # type hints for optional parameters

# --- Third-party ---
import numpy as np  # numerical arrays
import pandas as pd  # tabular I/O
import matplotlib.pyplot as plt  # plotting for QC

def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)  # convert to Path
    outdir.mkdir(parents=True, exist_ok=True)  # create directory tree if missing
    return outdir  # return as Path for downstream use

def main() -> None:
    ap = argparse.ArgumentParser(description="QC plots for wSMI results (epoch series, histogram, median matrix).")  # CLI help
    ap.add_argument("--epoch-csv", required=True, help="Path to wsmi_epoch_scalars.csv from wsmi_reduce.py.")  # per-epoch CSV
    ap.add_argument("--wsmi-npz", required=True, help="Path to wsmi_matrices.npz from wsmi_compute.py.")  # NPZ with Ws
    ap.add_argument("--summary-json", default=None, help="Optional path to wsmi_summary.json to annotate the title.")  # optional
    ap.add_argument("--out", required=True, help="Output directory for saved figures.")  # where to save figures
    ap.add_argument("--fig-dpi", type=int, default=150, help="Figure DPI, set externally.")  # DPI configurable
    ap.add_argument("--vmin", type=float, default=None, help="Optional lower color limit for matrix imshow.")  # color scale lower
    ap.add_argument("--vmax", type=float, default=None, help="Optional upper color limit for matrix imshow.")  # color scale upper
    ap.add_argument("--show-labels", action="store_true", help="Show channel labels on matrix axes if present.")  # axes labels toggle
    args = ap.parse_args()  # parse arguments

    outdir = _ensure_outdir(args.out)  # ensure output directory exists

    df = pd.read_csv(args.epoch_csv)  # load per-epoch scalars table
    series = df["wsmi_scalar"].to_numpy(dtype=float)  # extract scalar series as float array
    epoch_idx = df["epoch_index"].to_numpy(dtype=int)  # extract epoch indices as integers
    subj = df["subject_id"].iloc[0] if "subject_id" in df.columns and len(df) > 0 else None  # read subject id if present
    band = df["band"].iloc[0] if "band" in df.columns and len(df) > 0 else None  # read band label if present

    with np.load(Path(args.wsmi_npz), allow_pickle=True) as npz:  # open NPZ file produced by wsmi_compute.py
        Ws = npz["Ws"]  # load 3D array: (n_epochs, n_channels, n_channels)
        ch_names = npz["ch_names"].tolist() if "ch_names" in npz.files else None  # optional channel names list

    title_bits = []  # initialize a list of title components
    if subj:  # if subject id available
        title_bits.append(f"subject: {subj}")  # append subject id
    if band:  # if band label available
        title_bits.append(f"band: {band}")  # append band label
    if args.summary_json:  # if summary json provided, try to read scalar
        try:
            with open(args.summary_json, "r", encoding="utf-8") as f:  # open JSON file
                summary = json.load(f)  # parse JSON
                if "subject_scalar" in summary and summary["subject_scalar"] is not None:  # check key
                    title_bits.append(f"summary={summary['subject_scalar']:.3f}")  # add scalar to title bits
        except Exception:
            pass  # ignore any errors reading the optional summary

    title_suffix = " | ".join(title_bits) if title_bits else ""  # construct a suffix for figure titles

    # --- Plot 1: Epoch scalar time series ---
    plt.figure()  # create a new figure
    plt.plot(epoch_idx, series)  # line plot of scalars over epoch index (default style, no explicit colors)
    plt.xlabel("epoch index")  # label x-axis
    plt.ylabel("wSMI scalar")  # label y-axis
    plt.title(f"wSMI per epoch {title_suffix}".strip())  # set plot title with optional suffix
    plt.grid(True)  # add a grid for readability
    plt.tight_layout()  # compact layout to avoid overlaps
    plt.savefig(outdir / "wsmi_epoch_series.png", dpi=int(args.fig_dpi))  # save figure to disk

    # --- Plot 2: Histogram of epoch scalars ---
    plt.figure()  # create a new figure
    plt.hist(series[~np.isnan(series)])  # histogram of finite scalars (default bins; no explicit colors)
    plt.xlabel("wSMI scalar")  # label x-axis
    plt.ylabel("count")  # label y-axis
    plt.title(f"wSMI epoch scalar distribution {title_suffix}".strip())  # set title with optional suffix
    plt.tight_layout()  # compact layout
    plt.savefig(outdir / "wsmi_epoch_hist.png", dpi=int(args.fig_dpi))  # save figure

    # --- Plot 3: Median wSMI matrix across epochs ---
    finite_Ws = Ws[np.all(np.isfinite(Ws), axis=(1, 2))]  # keep only epochs with finite matrices
    if finite_Ws.shape[0] > 0:  # ensure there is at least one valid epoch matrix
        W_med = np.median(finite_Ws, axis=0)  # compute element-wise median across epochs
        plt.figure()  # create a new figure
        im = plt.imshow(W_med, vmin=args.vmin, vmax=args.vmax, origin="upper")  # show matrix image with optional limits
        plt.colorbar(im)  # add colorbar to interpret magnitude scale
        plt.title(f"Median wSMI matrix {title_suffix}".strip())  # set title
        if args.show_labels and ch_names is not None:  # optionally add channel labels
            ticks = np.arange(len(ch_names))  # create tick positions
            plt.xticks(ticks, ch_names, rotation=90)  # set x-ticks to channel names
            plt.yticks(ticks, ch_names)  # set y-ticks to channel names
        plt.tight_layout()  # compact layout for labels and colorbar
        plt.savefig(outdir / "wsmi_median_matrix.png", dpi=int(args.fig_dpi))  # save matrix figure

    # Done: intentionally no plt.show() in CLI to avoid blocking
    # Figures are saved to the requested output directory

if __name__ == "__main__":  # standard script entry point
    main()  # run the CLI
