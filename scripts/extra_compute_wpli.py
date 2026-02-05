#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute sensor-space wPLI for each *-epo.fif file using mne-connectivity.

The script loops over epoched EEG files, computes wPLI in a single frequency band,
and saves one compressed NPZ per file.
"""

from pathlib import Path                  # Path utilities for file system handling
import argparse                           # Simple command line argument parsing
import numpy as np                        # Array handling and NPZ saving
import mne                                # EEG/MEG tools
from mne_connectivity import spectral_connectivity_epochs  # wPLI implementation


def main() -> None:
    """Parse arguments, compute wPLI for all epoch files, and save NPZ outputs."""
    parser = argparse.ArgumentParser(     # Create argument parser
        description="Minimal wPLI computation using mne-connectivity."
    )
    parser.add_argument(                  # Input directory with *-epo.fif files
        "--input-dir", type=str, required=True
    )
    parser.add_argument(                  # Glob pattern to select epoch files; a pattern for the file names that should be icluded from "--input-dir
        "--glob", type=str, default="*-epo.fif"
    )
    parser.add_argument(                  # Base output directory
        "--out-dir", type=str, required=True
    )
    parser.add_argument(                  # Lower frequency bound (Hz)
        "--fmin", type=float, default=8.0
    )
    parser.add_argument(                  # Upper frequency bound (Hz)
        "--fmax", type=float, default=13.0
    )
    parser.add_argument(                  # Optional text tag added to output path
        "--subset-tag", type=str, default=None
    )
    parser.add_argument(                  # Overwrite existing outputs if specified
        "--overwrite", action="store_true"
    )
    parser.add_argument(                  # Number of parallel jobs for connectivity
        "--n-jobs", type=int, default=1
    )
    parser.add_argument(                  # Optional epoch selection by label names
        "--include-labels", nargs="+", default=None
    )

    args = parser.parse_args()           # Parse all command line arguments

    in_dir = Path(args.input_dir)        # Convert input directory to Path object
    out_dir = Path(args.out_dir)         # Convert output directory to Path object

    band_str = f"{int(args.fmin)}-{int(args.fmax)}Hz"  # Short text for frequency band

    if args.subset_tag is not None:      # Append subset tag to output directory if provided
        out_dir = out_dir / args.subset_tag

    out_dir = out_dir / f"wpli_{band_str}"  # Add band-specific subfolder
    out_dir.mkdir(parents=True, exist_ok=True)  # Create output directory tree

    for epo_path in sorted(in_dir.glob(args.glob)):  # Loop over all matching epoch files
        print(f"Computing wPLI for {epo_path.name}")  # Simple progress print

        out_name = f"{epo_path.stem}_wpli_{band_str}.npz"  # Output file name per subject
        out_path = out_dir / out_name                     # Full output path

        if out_path.exists() and not args.overwrite:      # Skip if file exists and no overwrite
            print(f"Skip existing: {out_path}")
            continue

        epochs = mne.read_epochs(epo_path, preload=True)  # Load epochs from disk
        epochs.pick("eeg")                              # Keep only EEG channels for connectivity

        if args.include_labels is not None:             # Optionally restrict epochs by label name
            event_id = epochs.event_id                  # Mapping from label name to event code
            wanted_codes = [event_id[label] for label in args.include_labels]  # Selected codes
            mask = np.isin(epochs.events[:, 2], wanted_codes)  # Boolean mask for selected epochs
            epochs = epochs[mask]                       # Subset epochs to selected events

        con = spectral_connectivity_epochs(
            epochs,                   # Epoched data
            method="wpli",            # Connectivity metric
            mode="multitaper",        # Spectral estimation mode
            fmin=args.fmin,           # Lower frequency bound (Hz)
            fmax=args.fmax,           # Upper frequency bound (Hz)
            faverage=True,            # Average over frequencies within the band
            mt_adaptive=False,        # Simpler multitaper configuration
            n_jobs=args.n_jobs,       # Number of parallel jobs
        )

        wpli = con.get_data(output="dense")   # Connectivity matrix (channels × channels × freqs)
        wpli = np.squeeze(wpli, axis=-1)      # Remove frequency axis when faverage=True

        freqs = np.asarray(con.freqs, dtype=np.float32)   # Frequencies used in estimation
        ch_names = np.asarray(epochs.ch_names, dtype=str) # Channel names for reference

        np.savez_compressed(                 # Save results and minimal metadata in NPZ
            out_path,
            wpli=wpli.astype(np.float32),    # wPLI matrix
            freqs=freqs,                     # Frequency vector (usually length 1)
            ch_names=ch_names,               # Channel name list
        )

        print(f"Saved: {out_path}")          # Confirm save location in console


if __name__ == "__main__":
    main()                                   # Run main function when script is executed