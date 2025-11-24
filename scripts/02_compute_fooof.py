#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute FOOOF (aperiodic + periodic components) for each *-epo.fif file.

For each file:
    - Compute Welch PSD per epoch and EEG channel.
    - Average PSD across epochs to get one spectrum per channel.
    - Fit FOOOFGroup across channels in a given frequency band.
    - Save FOOOF parameters and basic data into a compressed NPZ file.
"""

from pathlib import Path               # Path utilities for working with files and folders
import argparse                        # Command line argument parsing
import numpy as np                     # Numerical operations and NPZ saving
import mne                             # EEG/MEG tools
from fooof import FOOOFGroup           # FOOOF group fitting object


def main() -> None:
    """Parse CLI arguments, loop over epochs files, and compute FOOOF parameters."""
    parser = argparse.ArgumentParser(
        description="Minimal FOOOF computation on epoched EEG (one spectrum per channel)."
    )  # Create argument parser with short description

    # Input / output configuration
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input *-epo.fif files.",
    )  # Folder containing epoch files
    parser.add_argument(
        "--glob",
        type=str,
        default="*-epo.fif",
        help="Glob pattern for selecting epoch files (default: '*-epo.fif').",
    )  # Glob pattern used inside input-dir
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Base output directory for FOOOF NPZ files.",
    )  # Base folder for NPZ outputs

    # Frequency band for PSD + FOOOF
    parser.add_argument(
        "--fmin",
        type=float,
        default=1.0,
        help="Lower frequency bound (Hz) for PSD and FOOOF.",
    )  # Lower bound of fit range
    parser.add_argument(
        "--fmax",
        type=float,
        default=40.0,
        help="Upper frequency bound (Hz) for PSD and FOOOF.",
    )  # Upper bound of fit range

    # Optional subset tag to mirror wPLI / wSMI structure
    parser.add_argument(
        "--subset-tag",
        type=str,
        default=None,
        help="Optional tag appended to output folder name (e.g. 'sliding16s_s1s').",
    )  # Text tag that is added in output path

    # Overwrite behavior
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ outputs if they already exist.",
    )  # Flag that allows overwriting existing files

    # Parallel jobs for FOOOFGroup (internally uses joblib)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for FOOOFGroup.fit.",
    )  # Number of workers

    # Optional epoch filtering by event label names (only for event-locked epochs)
    parser.add_argument(
        "--include-labels",
        nargs="+",
        default=None,
        help=(
            "Optional list of epoch label names to include; "
            "labels must match epochs.event_id keys."
        ),
    )  # Event label names to keep

    # Simple FOOOF settings
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=6,
        help="Maximum number of peaks per spectrum (FOOOF max_n_peaks).",
    )  # Maximum number of oscillatory peaks to fit
    parser.add_argument(
        "--aperiodic-mode",
        type=str,
        default="fixed",
        choices=["fixed", "knee"],
        help="Aperiodic mode for FOOOF ('fixed' or 'knee').",
    )  # Aperiodic model type

    args = parser.parse_args()          # Parse command line arguments

    in_dir = Path(args.input_dir)       # Convert input directory to Path
    out_dir = Path(args.out_dir)        # Convert base output directory to Path

    # Short string for the frequency band to use in folder and file names
    band_str = f"{int(args.fmin)}-{int(args.fmax)}Hz"  # Example: "1-40Hz"

    # Optional subset tag for nested folder (e.g. sliding window vs events)
    if args.subset_tag is not None:
        out_dir = out_dir / args.subset_tag            # Append subset tag to out_dir

    # Add a FOOOF-specific subfolder with band label
    out_dir = out_dir / f"fooof_{band_str}"           # Append FOOOF band folder
    out_dir.mkdir(parents=True, exist_ok=True)        # Create folders if they do not exist

    # Loop over all epoch files in input directory matching the glob pattern
    for epo_path in sorted(in_dir.glob(args.glob)):
        print(f"Computing FOOOF for {epo_path.name}")  # Simple progress print

        # Construct output file name per subject
        out_name = f"{epo_path.stem}_fooof_{band_str}.npz"  # Example: sub-01-epo_fooof_1-40Hz.npz
        out_path = out_dir / out_name                       # Full output path

        # Skip existing outputs unless overwrite is requested
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing: {out_path}")             # Inform about skipped file
            continue                                        # Move to next file

        # Load epoched EEG data
        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR")  # Load epochs from disk
        epochs.pick("eeg")                                  # Keep only EEG channels

        # Optional subset of epochs by label names (only when include-labels is provided)
        if args.include_labels is not None:
            event_id = epochs.event_id                      # Mapping from label name to event code
            wanted_codes = [event_id[label] for label in args.include_labels]  # Selected event codes
            mask = np.isin(epochs.events[:, 2], wanted_codes)  # Boolean mask of selected epochs
            epochs = epochs[mask]                           # Restrict epochs object to these events

        # Compute PSD using Welch method in the target frequency band
        psd = epochs.compute_psd(                           # PSD object from MNE
            method="welch",                                 # PSD estimation method
            fmin=args.fmin,                                 # Lower frequency bound (Hz)
            fmax=args.fmax,                                 # Upper frequency bound (Hz)
        )
        freqs = psd.freqs                                   # Frequency axis (1D array)
        psds = psd.get_data()                               # PSD values: shape (n_epochs, n_channels, n_freqs)

        # Average PSD across epochs to get one spectrum per channel
        psd_mean = psds.mean(axis=0)                        # Shape: (n_channels, n_freqs)

        # Channel names as numpy array for storing with NPZ
        ch_names = np.asarray(epochs.ch_names, dtype=str)   # Channel name list

        # Initialize FOOOFGroup with simple, reasonable defaults
        fg = FOOOFGroup(
            peak_width_limits=[1.0, 12.0],                  # Allowed peak widths in Hz
            max_n_peaks=args.max_peaks,                     # Maximum number of peaks per spectrum
            min_peak_height=0.0,                            # Absolute minimum peak height
            peak_threshold=2.0,                             # Relative threshold in standard deviations
            aperiodic_mode=args.aperiodic_mode,             # 'fixed' or 'knee'
            verbose=False,                                  # Disable verbose printouts
        )

        # Fit FOOOFGroup across channels (one spectrum per channel)
        fg.fit(freqs, psd_mean, freq_range=[args.fmin, args.fmax])  # Fit FOOOF model for each channel

        # Extract aperiodic parameters and goodness-of-fit metrics
        aperiodic_params = fg.get_params("aperiodic_params")        # Offset, (knee), exponent
        exponent = fg.get_params("aperiodic_params", "exponent")    # Exponent only
        offset = fg.get_params("aperiodic_params", "offset")        # Offset only
        error = fg.get_params("error")                              # Model error per channel
        r2 = fg.get_params("r_squared")                             # R-squared per channel

        # Save frequency axis, mean PSD, FOOOF parameters and channel names
        np.savez_compressed(
            out_path,
            freqs=freqs.astype(np.float32),                         # Frequencies in Hz
            psd_mean=psd_mean.astype(np.float32),                   # Mean PSD per channel
            aperiodic_params=aperiodic_params.astype(np.float32),   # Full aperiodic parameter vectors
            exponent=exponent.astype(np.float32),                   # Aperiodic exponent per channel
            offset=offset.astype(np.float32),                       # Aperiodic offset per channel
            error=error.astype(np.float32),                         # FOOOF error per channel
            r2=r2.astype(np.float32),                               # FOOOF R^2 per channel
            ch_names=ch_names,                                      # Channel name list
        )

        print(f"Saved: {out_path}")                                 # Confirm output path in console


if __name__ == "__main__":
    main()                                                          # Run main if script is executed directly
