#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal wSMI runner with NICE (SymbolicMutualInformation, weighted).
- Input:  MNE Epochs FIF files (event-locked), e.g. "*-epo.fif"
- CSD:    Handled internally by NICE (no switches here)
- Output: One .npz per input with (n_epochs, n_ch, n_ch) + basic metadata
"""

from pathlib import Path
import argparse, json, sys
import numpy as np
import mne
from nice.markers.connectivity import SymbolicMutualInformation

def main():
    # ---- CLI ----
    ap = argparse.ArgumentParser(description="Compute per-epoch wSMI (NICE, minimal).")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--glob", default="*-epo.fif")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau-ms", type=float, default=8.0)
    args = ap.parse_args()

    # ---- Output folder ----
    run_dir = args.out_dir / args.input_dir.name / f"k{args.k}_tau{int(round(args.tau_ms))}ms_csd_nice"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process files ----
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No matches in {args.input_dir} with glob '{args.glob}'")
        return

    total = len(files)

    # ---- Progress bar function ----
    def render_progress(current: int, total: int, width: int = 30) -> None:
        ratio = current / total
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[PROGRESS] |{bar}| {current}/{total}")

    for idx, fpath in enumerate(files, start=1):
        print(f"[LOAD] {fpath}")
        epochs = mne.read_epochs(fpath, preload=True, verbose="ERROR")

        # Convert tau from ms to integer samples
        sfreq = float(epochs.info["sfreq"])
        tau_samp = int(round((args.tau_ms / 1000.0) * sfreq))

        # Compute wSMI with NICE (may return (C,C,E))
        smi = SymbolicMutualInformation(
            tmin=None, tmax=None, kernel=args.k, tau=tau_samp,# tmin and tmax can be None to use full epoch
            method='weighted', backend='python',
            method_params={'nthreads': 'auto'}, comment='weighted'
        )
        smi.fit(epochs)
        wsmi = smi.data_# Extract data array
        wsmi = np.asarray(wsmi)
        if wsmi.shape[-1] == len(epochs):  # (C,C,E) -> (E,C,C)
            wsmi = np.moveaxis(wsmi, -1, 0)# Move epochs to first dim

        # Events (simple: numeric event codes) and channel names
        events = epochs.events[:, 2] if getattr(epochs, "events", None) is not None else np.arange(len(epochs))
        ch_names = np.array(epochs.ch_names, dtype=object)

        # Metadata
        meta = dict(
            k=int(args.k), tau_ms=float(args.tau_ms), tau_samples=int(tau_samp),
            sfreq=sfreq, backend="nice", nice_fn="SymbolicMutualInformation"
        )

        # Save
        out_name = f"{fpath.stem.replace('-epo','')}_wsmi_k{args.k}_tau{int(round(args.tau_ms))}ms.npz"
        out_path = run_dir / out_name
        np.savez_compressed(out_path, wsmi=wsmi.astype(np.float32), events=events, ch_names=ch_names, meta=json.dumps(meta))
        print(f"[OK] Saved -> {out_path}")
        render_progress(idx, total)

    if total: # Final flush for progress bar
        sys.stdout.flush() # Ensure progress bar is printed

if __name__ == "__main__":
    main()
