#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal wSMI runner with NICE (SymbolicMutualInformation, weighted).
- Input:  MNE Epochs FIF files (event-locked), e.g. "*-epo.fif"
- CSD:    Handled internally by NICE (no switches here)
- Output: One .npz per input with wsmi (E,C,C) and inline event metadata
"""

from pathlib import Path
import argparse, json, sys, logging
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
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ outputs if present.")
    ap.add_argument(
        "--include-labels",
        nargs="+",
        help="Only include epochs whose label names (from epochs.event_id keys) are in this list. Example: --include-labels 'Familiar voice' 'Medical staff' 'Resting'",
    )
    ap.add_argument(
        "--subset-tag",
        type=str,
        default=None,
        help="Optional tag to include in the output folder name to describe the subset (e.g., familiar_medical_resting).",
    )
    args = ap.parse_args()

    # ---- Logging ----
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("compute_wsmi")

    # ---- Output folder ----
    subdir_name = args.input_dir.name
    if args.subset_tag:
        # Append a single-underscore separated tag to make the subset explicit
        subdir_name = f"{subdir_name}_{args.subset_tag}"
    # Append NICE+CSD marker at the end of the parent subdir
    subdir_name = f"{subdir_name}_nice_csd"
    # Keep k/tau folder simple without repeating the nice_csd suffix
    run_dir = args.out_dir / subdir_name / f"k{args.k}_tau{int(round(args.tau_ms))}ms"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process files ----
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        log.warning(f"No matches in {args.input_dir} with glob '{args.glob}'")
        return

    total = len(files)

    # ---- Progress bar function ----
    def render_progress(current: int, total: int, width: int = 30) -> None:
        ratio = current / total
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        log.info(f"|{bar}| {current}/{total}")

    for idx, fpath in enumerate(files, start=1):
        # Derive output path early to support overwrite checks before heavy computation
        out_name = f"{fpath.stem.replace('-epo','')}_wsmi_k{args.k}_tau{int(round(args.tau_ms))}ms.npz"
        out_path = run_dir / out_name

        if out_path.exists() and not args.overwrite:
            log.info(f"Skip (exists): {out_path}")
            render_progress(idx, total)
            continue

        log.info(f"Load epochs: {fpath}")
        epochs = mne.read_epochs(fpath, preload=True, verbose="ERROR")

        # Convert tau from ms to integer samples
        sfreq = float(epochs.info["sfreq"])
        tau_samp = int(round((args.tau_ms / 1000.0) * sfreq))

        # Optional filtering by event label names
        filtered_info = None
        if args.include_labels:
            # Map label names to event codes using epochs.event_id
            event_id = epochs.event_id or {}
            wanted_labels = [str(x) for x in args.include_labels]
            wanted_codes = [int(event_id[l]) for l in wanted_labels if l in event_id]
            E_before = len(epochs)
            if wanted_codes:
                mask = np.isin(epochs.events[:, 2], np.array(wanted_codes, dtype=int))
                epochs = epochs[mask]
            else:
                # No matching labels found in this file -> produce zero epochs
                epochs = epochs[[]]
            E_after = len(epochs)
            if E_after == 0:
                log.info(f"Skip (no matching epochs after filter {wanted_labels}): {fpath}")
                render_progress(idx, total)
                continue
            filtered_info = {
                "include_labels": wanted_labels,
                "E_before": int(E_before),
                "E_after": int(E_after),
            }

        # Compute wSMI with NICE (may return (C,C,E))
        # method_params kept explicit to record CSD behavior in metadata
        method_params = {"nthreads": "auto"}
        smi = SymbolicMutualInformation(
            tmin=None, tmax=None, kernel=args.k, tau=tau_samp,  # full epoch
            method='weighted', backend='python',
            method_params=method_params, comment='weighted'
        )
        smi.fit(epochs)
        wsmi = smi.data_  # Extract data array
        wsmi = np.asarray(wsmi)
        if wsmi.shape[-1] == len(epochs):  # (C,C,E) -> (E,C,C)
            wsmi = np.moveaxis(wsmi, -1, 0)  # Move epochs to first dim

        # ---- Event metadata ----
        # Build mapping from numeric code to label name using epochs.event_id
        id2label = {int(v): str(k) for k, v in (epochs.event_id or {}).items()}
        # Numeric event codes per epoch (shape: E,)
        events = epochs.events[:, 2].astype(np.int64)
        # Labels per epoch derived from mapping; fallback to stringified code
        if id2label:
            event_labels = np.asarray([id2label.get(int(e), str(int(e))) for e in events], dtype=np.unicode_)
        else:
            event_labels = np.asarray([str(int(e)) for e in events], dtype=np.unicode_)
        # Unique codes and corresponding unique label names
        codes_unique = np.unique(events)
        if id2label:
            names_unique = np.asarray([id2label.get(int(c), str(int(c))) for c in codes_unique], dtype=np.unicode_)
        else:
            names_unique = np.asarray([str(int(c)) for c in codes_unique], dtype=np.unicode_)

        # ---- Metadata JSON ----
        k = int(args.k)
        tau_ms = float(args.tau_ms)
        epo_path = fpath
        meta = {
            "k": k,
            "tau_ms": tau_ms,
            "sfreq": float(epochs.info["sfreq"]),
            "csd_applied": (not method_params.get("bypass_csd", False)),
            "source_epochs": str(epo_path),
        }
        if filtered_info is not None:
            meta["filtered"] = filtered_info
        if args.subset_tag:
            meta["subset_tag"] = str(args.subset_tag)
        meta_json = json.dumps(meta)

        # Channel names
        # Save as Unicode array (dtype 'U') to allow loading with allow_pickle=False downstream
        ch_names = np.asarray(list(map(str, epochs.ch_names)), dtype=np.unicode_)

        # ---- Save NPZ ----
        np.savez_compressed(
            out_path,
            wsmi=wsmi.astype(np.float32),
            events=events,
            event_labels=event_labels,
            event_codes=codes_unique,
            event_names=names_unique,
            ch_names=ch_names,
            meta_json=meta_json,
        )
        log.info(f"Saved: {out_path}")
        render_progress(idx, total)

    if total: # Final flush for progress bar
        sys.stdout.flush() # Ensure progress bar is printed

if __name__ == "__main__":
    main()
