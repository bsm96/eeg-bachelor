#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-epoch weighted Symbolic Mutual Information (wSMI) matrices with NICE library(NICE-only).

- Input:  MNE Epochs FIF files (event-locked)
- CSD:    By default, NICE computes CSD internally. If your inputs are already CSD-transformed,
          pass --assume-csd to bypass NICE's internal CSD step (no double CSD).
- Output: .npz per input file with wsmi (n_epochs, n_channels, n_channels) + metadata
- UX:     • Progress bar over files
          • Suppress the misleading "Computing CSD" line when --assume-csd is set
          • Run-folder suffix encodes who applied CSD: _csd_nice / _csd_mne / _nocsd

Examples (PowerShell):
  # Inputs WITHOUT CSD (let NICE apply CSD internally)
  python scripts/compute_wsmi.py `
    --input-dir data/epochs/events_tmin-0p2_tmax15 `
    --glob "*-epo.fif" `
    --k 3 --tau-ms 8 `
    --out-dir data/processed/wsmi

  # Inputs ALREADY CSD-transformed (e.g., *_csd-epo.fif) → bypass internal CSD
  python scripts/compute_wsmi.py `
    --input-dir data/epochs/events_tmin-0p2_tmax15_csd `
    --glob "*_csd-epo.fif" `
    --k 3 --tau-ms 8 `
    --out-dir data/processed/wsmi `
    --assume-csd
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np
import mne

# progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **k):  # type: ignore
        return x

# Require NICE library, so checking for that
try:
    from nice.markers.connectivity import epochs_compute_wsmi  # type: ignore
except Exception:
    raise SystemExit(
        "NICE is required. Install in your active environment, e.g.:\n"
        "  python -m pip install --no-build-isolation git+https://github.com/nice-tools/nice\n"
        "If you see MNE import errors inside NICE, use: conda install -c conda-forge 'mne<1.7'\n"
    )


def _str_unique_channel_types(epochs: mne.Epochs) -> List[str]:
    """Return sorted unique channel types for provenance."""
    try:
        types = epochs.get_channel_types()  # e.g., 'eeg', 'eog', 'stim', 'csd', 'misc', …
        return sorted(set(types))
    except Exception:  # pragma: no cover
        return []


@contextmanager
def _suppress_csd_line(active: bool):
    """
    Suppress the *misleading* 'Computing CSD' line that NICE prints,
    but only when --assume-csd is True (i.e., we bypass CSD).
    Other output lines are replayed verbatim.
    """
    if not active:
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = buf_out, buf_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        # dont reprint "Computing CSD" lines if it is not computing CSD
        for line in buf_out.getvalue().splitlines():
            if "Computing CSD" in line:
                continue
            print(line)
        for line in buf_err.getvalue().splitlines():
            if "Computing CSD" in line:
                continue
            print(line, file=sys.stderr)


def _infer_csd_tag(assume_csd: bool, no_csd: bool, ch_types_before: List[str]) -> str:
    """
    Decide how to tag the run folder:
      - 'csd_mne'  : inputs were already CSD (assume external/MNE CSD)
      - 'csd_nice' : NICE applies CSD internally
      - 'no_csd'   : no CSD applied at all (raw data)
    """
    if no_csd:
        return "no_csd"
    if assume_csd:
        return "csd_mne"
    return "csd_nice"


def parse_args():
    ap = argparse.ArgumentParser(description="Compute per-epoch wSMI matrices using NICE (no reduction).")
    ap.add_argument("--input-dir", required=True, type=Path, help="Directory containing Epochs FIF files")
    ap.add_argument("--glob", default="*-epo.fif", help="Glob for input files (default: '*-epo.fif')")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output root directory (npz files will be created here)")
    ap.add_argument("--k", type=int, default=3, help="Embedding order k (default: 3)")
    ap.add_argument("--tau-ms", type=float, default=8.0, help="Delay τ in milliseconds (default: 8)")
    ap.add_argument("--montage", default=None, help="Optional montage name (e.g., 'standard_1020') for provenance")
    ap.add_argument("--pick-eeg-only", action="store_true",
                    help="Attempt to keep only EEG channels (safe fallback if none found)")
    ap.add_argument("--assume-csd", action="store_true",
                    help="Inputs are already CSD-transformed; bypass NICE's internal CSD")
    ap.add_argument("--no-csd", action="store_true",
                    help="Skip CSD entirely (both MNE and NICE); compute wSMI on raw data")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return ap.parse_args()


def main():
    args = parse_args()

    in_dir: Path = args.input_dir
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"[WARN] No files matched in {in_dir} with glob '{args.glob}'")
        return

    # Always use events_tmin-0p2_tmax15_csd as the base output folder
    run_root = out_root / "events_tmin-0p2_tmax15_csd"

    # Iterate with a progress bar
    pbar = tqdm(files, desc="wSMI (files)", unit="file")
    run_dir = None  # will be set on first iteration when we know csd_tag

    for f_i, fpath in enumerate(pbar, start=1):
        pbar.set_postfix_str(fpath.name)

        # Read epochs
        epochs = mne.read_epochs(fpath, preload=True, verbose="ERROR")

        # Optional: set montage (provenance only; not required for NICE to run)
        if args.montage:
            try:
                epochs.set_montage(args.montage, match_case=False, on_missing="warn")
            except Exception as e:
                print(f"[WARN] montage '{args.montage}' failed on {fpath.name}: {e}")

        # Optional: keep EEG channels only (but safely fall back if none labeled 'eeg')
        if args.pick_eeg_only:
            ctypes = epochs.get_channel_types()
            eeg_idx = [i for i, t in enumerate(ctypes) if t == "eeg"]
            if len(eeg_idx) > 0:
                epochs.pick(eeg_idx)
            else:
                # keep non-artifact channels; drop stim/eog/ecg if present
                keep = [i for i, t in enumerate(ctypes) if t not in ("stim", "eog", "ecg")]
                epochs.pick(keep)
                print("[WARN] No channels typed as 'eeg'; kept non-stim/eog/ecg channels instead.")

        # Compute tau in samples
        sfreq = float(epochs.info["sfreq"])
        tau_samp = int(round((args.tau_ms / 1000.0) * sfreq))
        if tau_samp < 1:
            raise ValueError(f"tau-ms={args.tau_ms} too small for sfreq={sfreq} Hz → tau_samples={tau_samp}")

        # Decide CSD tag once (first file)
        ch_types_before = _str_unique_channel_types(epochs)
        if run_dir is None:
            csd_tag = _infer_csd_tag(args.assume_csd, args.no_csd, ch_types_before)
            run_dir = run_root / f"k{args.k}_tau{int(round(args.tau_ms))}ms_{csd_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] CSD mode: {csd_tag} | channel types (first file): {ch_types_before}")
        else:
            # For subsequent files, recompute csd_tag to use in filename
            csd_tag = _infer_csd_tag(args.assume_csd, args.no_csd, ch_types_before)

        # Build output filename - include CSD tag in patient filename
        base = fpath.stem.replace("-epo", "").replace("_csd", "")
        out_path = run_dir / f"{base}_{csd_tag}_wsmi_k{args.k}_tau{int(round(args.tau_ms))}ms.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists → {out_path}")
            continue

        # ---- wSMI via NICE ----
        # Suppress misleading "Computing CSD" line if we bypass CSD
        with _suppress_csd_line(active=(args.assume_csd or args.no_csd)):
            if args.assume_csd or args.no_csd:
                import mne.preprocessing as _mp
                _orig_csd = _mp.compute_current_source_density
                _mp.compute_current_source_density = lambda inst, **kw: inst  # no-op
                try:
                    wsmi_out = epochs_compute_wsmi(epochs, kernel=args.k, tau=tau_samp)
                finally:
                    _mp.compute_current_source_density = _orig_csd
                nice_csd_bypassed = True
            else:
                wsmi_out = epochs_compute_wsmi(epochs, kernel=args.k, tau=tau_samp)
                nice_csd_bypassed = False

        # ---- Axis handling: NICE may return (n_ch, n_ch, n_epochs); convert to (n_epochs, n_ch, n_ch) ----
        wsmi_arr = wsmi_out[0] if isinstance(wsmi_out, (tuple, list)) else wsmi_out
        wsmi = np.asarray(wsmi_arr)
        if wsmi.ndim != 3:
            raise RuntimeError(f"Unexpected wSMI ndim={wsmi.ndim}; expected 3D array.")

        E = len(epochs)
        if wsmi.shape[0] == E:
            wsmi_axis = "epochs,channels,channels"
        elif wsmi.shape[-1] == E and wsmi.shape[0] == wsmi.shape[1]:
            wsmi = np.moveaxis(wsmi, -1, 0)  # (C,C,E) -> (E,C,C)
            wsmi_axis = "channels,channels,epochs -> corrected to epochs,channels,channels"
        else:
            raise RuntimeError(
                f"Unexpected wSMI shape {wsmi.shape}; cannot infer epochs axis (E={E})."
            )

        if wsmi.shape[1] != wsmi.shape[2]:
            raise RuntimeError(f"wSMI channel dims differ: {wsmi.shape[1]} vs {wsmi.shape[2]}")

        # Event labels (from metadata if available)
        if epochs.metadata is not None and "Event" in epochs.metadata.columns:
            events = epochs.metadata["Event"].astype(str).to_numpy()
        else:
            # fallback to numeric event codes (3rd col of epochs.events)
            events = np.array([str(e) for e in getattr(epochs, "events", np.zeros((len(epochs), 3)))[:, 2]])

        meta = dict(
            k=int(args.k),
            tau_ms=float(args.tau_ms),
            tau_samples=int(tau_samp),
            sfreq=sfreq,
            backend="nice",
            nice_fn="nice.markers.connectivity.epochs_compute_wsmi",
            nice_csd_bypassed=bool(nice_csd_bypassed),
            csd_tag=_infer_csd_tag(args.assume_csd, args.no_csd, ch_types_before),
            pre_csd_input=ch_types_before,     # e.g., ['eeg'] or ['csd'] or ['misc', ...]
            montage=args.montage,
            input_path=str(fpath),
            input_wsmi_shape=tuple(np.asarray(wsmi_arr).shape),
            wsmi_axis_order=wsmi_axis,
        )

        np.savez_compressed(
            out_path,
            wsmi=wsmi.astype(np.float32, copy=False),
            events=events,
            ch_names=np.array(epochs.ch_names, dtype=object),
            meta=json.dumps(meta),
        )
        print(f"[OK] Saved → {out_path}")

    if run_dir is None:
        print("[DONE] No outputs (nothing matched).")
    else:
        print(f"[DONE] wrote results under {run_dir.parent}")


if __name__ == "__main__":
    main()
