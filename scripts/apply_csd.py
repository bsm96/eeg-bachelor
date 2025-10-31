#!/usr/bin/env python
"""
Apply EEG Current Source Density (CSD; surface Laplacian) using MNE-Python.

This script applies CSD to preprocessed FIF files, typically epochs ("*-epo.fif")
or raw recordings ("*-raw.fif"). It is non-destructive: results are written to a
separate output directory with a "_csd" suffix, and existing outputs are skipped
unless --overwrite is set.

Usage examples (PowerShell):

  # Apply to epochs in data/epochs/run_v1
  python scripts/apply_csd.py --mode epochs \
    --input-dir data/epochs/run_v1 --montage standard_1020

  # Apply to raw FIF in data/Raws_new_ica
  python scripts/apply_csd.py --mode raw \
    --input-dir data/Raws_new_ica --glob "*.fif" --montage standard_1020

Notes:
- CSD requires a valid EEG montage. If files don't have a montage set, this
  script will attach a standard montage (default: standard_1020). If some EEG
  channel names aren't found in the montage, you can:
    - proceed and ignore missing positions (--allow-partial), or
    - drop EEG channels without known positions (--drop-missing).
- Only EEG channels are transformed; non-EEG channels are preserved.
- Parameters exposed: lambda2 (smoothing), stiffness (spline rigidity).

References:
- MNE-Python: mne.preprocessing.compute_current_source_density

"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mne
from mne.preprocessing import compute_current_source_density
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, **kwargs):  # type: ignore
        return x


@dataclass
class CSDParams:
    montage: str = "standard_1020"
    lambda2: float = 1e-5
    stiffness: int = 4
    allow_partial: bool = False
    drop_missing: bool = False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply EEG CSD (surface Laplacian) to FIF files")
    p.add_argument("--mode", choices=["epochs", "raw"], default="epochs",
                   help="File type to process: MNE Epochs or Raw FIF")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory to scan for input FIF files")
    p.add_argument("--glob", type=str, default=None,
                   help="Glob pattern to match files (default depends on --mode)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <input-dir>_csd)")
    p.add_argument("--montage", type=str, default="standard_1020",
                   help="Standard montage name (e.g., standard_1020, standard_1005)")
    p.add_argument("--lambda2", type=float, default=1e-5,
                   help="CSD smoothing parameter (lambda^2)")
    p.add_argument("--stiffness", type=int, default=4,
                   help="CSD spline stiffness")
    p.add_argument("--allow-partial", action="store_true",
                   help="Proceed even if some EEG channels are missing in the montage (ignored)")
    p.add_argument("--drop-missing", action="store_true",
                   help="Drop EEG channels that are not present in the montage before CSD")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write outputs; just print what would happen")
    args = p.parse_args(argv)

    if args.glob is None:
        args.glob = "*-epo.fif" if args.mode == "epochs" else "*-raw.fif"
    return args


def infer_out_dir(input_dir: Path, out_dir: Optional[Path]) -> Path:
    return out_dir if out_dir is not None else input_dir.with_name(input_dir.name + "_csd")


def find_files(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(input_dir.rglob(pattern))


def load_inst(path: Path, mode: str):
    if mode == "epochs":
        return mne.read_epochs(path, preload=True, verbose="ERROR")
    else:
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def eeg_channel_names(inst) -> List[str]:
    picks = mne.pick_types(inst.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
    return [inst.ch_names[i] for i in picks]


def ensure_montage(inst, montage_name: str, allow_partial: bool, drop_missing: bool) -> None:
    """Ensure an EEG montage is set on the instance.

    - If already set (has dig or montage), do nothing.
    - Otherwise, attach a standard montage. If some EEG channels are missing in the montage:
        * If drop_missing: drop those EEG channels.
        * Else if allow_partial: keep them (they won't be CSD-transformed properly).
        * Else: raise a RuntimeError.
    """
    # If montage already present and populated, keep it
    try:
        existing = inst.get_montage()
    except Exception:
        existing = None
    if existing is not None and getattr(existing, "ch_names", []):
        return

    mont = mne.channels.make_standard_montage(montage_name)
    eeg_names = eeg_channel_names(inst)
    in_mont = sorted([ch for ch in eeg_names if ch in mont.ch_names])
    missing = sorted([ch for ch in eeg_names if ch not in mont.ch_names])

    if missing:
        msg = (
            f"EEG channels missing in montage '{montage_name}': {len(missing)} / {len(eeg_names)}\n"
            f"Missing (first 20): {missing[:20]}"
        )
        if drop_missing:
            # Drop missing EEG channels before setting montage
            inst.pick([ch for ch in inst.ch_names if (ch not in missing)])
            # Recompute names after drop
            eeg_names = eeg_channel_names(inst)
            in_mont = sorted([ch for ch in eeg_names if ch in mont.ch_names])
            print(f"[montage] Dropped {len(missing)} EEG channels without positions.")
        elif allow_partial:
            print(f"[montage] Proceeding with partial montage; {msg}")
        else:
            raise RuntimeError(
                msg + "\nUse --allow-partial to proceed, or --drop-missing to remove them."
            )

    # Attach montage, ignoring any residual non-matching channels
    try:
        inst.set_montage(mont, on_missing="ignore")
    except TypeError:
        # Older MNE without on_missing
        inst.set_montage(mont)


def apply_csd_to_inst(inst, params: CSDParams):
    """Run CSD on a loaded Raw/Epochs instance and return the transformed copy.

    Only lambda2 and stiffness are passed for broad compatibility across MNE versions.
    """
    # Ensure montage is set or attach one
    ensure_montage(inst, params.montage, params.allow_partial, params.drop_missing)

    # Apply CSD; returns a copy by default
    csd = compute_current_source_density(inst, lambda2=float(params.lambda2), stiffness=int(params.stiffness))
    return csd


def make_out_path(out_dir: Path, src_file: Path) -> Path:
    name = src_file.name
    if name.endswith("-epo.fif"):
        name = name.replace("-epo.fif", "_csd-epo.fif")
    elif name.endswith("-raw.fif"):
        name = name.replace("-raw.fif", "_csd-raw.fif")
    else:
        stem, ext = src_file.stem, src_file.suffix
        name = f"{stem}_csd{ext}"
    return out_dir / name


def write_sidecar_json(out_path: Path, src_file: Path, params: CSDParams, inst) -> None:
    meta = {
        "source_file": str(src_file),
        "output_file": str(out_path),
        "n_channels": len(inst.ch_names),
        "sfreq": float(inst.info.get("sfreq", 0.0)),
        "params": asdict(params),
        "tool": "apply_csd.py",
        "mne_version": getattr(mne, "__version__", "unknown"),
    }
    with out_path.with_suffix(out_path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_dir: Path = args.input_dir.resolve()
    out_dir: Path = infer_out_dir(input_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CSDParams(
        montage=args.montage,
        lambda2=args.lambda2,
        stiffness=args.stiffness,
        allow_partial=bool(args.allow_partial),
        drop_missing=bool(args.drop_missing),
    )

    files = find_files(input_dir, args.glob)
    if not files:
        print(f"No files matched pattern '{args.glob}' under {input_dir}")
        return 1

    print(f"Mode: {args.mode}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {out_dir}  (exists: {out_dir.exists()})")
    print(f"Files to process: {len(files)} (pattern: {args.glob})")
    print(f"Parameters: montage={params.montage}, lambda2={params.lambda2}, stiffness={params.stiffness}, "
          f"allow_partial={params.allow_partial}, drop_missing={params.drop_missing}")
    if args.dry_run:
        print("[dry-run] Will not write files.")

    n_done = 0
    n_skipped = 0
    for src in tqdm(files, desc="CSD", unit="file"):
        out_path = make_out_path(out_dir, src)
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        try:
            inst = load_inst(src, args.mode)
        except Exception as e:
            print(f"[ERROR] Failed to read {src.name}: {e}")
            continue

        try:
            csd_inst = apply_csd_to_inst(inst, params)
        except Exception as e:
            print(f"[ERROR] Failed CSD for {src.name}: {e}")
            continue

        if not args.dry_run:
            try:
                csd_inst.save(out_path, overwrite=True)
                write_sidecar_json(out_path, src, params, csd_inst)
                n_done += 1
            except Exception as e:
                print(f"[ERROR] Failed to write {out_path.name}: {e}")
                continue

        # Proactively free memory for long loops
        del inst, csd_inst

    print(f"Done. Wrote: {n_done}, skipped existing: {n_skipped}, total matched: {len(files)}")
    return 0 if n_done > 0 or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
