# 01_make_epochs.py
# Purpose: make fixed-length epochs from preprocessed FIF files.
# Now supports CLI flags for --epoch-length, --stride, --in-dir, --out-dir, and --suffix.
# Note: do NOT re-run autoreject/ICA here; only slice the already cleaned data.

from pathlib import Path
import argparse
import json
import sys
from time import perf_counter
import mne
from tqdm import tqdm

# --- Defaults (can be overridden via CLI) ---
DEFAULT_IN_DIR = Path("data") / "used_raws"               # folder with preprocessed *_raw.fif
DEFAULT_EPOCH_LENGTH_S = 16.0                              # seconds per epoch
DEFAULT_STRIDE_S = 1.0                                     # seconds between starts -> 15 s overlap
DEFAULT_SUFFIX = "window16s_stride1s"                      # explicit naming for default config
REJECT_BY_ANNOTATION = True                                # honor bad/annotated segments from preprocessing
EXPECTED_SFREQ = 250.0                                     # dataset harmonized to 250 Hz

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make fixed-length epochs from preprocessed *_raw.fif files.")
    ap.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR), help="Folder with *_raw.fif files")
    ap.add_argument("--out-dir", type=str, default="data/epochs", help="Output root folder for epochs")
    ap.add_argument("--epoch-length", type=float, default=DEFAULT_EPOCH_LENGTH_S, help="Epoch length in seconds")
    ap.add_argument("--stride", type=float, default=DEFAULT_STRIDE_S, help="Stride in seconds between epoch starts")
    ap.add_argument("--suffix", type=str, default=None, help="Optional suffix for the output subfolder; if omitted, derived from epoch/stride")
    return ap.parse_args()


def _derive_suffix(epoch_length_s: float, stride_s: float, explicit: str | None) -> str:
    if isinstance(explicit, str) and explicit:
        return explicit
    # Use compact ints when possible (15.0 -> 15)
    el = int(epoch_length_s) if abs(epoch_length_s - int(epoch_length_s)) < 1e-9 else epoch_length_s
    st = int(stride_s) if abs(stride_s - int(stride_s)) < 1e-9 else stride_s
    return f"window{el}s_stride{st}s"


def main() -> int:
    args = _parse_args()

    in_dir = Path(args.in_dir)
    suffix = _derive_suffix(args.epoch_length, args.stride, args.suffix)
    out_dir = Path(args.out_dir) / suffix

    out_dir.mkdir(parents=True, exist_ok=True)

    fif_files = sorted(in_dir.glob("*.fif"))
    if not fif_files:
        print(f"[WARN] No FIF files found in: {in_dir}")
        return 1

    summary = []
    for fif in tqdm(fif_files, desc="Epoching files", unit="file"):
        start = perf_counter()
        try:
            # Load preprocessed recording
            raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)

            # Pick EEG channels up front (compatibility across MNE versions)
            picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False)
            raw = raw.copy().pick(picks)

            # Sanity check on sampling frequency
            sfreq = float(raw.info["sfreq"])
            if abs(sfreq - EXPECTED_SFREQ) > 1e-6:
                tqdm.write(f"[NOTE] {fif.name}: sfreq is {sfreq} Hz (expected {EXPECTED_SFREQ}). Continuing.")

            # Make fixed-length epochs; overlap = duration - stride
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=float(args.epoch_length),
                overlap=float(args.epoch_length) - float(args.stride),
                reject_by_annotation=REJECT_BY_ANNOTATION,
                preload=True,
                verbose=False,
            )

            if len(epochs) == 0:
                tqdm.write(f"[WARN] {fif.name}: produced 0 epochs (all time annotated as bad?). Skipping save.")
                continue

            # Stable and explicit filename; drop trailing "_raw" if present
            stem = fif.stem[:-4] if fif.stem.endswith("_raw") else fif.stem
            out_fif = out_dir / f"{stem}_{suffix}-epo.fif"

            # Save epochs; allow overwrite so re-runs are idempotent
            epochs.save(out_fif, overwrite=True)

            elapsed = perf_counter() - start
            tqdm.write(f"[OK] {out_fif.name}  | n_epochs={len(epochs)}  n_channels={len(epochs.ch_names)}  ({elapsed:.1f}s)")

            # Compact summary for later QC/logging
            summary.append({
                "file": fif.name,
                "out_file": out_fif.name,
                "n_channels": len(epochs.ch_names),
                "sfreq": sfreq,
                "epoch_length_s": float(args.epoch_length),
                "stride_s": float(args.stride),
                "reject_by_annotation": REJECT_BY_ANNOTATION,
                "n_epochs": int(len(epochs)),
                "duration_total_min": round(raw.n_times / sfreq / 60.0, 2),
                "elapsed_s": round(elapsed, 2),
            })

        except Exception as exc:
            tqdm.write(f"[ERROR] Failed on {fif.name}: {exc}")
            continue

    # Persist exact settings used
    cfg_used = {
        "epoch_length_s": float(args.epoch_length),
        "stride_s": float(args.stride),
        "reject_by_annotation": REJECT_BY_ANNOTATION,
        "expected_sfreq": EXPECTED_SFREQ,
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "suffix": suffix,
    }
    (out_dir / "epochs_config_used.json").write_text(json.dumps(cfg_used, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    tqdm.write(f"\n[DONE] Epochs written to: {out_dir}")
    tqdm.write(f"[INFO] Config saved to:   {out_dir / 'epochs_config_used.json'}")
    tqdm.write(f"[INFO] Summary saved to:  {out_dir / 'summary.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
