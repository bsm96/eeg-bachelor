# 01_make_epochs.py
# Purpose: make fixed-length epochs (16 s, stride 1 s) from preprocessed FIF files.
# Note: do NOT re-run autoreject/ICA here; only slice the already cleaned data.

from pathlib import Path
import json
import sys
from time import perf_counter
import mne
from tqdm import tqdm

# --- Config (kept inline for simplicity) ---
IN_DIR = Path("data") / "used_raws"                       # folder with preprocessed *_raw.fif
SUFFIX = "window16s_stride1s"                             # professional, explicit naming
OUT_DIR = Path("data") / "epochs" / SUFFIX                # output folder for epoch files
EPOCH_LENGTH_S = 16.0                                     # seconds per epoch
STRIDE_S = 1.0                                            # seconds between starts -> 15 s overlap
REJECT_BY_ANNOTATION = True                               # honor bad/annotated segments from preprocessing
EXPECTED_SFREQ = 250.0                                    # dataset harmonized to 250 Hz

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fif_files = sorted(IN_DIR.glob("*.fif"))
    if not fif_files:
        print(f"[WARN] No FIF files found in: {IN_DIR}")
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
                duration=EPOCH_LENGTH_S,
                overlap=EPOCH_LENGTH_S - STRIDE_S,
                reject_by_annotation=REJECT_BY_ANNOTATION,
                preload=True,
                verbose=False,
            )

            if len(epochs) == 0:
                tqdm.write(f"[WARN] {fif.name}: produced 0 epochs (all time annotated as bad?). Skipping save.")
                continue

            # Stable and explicit filename; drop trailing "_raw" if present
            stem = fif.stem[:-4] if fif.stem.endswith("_raw") else fif.stem
            out_fif = OUT_DIR / f"{stem}_{SUFFIX}-epo.fif"

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
                "epoch_length_s": EPOCH_LENGTH_S,
                "stride_s": STRIDE_S,
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
        "epoch_length_s": EPOCH_LENGTH_S,
        "stride_s": STRIDE_S,
        "reject_by_annotation": REJECT_BY_ANNOTATION,
        "expected_sfreq": EXPECTED_SFREQ,
        "in_dir": str(IN_DIR),
        "out_dir": str(OUT_DIR),
        "suffix": SUFFIX,
    }
    (OUT_DIR / "epochs_config_used.json").write_text(json.dumps(cfg_used, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    tqdm.write(f"\n[DONE] Epochs written to: {OUT_DIR}")
    tqdm.write(f"[INFO] Config saved to:   {OUT_DIR / 'epochs_config_used.json'}")
    tqdm.write(f"[INFO] Summary saved to:  {OUT_DIR / 'summary.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
