# Commands cheat sheet

Below are copyable examples to generate MNE epochs under `data/epochs` using the new CLI script.

Output location:
- Saved epochs are organized under `data/epochs/<spec>/` where `<spec>` summarizes the main criteria.
- Examples:
    - Sliding: `data/epochs/window16s_stride16s/subject123-epo.fif`
    - Events: `data/epochs/tmin-0p2_tmax15/subject123-epo.fif`
- Existing files: By default, existing files are skipped. Use `--overwrite` to force re-writing.

## Event-locked epochs (C&K-like)

This matches the common setup with a short pre-event baseline and a 15 s post-onset window. It searches preprocessed FIF files under `data/Raws_new_ica/` and writes `.fif` epochs to `data/epochs/`.

```powershell
python scripts/make_epochs.py `
    --input-dir "data/Raws_new_ica" `
    --glob "*.fif" `
    --out-dir "data/epochs" `
    --mode events `
    --events "Resting,Familiar voice,Medical staff" `
    --tmin -0.2 `
    --tmax 15 `
    --sfreq 250 `
    --l-freq 0.1 `
    --h-freq 45 `
    --picks eeg
```
### Della Bella method (almost)
python scripts/make_epochs.py `
    --input-dir "data/Raws_new_ica" `
    --glob "*.fif" `
    --out-dir "data/epochs" `
    --mode events `
    --events "Resting,Familiar voice,Medical staff" `
    --l-freq 0.1 `
    --h-freq 45 `
    --picks eeg

python scripts/make_epochs.py `
    --input-dir "data/Raws_new_ica" `
    --glob "*.fif" `
    --out-dir "data/epochs" `
    --mode events `
    --events "Resting,Familiar voice,Medical staff" `
    --picks eeg


Notes:
- `--tmin -0.2 --tmax 15` reproduces the typical C&K event window.
- By default, the script does not filter, notch, or resample; it keeps the data as-is unless flags are provided.
- The example above assumes you start from preprocessed FIF under `data/Raws_new_ica`.
- If starting from raw EDF instead, see the alternative below.

Alternative (from raw EDF):

```powershell
python scripts/make_epochs.py `
    --input-dir "EDF filer" `
    --glob "*.edf" `
    --out-dir "data/epochs" `
    --mode events `
    --events "Resting,Familiar voice,Medical staff" `
    --tmin -0.2 `
    --tmax 15 `
    --sfreq 250 `
    --l-freq 0.1 `
    --h-freq 45 `
    --use-notch `
    --notch-freqs 50 `
    --reject-annot BAD_ `
    --picks eeg `
    --rename-eeg
```

## Sliding-window epochs

This creates fixed-length windows with a chosen stride across continuous data. Adjust `--window` and `--stride` as needed.

```powershell
python scripts/make_epochs.py `
    --input-dir "data/Raws_new_ica" `
    --glob "*.fif" `
    --out-dir "data/epochs" `
    --mode sliding `
    --window 2.5 `
    --stride 2.5 `
    --picks eeg
```

## PowerShell line continuation

In Windows PowerShell, use the backtick character (`) at the end of a line to continue a long command on the next line.

Notes:
- The backtick must be the very last character on the line (no trailing spaces or tabs).
- The caret (^) is a CMD.exe continuation, not PowerShell.
- The backslash (\) is a Unix/bash convention and won’t work for line continuation in PowerShell.
- Alternatively, keep the command on a single line (see the VS Code terminal examples below).

Tips:
- Use `--start-offset` and `--stop-offset` to crop the raw span before sliding windowing (e.g., skip the first 60 s of long resting periods).
- Use `--glob "*.fif"` if starting from already preprocessed FIF files.

## VS Code terminal (repo root, conda env active)

When the terminal is opened at the repository root and a conda environment is already active (e.g., `conda activate eeg_env`), run the same commands without path adjustments. Examples:

- Event-locked (from preprocessed FIF in `data/Raws_new_ica`):

```powershell
python scripts/make_epochs.py --input-dir "data/Raws_new_ica" --glob "*.fif" --out-dir "data/epochs" --mode events --events "Resting,Familiar voice,Medical staff" --tmin -0.2 --tmax 15 --picks eeg
```

- Sliding windows (from preprocessed FIF):

```powershell
python scripts/make_epochs.py --input-dir "data/Raws_new_ica" --glob "*.fif" --out-dir "data/epochs" --mode sliding --window 2.5 --stride 2.5 --picks eeg
```

- Alternative (from raw EDF):

```powershell
python scripts/make_epochs.py --input-dir "EDF filer" --glob "*.edf" --out-dir "data/epochs" --mode events --events "Resting,Familiar voice,Medical staff" --tmin -0.2 --tmax 15 --sfreq 250 --l-freq 0.1 --h-freq 45 --use-notch --notch-freqs 50 --reject-annot BAD_ --picks eeg --rename-eeg
```

Note:
- If `python` does not point to the expected interpreter, use `where python` (Windows) to verify, or run explicitly via `conda run -n eeg_env python ...`.


## Current Source Density (CSD)

Apply EEG current source density (surface Laplacian) using MNE. Results are written non-destructively to a sibling folder with a `_csd` suffix. Existing outputs are skipped unless `--overwrite` is provided.

Output naming:
- Epochs: `*_csd-epo.fif` written under `<input-dir>_csd/`
- Raw:    `*_csd-raw.fif` written under `<input-dir>_csd/`

### CSD for epochs
# it overrides if the csd already exists

```powershell
python scripts/apply_csd.py `
    --mode epochs `
    --input-dir "data\epochs\window15s_stride15s" `
    --montage standard_1020
```

### CSD for raw FIF (preprocessed)

```powershell
python scripts/apply_csd.py `
    --mode raw `
    --input-dir "data/Raws_new_ica" `
    --glob "*.fif" `
    --montage standard_1020
```

Advanced flags:
- `--montage standard_1020` : international 10-20 system
- `--allow-partial`: proceed even if some EEG channels are missing in the montage (kept but won’t be CSD-transformed properly)
- `--drop-missing`: drop EEG channels that are not present in the montage before CSD
- `--lambda2 1e-5`, `--stiffness 4`: CSD parameters you can tune
- `--overwrite`: rewrite outputs if already present
- `--dry-run`: print actions without writing files

#### One-line variants (VS Code terminal, env active)

```powershell
python scripts/apply_csd.py --mode epochs --input-dir "data/epochs/run_v1" --montage standard_1020
```

```powershell
python scripts/apply_csd.py --mode raw --input-dir "data/Raws_new_ica" --glob "*.fif" --montage standard_1020
```

