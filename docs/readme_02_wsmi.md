# wSMI (NICE) — Short Guide

Compute per-epoch wSMI connectivity (SymbolicMutualInformation, weighted) from MNE Epochs (`*-epo.fif`). One NPZ is written per input with matrices shaped (E, C, C) and inline event metadata.

## Minimal example (PowerShell)
```powershell
python scripts/compute_wsmi.py `
  --input-dir data/epochs/events_tmin-0p2_tmax15 `
  --out-dir data/processed/wsmi `
  --k 3 `
  --tau-ms 8 `
  --overwrite
```

### Filter by specific event labels and tag outputs
You can restrict epochs to specific labels (from `epochs.event_id` keys) and tag the output folder name:

```powershell
python scripts/compute_wsmi.py `
  --input-dir data/epochs/events_tmin-0p2_tmax15 `
  --out-dir data/processed/wsmi `
  --k 3 `
  --tau-ms 8 `
  --include-labels "Familiar voice" "Medical staff" "Resting" `
  --subset-tag familiar_medical_resting `
  --overwrite
```

## Inputs
- `--input-dir` (required): Folder with `*-epo.fif`
- `--out-dir` (required): Root for outputs
- `--glob` (optional): Pattern for epoch files (default: `*-epo.fif`)
- `--k`, `--tau-ms` (optional): Kernel and delay in ms (converted to samples per file)
- `--overwrite` (optional): Replace existing NPZs
- `--include-labels` (optional): Only include epochs with these label names (repeat or space-separated)
- `--subset-tag` (optional): Append a tag to the output subfolder name (e.g., `events_..._mytag`)

## Output
- Folder: `data/processed/wsmi/<input-folder>[_<subset-tag>]_nice_csd/k<k>_tau<tau_ms>ms/`
- File: `<stem>_wsmi_k<k>_tau<tau_ms>ms.npz`
- Keys inside each NPZ:
  - `wsmi`: float32, shape `(E, C, C)`
  - `events`: int64, shape `(E,)`
  - `event_labels`: str array, shape `(E,)`
  - `event_codes`: unique int64 codes
  - `event_names`: unique str labels
  - `ch_names`: channel names (dtype=object)
  - `meta_json`: JSON string with `k`, `tau_ms`, `sfreq`, `csd_applied`, `source_epochs`
    - When filtering is used: includes `filtered` (`include_labels`, `E_before`, `E_after`) and `subset_tag` if provided

Note: NICE applies current source density internally; `tau-ms` is converted to integer samples using each file’s sampling rate.