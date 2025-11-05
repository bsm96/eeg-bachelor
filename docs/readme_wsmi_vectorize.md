# wSMI Vectorization — Short Guide

Converts per-epoch wSMI cubes (E, C, C) into feature matrices (E, P) using the upper triangle (no diagonal). Writes per-run `pairs_idx.npy`, `pairs.json` (if channel names exist), `manifest.csv`, and per-file `<name>_vec.npz`.

## Inputs

- `--input-dir` (required): Folder with `.npz` files from the wSMI step
- `--out-dir` (required): Output folder
- `--glob` (optional): File pattern (default `*.npz`)
- `--overwrite` (optional): Replace existing outputs

## Example

```powershell
python scripts/vectorize_wsmi.py `
  --input-dir "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\k3_tau8ms" `
  --out-dir   "data\processed\wsmi\events_tmin-0p2_tmax15_familiar_medical_resting_nice_csd\k3_tau8ms_vec" `
  --glob "*.npz" `
  --overwrite
```

## Notes

- Expects key `wsmi` with shape (E, C, C); fails fast on invalid inputs.
- Loads with `allow_pickle=False` only. To get `pairs.json` with names and preserve metadata, ensure the compute step saves strings (ch_names, event_labels, event_names) as Unicode arrays (not object).
- Outputs in `--out-dir`:
  - `pairs_idx.npy`: shape (2, P) with upper-tri indices
  - `pairs.json`: optional, channel-name pairs in column order
  - `manifest.csv`: vec_file,source_file,E,C,P
  - `<name>_vec.npz`: contains X (E, P), E, C, P, source_file, and any preserved metadata (events, event_labels, event_codes, event_names, ch_names, meta_json)
