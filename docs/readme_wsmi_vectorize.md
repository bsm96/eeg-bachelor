# wSMI Vectorization — Short Guide

Converts wSMI connectivity cubes (E, C, C) into feature matrices (E, P) by flattening the upper triangle (no diagonal). Produces per-run `pairs_idx.npy`, `pairs.json` (if channel names exist), `manifest.csv`, and one `<name>_vec.npz` per input.

## Inputs

- `--input-dir` (required): Folder with `.npz` files (typically from the wSMI step)
- `--out-dir` (required): Output folder
- `--glob` (optional): File pattern (default `*.npz`)
- `--overwrite` (optional): Replace existing outputs

## Example

```powershell
python scripts/vectorize_wsmi.py `
  --input-dir data/processed/wsmi/events_tmin-0p2_tmax15/k3_tau8ms_csd_nice `
  --out-dir data/processed/wsmi/events_tmin-0p2_tmax15/k3_tau8ms_csd_nice_vec
```
