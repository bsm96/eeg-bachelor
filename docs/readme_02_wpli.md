# 02_compute_wpli (wPLI)

Computes sensor-space wPLI connectivity in a frequency band (default: alpha 8–13 Hz) for `*-epo.fif` epoch files and saves one connectivity matrix per input file.

The current implementation lives in `scripts/extra_compute_wpli.py` and is configured via CLI arguments (paths are not hardcoded).

## Example usage (bash)
```bash
python scripts/extra_compute_wpli.py \
  --input-dir "data/epochs/sliding_window16s_stride1s" \
  --glob "*-epo.fif" \
  --out-dir "data/processed" \
  --fmin 8 \
  --fmax 13 \
  --n-jobs -2 \
  --overwrite
```

## Input

- Any folder you pass via `--input-dir`, with files matching `--glob` (default: `*-epo.fif`).

## Output

- Saved under:
  - `<out-dir>/<subset-tag>/wpli_<fmin>-<fmax>Hz/<name>_wpli_<fmin>-<fmax>Hz.npz` (if `--subset-tag` is provided)
  - otherwise `<out-dir>/wpli_<fmin>-<fmax>Hz/<name>_wpli_<fmin>-<fmax>Hz.npz`

Each `.npz` contains:
- `wpli`: wPLI matrix (channels × channels)
- `freqs`: frequency vector used by `mne-connectivity` (usually length 1 when `faverage=True`)
- `ch_names`: channel names