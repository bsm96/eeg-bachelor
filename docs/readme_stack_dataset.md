# Stacking wSMI Feature Files (Short Guide)

Create one dataset NPZ by stacking multiple `*_vec.npz` files (from vectorize_wsmi.py) and attaching labels from a CSV.

## Inputs
- `--labels-csv` (required): CSV with columns: `source_file,subject,condition`
  - `source_file` must match the basename of the vectorized file’s `source_file` (without directories)
  - Example row: `patient36_wsmi.npz,36,task`
- `--input-dir` (optional): Directory containing `*_vec.npz` files
  - If omitted, the script tries to infer a single vectorization folder under `data/processed/wsmi/`
- `--glob` (optional): Pattern to select files within `--input-dir` (default: `*_vec.npz`)
- `--out-file` (optional): Output NPZ path (default: `data/processed/datasets/<lastdir>_stack.npz`)

## What it does
- Loads each `*_vec.npz` (`X` shape: E×P) and checks that P is constant across files
- Looks up `subject` and `condition` from `--labels-csv` using the file basename
- Stacks all epochs to build one dataset and records per-file indices
- Finds and stores references to `pairs_idx.npy` and `pairs.json` if available

## Output (NPZ keys)
- `X` (float32, N×P) — all epochs stacked
- `y_cond` (int32, N) — encoded condition labels
- `groups` (int32, N) — subject IDs for group/LOSO
- `subjects` (str, N) and `conditions` (str, N) — original labels
- `files` (str, N) — source basenames per epoch
- `file_index` (int32, N) — index of the source file per epoch
- `file_ptr` (int32, F+1) — cumulative epoch offsets per file
- `P` (int32), `N` (int32) — feature and sample counts
- `pairs_idx_path`, `pairs_json_path` (str) — paths to pair metadata (if found)

## Minimal example (PowerShell)
```powershell
python scripts/stack_dataset.py `
  --labels-csv data/CK/labels.csv `
  --input-dir data/processed/wsmi/sessionA_vec `
  --glob "*_vec.npz" `
  --out-file data/processed/datasets/sessionA_stack.npz
```

## CSV format reminder
- Required header: `source_file,subject,condition`
- `source_file` must match what `vectorize_wsmi.py` recorded (basename). If unsure, inspect the `source_file` field inside any `*_vec.npz`.
