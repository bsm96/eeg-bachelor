# 01_make_epochs.py

Create MNE epochs from raw EEG files (EDF/FIF).

## Modes

- **events**: Event-locked epochs from annotations (e.g., Resting, Familiar voice, Medical staff)
- **sliding**: Fixed-length overlapping windows across continuous data

## Example Usage

### Event-locked epochs (default -0.2 to 15.0 s)
```bash
python scripts/01_make_epochs.py `
  --mode events `
  --input-dir "data/CK/Raws_new_ica" `
  --glob "*.fif" `
  --events "Familiar voice,Medical staff,Resting" 
  --overwrite
```


### Sliding window epochs (16 s window, 1 s stride)
```bash
python scripts/01_make_epochs.py `
  --mode sliding `
  --window 16.0 `
  --stride 1.0 `
  --input-dir "data/CK/Raws_new_ica" `
  --glob "*.fif" `
  --n-jobs -2 `
  --overwrite
```

python scripts\01_make_epochs.py --mode sliding --window 16.0 --stride 1.0 --input-dir "data\CK\raws_missing" --glob "*.fif" --sfreq 125 --n-jobs 1 --overwrite
python scripts\01_make_epochs.py --mode sliding --window 16.0 --stride 1.0 --input-dir "data\CK\Raws_new_ica" --glob "*.fif" --sfreq 125 --n-jobs 1 --overwrite
python scripts\01_make_epochs.py --mode sliding --window 16.0 --stride 1.0 --input-dir "data\CK\used_raws" --glob "*.fif" --sfreq 125 --n-jobs 1 --overwrite

## Key Arguments

- `--mode`: `events` or `sliding`
- `--input-dir`: Folder with EDF/FIF files
- `--glob`: File pattern (e.g., `*.fif`, `*.edf`)
- `--out-dir`: Output directory (default: `data/epochs`)
- `--n-jobs`: Parallel jobs (`1`=sequential, `-1`=all cores, `-2`=all but one)
- `--overwrite`: Overwrite existing output files

### Events mode
- `--tmin`, `--tmax`: Time window relative to event (default: -0.2, 15.0 s)
- `--events`: Comma-separated annotation labels to include

### Sliding mode
- `--window`: Window length in seconds
- `--stride`: Stride/step size in seconds
- `--start-offset`, `--stop-offset`: Crop data before windowing

## Output

Epochs saved as `-epo.fif` files in subdirectories named by parameters:
- Events: `data/epochs/events/` (or with custom tmin/tmax if specified)
- Sliding: `data/epochs/sliding_window16s_stride1s/`
