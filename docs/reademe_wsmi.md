# wSMI Computation — Quick Guide

Computes per-epoch wSMI connectivity matrices using NICE's SymbolicMutualInformation.

---

## Usage

```powershell
python scripts/compute_wsmi.py `
  --input-dir <path_to_epochs> `
  --out-dir data/processed/wsmi `
  --k 3 `
  --tau-ms 8
  [--tmin <sec>] [--tmax <sec>] \
  [--backend python|openmp] [--nthreads auto|<int>] \
  [--overwrite]
```

Example for the 15 s resting-state epochs bundled with the project:

```powershell
python scripts/compute_wsmi.py `
  --input-dir data/epochs/events_tmin-0p2_tmax15 `
  --out-dir data/processed/wsmi `
  --k 3 `
  --tau-ms 8
```

### What the script does

- **Discovers input epochs** using the provided glob (default `*-epo.fif`).
- **Creates** `data/processed/wsmi/<input-folder>/k<k>_tau<tau_ms>ms_csd_nice/` on demand.
- **Loads each epoch file with MNE** and converts `tau-ms` to integer samples.
- **Initialises the NICE `SymbolicMutualInformation` marker** with the chosen kernel, tau, optional time window (`tmin`, `tmax`), backend, and thread settings.
- **Invokes `fit(epochs)`**. Internally NICE applies current source density, low-pass filtering at `sfreq / (k * tau)`, symbolic encoding, and wSMI estimation.
- **Collects the resulting matrices** (`smi.data_`), ensures they are shaped `(n_epochs, n_channels, n_channels)`, and serialises them alongside metadata and events.
- **Displays a lightweight ASCII progress bar** so lengthy batches show incremental progress.

### Arguments in detail

- `--input-dir`: Folder that contains the epoch FIF files you want to process.
- `--out-dir`: Root folder where output subdirectories will be created.
- `--k`: Embedding dimension for symbolic permutation; impacts filter cutoff (`sfreq / (k * tau)`).
- `--tau-ms`: Delay in milliseconds between samples forming a symbol. Converted to samples per recording.
- `--tmin`, `--tmax`: Optional time window (seconds) inside each epoch; `None` uses the full epoch.
- `--backend`: `python` (pure NumPy) or `openmp` (if the optimised extension is installed).
- `--nthreads`: Thread count when using `openmp`; `auto` lets NICE pick a value.
- `--glob`: Pattern for selecting epoch files; adjust if your filenames differ.
- `--overwrite`: If supplied, existing `.npz` files are replaced; otherwise they are skipped.

### Output layout

- Folder: `data/processed/wsmi/<input-folder>/k<k>_tau<tau_ms>ms_csd_nice/`.
- File name: `<epoch-stem>_wsmi_k<k>_tau<tau_ms>ms.npz` (one per input file).
- Each archive contains:
  - `wsmi`: `float32`, shape `(n_epochs, n_channels, n_channels)`.
  - `events`: integer array of epoch event codes.
  - `ch_names`: channel labels (dtype `object`).
  - `meta`: JSON string describing sfreq, kernel, tau, tau in samples, backend, NICE function, and comment.

### Notes on preprocessing

- NICE computes **current source density (CSD)** internally unless `method_params={'bypass_csd': True}` is set.
- A **Butterworth low-pass filter** is applied inside NICE at `sfreq / (k * tau)` (≈41.7 Hz for `k=3`, `tau=8 ms` at 125 Hz) to avoid aliasing before symbolic encoding.
- You can override the cutoff via `method_params={'filter_freq': <Hz>}` if required.

### About `SymbolicMutualInformation`-function in the `compute_wsmi.py` file

- **Where it lives:** `nice.markers.connectivity.SymbolicMutualInformation` derives from NICE’s `BaseMarker` and adheres to a scikit-learn-like interface (`fit`, `transform`, `fit_transform`).
- **Constructor parameters:**
  - `tmin`, `tmax` (seconds) select the temporal window inside each epoch. `None` keeps the whole epoch.
  - `kernel` (int) sets the permutation order. Larger kernels increase the factorial symbol space and lower the default anti-aliasing cutoff.
  - `tau` (samples) is the lag between samples that form one symbol; in the script we convert the CLI `--tau-ms` to samples per recording.
  - `backend` chooses between `"python"` (pure NumPy/Numba) and `"openmp"` (compiled routine, if the wheel is available).
  - `method` toggles between `'weighted'` (wSMI) and `'default'` (plain SMI), affecting which matrix is surfaced.
  - `comment` is a free-text tag saved alongside outputs so downstream tools can identify the run.
  - `method_params` injects backend-specific tweaks, for example:
    - `{'filter_freq': 50.0}` overrides the automatic low-pass (`sfreq / (kernel * tau)`).
    - `{'bypass_csd': True}` skips the built-in current source density step and works on the raw epochs.
    - `{'nthreads': 4}` (or `'auto'`) controls OpenMP parallelism.
- **Lifecycle methods:**
  - `fit(epochs)` performs the full computation: optional CSD transform, Butterworth low-pass, symbolic encoding, and mutual information estimation. After completion, `smi.data_` has shape `(n_channels, n_channels, n_epochs)`.
  - `transform(epochs)` and `fit_transform(epochs)` are convenience wrappers that return the marker instance, enabling integration with other NICE pipelines.
  - `save(path)` / `load(path)` persist or reload a fitted marker using NICE’s storage format.
  - `reduce_to_epochs()`, `reduce_to_topo()`, and `reduce_to_scalar()` collapse the 3-D connectivity tensor when you need per-epoch vectors, static connectivity maps, or single summary numbers.
- **Runtime attributes exposed after `fit`:**
  - `data_`: the wSMI matrix `(channels × channels × epochs)` (the script reorders axes before writing `.npz`).
  - `events_`: event codes copied from the input `Epochs` object.
  - `ch_names_`: channel labels aligned with the matrix indices.
  - `meta_`: dictionary with runtime metadata (sfreq, kernel, tau, filter frequency, backend, comment, etc.).
- **Usage in this repository:** every FIF file is processed independently. We instantiate `SymbolicMutualInformation` inside the loop with the per-file `tau` (in samples), call `fit(epochs)`, move the axes to `(n_epochs, n_channels, n_channels)`, then persist the result together with events, channel names, and metadata.

### Troubleshooting

- `AttributeError: 'SymbolicMutualInformation' object has no attribute 'compute'`: use `fit(epochs)` and read `marker.data_` (the script already does this).
- `TypeError: ... object is not callable`: same root cause—avoid calling the instance directly.
- `The '<' operator is reserved for future use`: replace placeholder `<path_to_epochs>` with an actual path when invoking from PowerShell.

---

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-dir` | Directory with epochs FIF files | *required* |
| `--out-dir` | Output directory | *required* |
| `--k` | Embedding dimension | 3 |
| `--tau-ms` | Time delay (ms) | 8.0 |
| `--tmin` | Window start seconds (None = epoch start) | None |
| `--tmax` | Window end seconds (None = epoch end) | None |
| `--backend` | NICE backend | `python` |
| `--nthreads` | Thread count (for backend) | `auto` |
| `--glob` | File pattern | `*-epo.fif` |
| `--overwrite` | Overwrite existing files | False |

---

## CSD

Inputs should be raw (non-CSD) epochs. NICE applies CSD internally.

```powershell
--input-dir data/epochs/events_tmin-0p2_tmax15
```
→ Output folder: `events_tmin-0p2_tmax15_csd/k3_tau8ms_csd_nice/`
→ File name: `patient_csd_nice_wsmi_k3_tau8ms.npz`

---

## Output

Each `.npz` file contains:
- `wsmi`: (n_epochs, n_channels, n_channels) — connectivity matrices
- `events`: (n_epochs,) — event labels
- `ch_names`: (n_channels,) — channel names
- `meta`: JSON with parameters (k, tau, sfreq, csd_tag, etc.)
