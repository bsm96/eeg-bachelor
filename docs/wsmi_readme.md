# wSMI Computation — Quick Guide

Computes per-epoch wSMI connectivity matrices using NICE library.

---

## Usage

```powershell
python scripts/compute_wsmi.py \
  --input-dir <path_to_epochs> \
  --out-dir data/processed/wsmi \
  --k 3 \
  --tau-ms 8 \
  [--assume-csd] \
  [--overwrite]
```

---

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-dir` | Directory with epochs FIF files | *required* |
| `--out-dir` | Output directory | *required* |
| `--k` | Embedding dimension | 3 |
| `--tau-ms` | Time delay (ms) | 8.0 |
| `--assume-csd` | Input already CSD-transformed | False |
| `--glob` | File pattern | `*-epo.fif` |
| `--overwrite` | Overwrite existing files | False |

---

## CSD Modes

**Option 1: NICE applies CSD** (default)
```powershell
--input-dir data/epochs/events_tmin-0p2_tmax15
```
→ Output: `k3_tau8ms_csd_nice/patient_csd_nice_wsmi_k3_tau8ms.npz`

**Option 2: MNE already applied CSD** (recommended)
```powershell
--input-dir data/epochs/events_tmin-0p2_tmax15_csd --assume-csd
```
→ Output: `k3_tau8ms_csd_mne/patient_csd_mne_wsmi_k3_tau8ms.npz`

> **Note:** CSD is mandatory. `--no-csd` is not supported by NICE.

---

## Output

Each `.npz` file contains:
- `wsmi`: (n_epochs, n_channels, n_channels) — connectivity matrices
- `events`: (n_epochs,) — event labels
- `ch_names`: (n_channels,) — channel names
- `meta`: JSON with parameters (k, tau, sfreq, csd_tag, etc.)
