# 02_compute_wpli.py

Computes wPLI connectivity in the alpha band (8–13 Hz) for all sliding-window epochs and saves one wPLI matrix per file.

The input and output paths are defined **inside** `scripts/02_compute_wpli.py` and should be changed there if your folder structure is different.

## Input

- `data/epochs/sliding_window16s_stride1s/*-epo.fif` (EEG epochs from 16 s window, 1 s stride; default in the script)

## Output

- `data/processed/wpli_alpha/<name>_wpli_alpha.npy`  
  - NumPy array with wPLI matrix (channels × channels; default in the script)
