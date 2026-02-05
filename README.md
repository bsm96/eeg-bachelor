### eeg-bachelor

Small EEG analysis project for my bachelor thesis. The code is mainly a set of scripts + notebooks built around MNE.

Note: Because Rigshospitalet complained about some Python versions on their systems because of security measures, I ended up having to use multiple Python versions and run parts of the code in a different way. This also caused some compatibility issues.

My recommendation is: run the code and install packages as you go, whenever you discover that something is missing.

Run the scripts in numerical order.

## What is in this repo

- Scripts for preprocessing and feature generation: [scripts/](scripts/)
- Analysis notebooks: [notebooks/](notebooks/)
- Short script-specific notes: [docs/](docs/)
- Configs: [configs/](configs/)
- Figures/exports: [plots/](plots/)

## Setup

I use conda with the environment in [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate eeg_env
```

## Typical workflow

1) Create epochs (events or sliding windows):

```bash
python scripts/01_make_epochs.py --help
```

2) Compute connectivity/features and build datasets:

```bash
python scripts/02_compute_wsmi.py --help
python scripts/03_vectorize_wsmi.py --help
python scripts/04_stack_dataset.py --help
```

3) Learn/assign brain states:

```bash
python scripts/05_learn_brain_states.py --help
python scripts/06_assign_states_and_we.py --help
```

More details for individual steps are in the docs folder (for example: [docs/readme_01_make_epochs.md](docs/readme_01_make_epochs.md)).