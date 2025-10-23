#!/usr/bin/env python3
from __future__ import annotations
import argparse                   # parse CLI flags
import os                         # env vars for subprocess
from pathlib import Path          # robust paths
import shlex                      # human-friendly logging
import subprocess as sp           # run our existing CLI scripts
from typing import Dict, List, Tuple

# --- Ensure src/ is importable for child processes (so `from eeg...` works) ---
ROOT = Path(__file__).resolve().parents[1]          # project root
SRC = ROOT / "src"                                  # src-layout root
ENV = os.environ.copy()                             # copy current env
ENV["PYTHONPATH"] = (                               # prepend src to PYTHONPATH for children
    (str(SRC) + os.pathsep + ENV["PYTHONPATH"]) if "PYTHONPATH" in ENV else str(SRC)
)

# --- Parse "alpha:8-12,theta:4-7" into [("alpha", 8.0, 12.0), ...] ---
def parse_bands(spec: str) -> List[Tuple[str, float, float]]:
    bands: List[Tuple[str, float, float]] = []
    for part in spec.split(","):
        name, rng = part.strip().split(":")
        lo_s, hi_s = rng.split("-")
        bands.append((name.strip(), float(lo_s), float(hi_s)))
    return bands

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch run wSMI compute+reduce for all bands and all epochs files.")
    ap.add_argument("--epochs-dir", type=str, required=True,
                    help="Folder containing multiple *-epo.fif files (one subject per file).")
    ap.add_argument("--bands", type=str, default="delta:1-4,theta:4-7,alpha:8-12,beta:13-30",
                    help="Comma-separated band specs 'name:lo-hi'. You can override this string.")
    ap.add_argument("--out-root", type=str, default="reports",
                    help="Root output folder for wsmi/ and reduce/ subfolders.")
    ap.add_argument("--k", type=int, default=3, help="Embedding dimension k.")
    ap.add_argument("--tau", type=int, default=8, help="Lag in samples tau.")
    ap.add_argument("--normalize", action="store_true", help="Normalize wSMI by ln(k!).")
    ap.add_argument("--strategy", type=str, default="median-trim",
                    help="Aggregation strategy for reduce step.")
    ap.add_argument("--trim-proportion", type=float, default=0.1,
                    help="Trim fraction per tail when strategy includes trimming.")
    ap.add_argument("--picks", type=str, default="eeg", help="MNE picks string used in compute.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip compute/reduce if target files already exist.")
    args = ap.parse_args()

    epochs_dir = Path(args.epochs_dir)
    out_root = Path(args.out_root)
    bands = parse_bands(args.bands)

    fif_files = sorted(epochs_dir.glob("*-epo.fif"))  # find all epoch files
    if not fif_files:
        raise FileNotFoundError(f"No *-epo.fif files found in: {epochs_dir}")

    for fif in fif_files:
        subj = fif.stem  # e.g., "patient2_window16s_stride1s-epo"
        for band_name, lo, hi in bands:
            # --- compute step output paths ---
            out_compute = out_root / "wsmi" / band_name / subj
            out_compute.mkdir(parents=True, exist_ok=True)
            npz_path = out_compute / "wsmi_matrices.npz"

            # --- reduce step output path ---
            out_reduce = out_root / "reduce" / band_name / subj
            out_reduce.mkdir(parents=True, exist_ok=True)
            summary_json = out_reduce / "wsmi_summary.json"

            # --- compute: only if not skipping or target missing ---
            if not (args.skip_existing and npz_path.exists()):
                cmd_compute = [
                    os.fspath(Path(os.sys.executable)),                  # current Python
                    os.fspath(ROOT / "scripts" / "wsmi_compute.py"),     # our compute CLI (uses MNE: also C&K/Engemann/Della Bella)
                    "--epochs", os.fspath(fif),
                    "--out", os.fspath(out_compute),
                    "--l-freq", str(lo),
                    "--h-freq", str(hi),
                    "--k", str(args.k),
                    "--tau", str(args.tau),
                    "--tie-break", "jitter",
                    "--picks", args.picks,
                ]
                if args.normalize:
                    cmd_compute.append("--normalize")
                # call and check
                print("\n[compute]", shlex.join(cmd_compute))
                r = sp.run(cmd_compute, env=ENV)
                if r.returncode != 0:
                    raise SystemExit(f"compute failed for {fif} {band_name}")

            # --- reduce: only if not skipping or target missing ---
            if not (args.skip_existing and summary_json.exists()):
                cmd_reduce = [
                    os.fspath(Path(os.sys.executable)),
                    os.fspath(ROOT / "scripts" / "wsmi_reduce.py"),       # our reduce CLI
                    "--wsmi-npz", os.fspath(npz_path),
                    "--out", os.fspath(out_reduce),
                    "--strategy", args.strategy,
                    "--subject-id", subj,
                    "--band", band_name,
                ]
                if "trim" in args.strategy:
                    cmd_reduce += ["--trim-proportion", str(args.trim_proportion)]
                print("[reduce ]", shlex.join(cmd_reduce))
                r = sp.run(cmd_reduce, env=ENV)
                if r.returncode != 0:
                    raise SystemExit(f"reduce failed for {fif} {band_name}")

    print("\nFINISH! Ran through all the bands for all patients with all the epochs.")

if __name__ == "__main__":
    main()
