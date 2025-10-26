#!/usr/bin/env python3

### Example of folder name that it will create: …/reduce/alpha__median-trim_tp0p1/…, whereas tp = trim proportion and 0p1 = 0.1, so 0p1 = 0.1 = 10% trim in each tail.

from __future__ import annotations  # enable future-compatible annotations
import argparse  # parse CLI flags for batch runs
import os  # read/modify environment for child processes
from pathlib import Path  # robust cross-platform paths
import shlex  # pretty-print command lines for logs
import subprocess as sp  # run compute/reduce scripts as child processes
from typing import List, Tuple  # precise type hints for clarity


# --- Ensure src/ is importable inside child processes (so "from eeg..." works) ---
ROOT = Path(__file__).resolve().parents[1]  # project root directory resolved from this script
SRC = ROOT / "src"  # points at the src-layout folder containing the 'eeg' package
ENV = os.environ.copy()  # copy current environment so we can modify safely
ENV["PYTHONPATH"] = (  # prepend src/ to PYTHONPATH for child processes
    f"{SRC}{os.pathsep}{ENV['PYTHONPATH']}" if "PYTHONPATH" in ENV else str(SRC)
)


# --- Helpers to parse band spec and to build safe folder tags ---
def parse_bands(spec: str) -> List[Tuple[str, float, float]]:
    """
    Parse a comma-separated band spec like 'delta:1-4,theta:4-7' into a list of tuples.
    """
    bands: List[Tuple[str, float, float]] = []  # container for (name, low, high) tuples
    for part in spec.split(","):  # split on commas to get each "name:lo-hi"
        name, rng = part.strip().split(":")  # split into the band name and numeric range string
        lo_s, hi_s = rng.split("-")  # split numeric range at the dash
        bands.append((name.strip(), float(lo_s), float(hi_s)))  # append parsed (name, low, high)
    return bands  # return ordered list of bands


def num_token(x: float | int | None) -> str:
    """
    Turn a numeric value into a filesystem-friendly token (e.g., 0.1 -> '0p1', -3 -> 'm3').
    """
    if x is None:  # guard if nothing to encode
        return "none"  # explicit placeholder token
    s = str(x)  # convert to string without forcing format (caller controls precision)
    s = s.replace("-", "m")  # avoid '-' in folder names by replacing with 'm' (for minus)
    s = s.replace(".", "p")  # avoid '.' in folder names by replacing with 'p' (for point)
    return s  # return tokenized string


def compute_tag(k: int, tau: int, normalize: bool, tie_break: str, picks: str) -> str:
    """
    Build a deterministic tag describing compute-stage settings for folder names.
    """
    parts = [  # collect tokens that describe compute configuration
        f"k{num_token(k)}",  # embedding dimension token
        f"tau{num_token(tau)}",  # lag token
        ("norm" if normalize else "unnorm"),  # normalization flag token
        tie_break,  # tie-breaking policy (already a short string)
        f"picks-{picks.replace(',', '_')}",  # picks token; commas converted to underscores
    ]
    return "__" + "_".join(parts)  # prefix with separator and join into one tag


def reduce_tag(strategy: str, trim_prop: float | None) -> str:
    """
    Build a deterministic tag describing reduce-stage settings for folder names.
    """
    base = strategy  # always include the strategy string itself
    if "trim" in strategy.lower():  # include trim proportion only when strategy uses trimming
        base += f"_tp{num_token(trim_prop)}"  # add a token with the numeric trim proportion
    return "__" + base  # prefix with separator for readability


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch run wSMI compute+reduce across bands and epoch files.")  # CLI help text
    ap.add_argument("--epochs-dir", type=str, required=True,
                    help="Folder containing multiple *-epo.fif files (one subject per file).")  # directory with epoch files
    ap.add_argument("--bands", type=str, default="delta:1-4,theta:4-7,alpha:8-12,beta:13-30",
                    help="Comma-separated band spec: 'name:lo-hi,name:lo-hi,...'.")  # band spec string
    ap.add_argument("--out-root", type=str, default="reports",
                    help="Root output folder under which 'wsmi/' and 'reduce/' will be created.")  # root for outputs
    ap.add_argument("--k", type=int, default=3, help="Embedding dimension (set externally).")  # embedding dimension
    ap.add_argument("--tau", type=int, default=8, help="Lag in samples (set externally).")  # lag in samples
    ap.add_argument("--normalize", action="store_true", help="Normalize wSMI by ln(k!).")  # normalization flag
    ap.add_argument("--tie-break", type=str, default="jitter", choices=["jitter", "ordinal"],
                    help="Tie policy for ordinal patterns (propagated to compute stage).")  # tie-break policy
    ap.add_argument("--picks", type=str, default="eeg", help="MNE picks passed to compute stage.")  # channel selector
    ap.add_argument("--strategy", type=str, default="median-trim",
                    help="Reduce-stage aggregation strategy.")  # aggregation strategy
    ap.add_argument("--trim-proportion", type=float, default=0.1,
                    help="Trim fraction per tail when strategy uses trimming.")  # trim fraction
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip compute/reduce for (band, subject) when target files already exist.")  # skipping flag
    args = ap.parse_args()  # parse CLI args into a namespace

    epochs_dir = Path(args.epochs_dir)  # Path object for the epochs directory
    out_root = Path(args.out_root)  # Path object for the output root directory
    bands = parse_bands(args.bands)  # parse bands string into a structured list

    fif_files = sorted(epochs_dir.glob("*-epo.fif"))  # discover all epochs files matching the pattern
    if not fif_files:  # guard when no files are found
        raise FileNotFoundError(f"No *-epo.fif files found in: {epochs_dir}")  # explicit error

    # build spec tags once (stable within a run)
    tag_compute = compute_tag(args.k, args.tau, args.normalize, args.tie_break, args.picks)  # compute-stage tag
    tag_reduce = reduce_tag(args.strategy, args.trim_proportion)  # reduce-stage tag

    for fif in fif_files:  # loop over all epoch files (subjects)
        subj = fif.stem  # base filename without extension; used as a subject identifier
        for band_name, lo, hi in bands:  # loop over all band definitions
            # --- construct output folders with spec tags included ---
            out_compute = out_root / "wsmi" / f"{band_name}{tag_compute}" / subj  # wsmi/<band><spec>/<subject>
            out_reduce = out_root / "reduce" / f"{band_name}{tag_reduce}" / subj  # reduce/<band><spec>/<subject>
            out_compute.mkdir(parents=True, exist_ok=True)  # make sure compute out folder exists
            out_reduce.mkdir(parents=True, exist_ok=True)  # make sure reduce out folder exists

            # --- key output files to decide skipping/re-running ---
            npz_path = out_compute / "wsmi_matrices.npz"  # NPZ path written by compute stage
            summary_json = out_reduce / "wsmi_summary.json"  # summary path written by reduce stage

            # --- COMPUTE stage ---
            if not (args.skip_existing and npz_path.exists()):  # run unless skipping and file already exists
                cmd_compute = [  # build command list for wsmi_compute.py
                    os.fspath(Path(os.sys.executable)),  # current Python interpreter path
                    os.fspath(ROOT / "scripts" / "wsmi_compute.py"),  # compute CLI script path
                    "--epochs", os.fspath(fif),  # path to current epochs file
                    "--out", os.fspath(out_compute),  # compute output folder (with spec tag)
                    "--l-freq", str(lo),  # low cutoff in Hz for this band
                    "--h-freq", str(hi),  # high cutoff in Hz for this band
                    "--k", str(args.k),  # embedding dimension as provided
                    "--tau", str(args.tau),  # lag in samples as provided
                    "--tie-break", args.tie_break,  # tie policy propagated from batch CLI
                    "--picks", args.picks,  # channel picks propagated from batch CLI
                ]
                if args.normalize:  # optionally include normalization flag
                    cmd_compute.append("--normalize")  # add normalize switch

                print("\n[compute]", shlex.join(cmd_compute))  # log the compute command nicely
                r = sp.run(cmd_compute, env=ENV)  # run compute stage as a child process
                if r.returncode != 0:  # check for non-zero exit code
                    raise SystemExit(f"compute failed for {fif} {band_name}")  # abort batch with explicit message

            # --- REDUCE stage ---
            if not (args.skip_existing and summary_json.exists()):  # run unless skipping and file already exists
                cmd_reduce = [  # build command list for wsmi_reduce.py
                    os.fspath(Path(os.sys.executable)),  # current Python interpreter path
                    os.fspath(ROOT / "scripts" / "wsmi_reduce.py"),  # reduce CLI script path
                    "--wsmi-npz", os.fspath(npz_path),  # path to NPZ produced by compute stage
                    "--out", os.fspath(out_reduce),  # reduce output folder (with spec tag)
                    "--strategy", args.strategy,  # aggregation strategy from batch CLI
                    "--subject-id", subj,  # subject identifier to embed in outputs
                    "--band", band_name,  # band label to embed in outputs
                ]
                if "trim" in args.strategy.lower():  # include trim proportion only when strategy requests it
                    cmd_reduce += ["--trim-proportion", str(args.trim_proportion)]  # add trim fraction

                print("[reduce ]", shlex.join(cmd_reduce))  # log the reduce command nicely
                r = sp.run(cmd_reduce, env=ENV)  # run reduce stage as a child process
                if r.returncode != 0:  # check for errors
                    raise SystemExit(f"reduce failed for {fif} {band_name}")  # abort batch with explicit message

    print("\nFINISH! Ran through all the bands for all patients with all the epochs.")

if __name__ == "__main__":
    main()
