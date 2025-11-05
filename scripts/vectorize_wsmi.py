"""
Vectorize wSMI matrices to per-epoch feature matrices.

Purpose
-------
Converts per-epoch wSMI connectivity cubes (E, C, C) into 2D feature matrices
of shape (E, P), where P is the number of unique channel pairs P = C*(C-1)/2.
The upper triangle (excluding the diagonal) is used as the feature vector order.

Outputs per recording an "*_vec.npz" file with keys:
  - X: float32 array (E, P)
  - E: int number of epochs
  - C: int number of channels
  - P: int number of features (pairs)
  - source_file: str original .npz file path
  - ch_names: list[str] channel names (if available in input)

Additionally writes once per run into the output directory:
  - pairs_idx.npy: [2, P] int64 array with row 0 = I, row 1 = J (np.triu_indices(C, 1))
  - pairs.json: list of [ch_i, ch_j] names in the same column order as X
  - manifest.csv: CSV with vec_file,source_file,E,C,P

Example usage
-------------
python scripts/vectorize_wsmi.py \
  --input-dir data/processed/wsmi/k3_tau8ms_csd_nice \
  --glob "*.npz" \
  --out-dir data/processed/wsmi/k3_tau8ms_csd_nice_vec
  --out-dir
"""

# vectorize_wsmi.py
# Input: dine .npz (E, C, C). (epochs, channels, channels)
# Output: pr. recording en X med rækker = epoker og kolonner = upper-triangle (k=1) af wSMI-matrixen (dvs. feature-vector 𝑓 pr. epoke) + en “pairs.json” med (ch_i, ch_j) → kolonneindeks.
# Tip: i,j = np.triu_indices(C, k=1) og f = W[i, j].

from __future__ import annotations  # Postponed evaluation of type annotations for forward references

import argparse  # Command line interface parsing
import csv  # Writing the manifest.csv file
import json  # Writing pairs.json with channel name pairs
import logging  # Structured logging for progress and diagnostics
import os  # Filesystem utilities
import sys  # Exit code handling
import time  # Timing per-file processing
from dataclasses import dataclass  # Lightweight containers for loaded data
from typing import Iterable, List, Optional, Sequence, Tuple  # Type hints for clarity

import numpy as np  # Numerical operations and NPZ I/O


@dataclass  # Structure carrying data, channel names (normalized and raw), and passthrough fields
class LoadedWSMI:
	data: np.ndarray  # (E, C, C) cube with one matrix per epoch
	ch_names: Optional[List[str]]  # Optional normalized list of channel names for validation and pairs
	raw_ch_names: Optional[np.ndarray]  # Raw channel names array from input NPZ to preserve unchanged
	extras: dict  # Passthrough fields: events, event_labels, event_codes, event_names, meta_json


def setup_logging(level: int = logging.INFO) -> None:
	"""Configure root logger with a concise format."""
	# Keep logs short and consistent across the script
	logging.basicConfig(
		level=level,  # Default to INFO to show progress without being too verbose
		format="[%(levelname)s] %(message)s",  # Compact prefix with level only
	)


def list_npz_files(input_dir: str, pattern: str) -> List[str]:
	"""List .npz files in alphabetical order matching pattern within input_dir."""
	import glob  # Local import to avoid polluting module scope

	# Resolve the glob against the input directory
	files = glob.glob(os.path.join(input_dir, pattern))  # Collect matching paths
	files = [f for f in files if f.lower().endswith(".npz")]  # Only keep .npz files
	files.sort()  # Deterministic processing order
	return files  # Alphabetically sorted list


def _safe_to_list(x) -> Optional[List[str]]:
	"""Convert ch_names-like payload to a list of str, handling bytes/object arrays.

	Returns None if conversion is not possible.
	"""
	try:
		if x is None:
			return None  # Nothing to convert
		if isinstance(x, (list, tuple)):
			out = []  # Build a normalized list of strings
			for v in x:
				if isinstance(v, bytes):
					out.append(v.decode("utf-8", errors="replace"))  # Decode bytes safely
				else:
					out.append(str(v))  # Stringify any other type
			return list(out)
		if isinstance(x, np.ndarray):
			# Object arrays often wrap lists or strings; normalize to a Python list
			if x.dtype == object:
				# Single scalar object → extract, single list in a 1D array → unwrap
				if x.ndim == 0:
					x = [x.item()]  # Convert 0-dim object array to a list
				elif x.ndim == 1 and len(x) == 1 and isinstance(x[0], (list, tuple, np.ndarray)):
					x = x[0]  # Unwrap single nested container
				else:
					x = x.tolist()  # Fallback: convert to Python list
			elif x.dtype.kind in {"U", "S"}:
				x = x.tolist()  # Unicode/bytes arrays → list
			else:
				# Not strings; stringify elements
				x = [str(v) for v in x.tolist()]
			return _safe_to_list(x)  # Recurse with a now-Python list
	except Exception:
		return None  # Any failure → treat as unavailable
	return None  # If none of the branches applied


def load_wsmi(npz_path: str) -> LoadedWSMI:
	"""Load wSMI and passthrough fields from an .npz file.

	Preference order for payload:
	- Key 'wsmi' if present
	- Otherwise, the first numeric array with 2 or 3 dimensions

	Attempts to read and preserve unchanged when present:
	- events, event_labels, event_codes, event_names, ch_names, meta_json
	Channel names are returned both as a normalized Python list (for validation) and the raw array (for saving unchanged).
	"""
	def _process(npz) -> LoadedWSMI:
		keys = list(npz.files)
		# Require canonical 'wsmi' key only
		if "wsmi" not in keys:
			raise ValueError("Missing 'wsmi' key in NPZ file")
		payload = npz["wsmi"]
		chosen_key = "wsmi"

		# Channel names: raw for saving, normalized list for validation and pairs.json
		raw_ch = None
		norm_ch: Optional[List[str]] = None
		for ck in ("ch_names", "channels", "chan_names"):
			if ck in keys:
				raw_ch = npz[ck]
				norm_ch = _safe_to_list(raw_ch)
				break

		# Passthrough fields (preserved as-is)
		extras = {}
		for k in ("events", "event_labels", "event_codes", "event_names", "meta_json"):
			if k in keys:
				extras[k] = npz[k]

		logging.info(
			f"Loaded '{os.path.basename(npz_path)}' with key '{chosen_key}', shape={payload.shape}, dtype={payload.dtype}"
		)
		return LoadedWSMI(data=payload, ch_names=norm_ch, raw_ch_names=raw_ch, extras=extras)

	try:
		# Load safely without pickle; fail fast on object arrays or malformed files
		with np.load(npz_path, allow_pickle=False) as npz:
			return _process(npz)
	except Exception as e:
		raise RuntimeError(f"Failed to load '{npz_path}': {e}") from e


def ensure_ecc(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
	"""Validate array is shaped (E, C, C). Returns (arr, E, C) or fails fast.

	Only accepts arrays with ndim == 3 and square last two dimensions.
	"""
	if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
		raise ValueError("Expected wsmi with shape (E, C, C)")
	E, C = int(arr.shape[0]), int(arr.shape[1])
	return arr, E, C


def compute_pairs(C: int, ch_names: Optional[Sequence[str]]) -> Tuple[np.ndarray, np.ndarray, Optional[List[Tuple[str, str]]]]:
	"""Compute upper-triangular indices and optional name pairs for channels.

	Returns (I, J, pairs_names) where pairs_names is a list of (name_i, name_j) or None.
	"""
	I, J = np.triu_indices(C, k=1)  # Indices for upper triangle without diagonal
	I = I.astype(np.int64, copy=False)  # Store indices as 64-bit for compatibility
	J = J.astype(np.int64, copy=False)
	pairs_names = None  # Default: no names if ch_names not available
	if ch_names is not None and len(ch_names) == C:
		# Build pairs list in the same order as indices
		pairs_names = [(str(ch_names[i]), str(ch_names[j])) for i, j in zip(I.tolist(), J.tolist())]
	return I, J, pairs_names  # Return indices and optional names


def vectorize_upper(ecc: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
	"""Vectorize (E, C, C) by taking upper triangle without diagonal.

	Returns X with shape (E, P) as float32.
	"""
	# Advanced indexing across the first axis to gather pairs per epoch
	X = ecc[:, I, J]  # Shape becomes (E, P) due to broadcasting of I,J over E
	if X.ndim != 2:
		# Defensive: enforce 2D (E, P) in rare edge cases
		X = np.reshape(X, (ecc.shape[0], -1))
	return X.astype(np.float32, copy=False)  # Use float32 to save space downstream


def save_vectors(
	out_file: str,
	X: np.ndarray,
	E: int,
	C: int,
	P: int,
	source_file: str,
	raw_ch_names: Optional[np.ndarray],
	extras: Optional[dict],
) -> None:
	"""Save vectorized features and preserved metadata for a single recording.

	Only writes passthrough fields that were present in the input.
	"""
	kw = {
		"X": X,  # Feature matrix (E, P)
		"E": int(E),  # Number of epochs
		"C": int(C),  # Number of channels
		"P": int(P),  # Number of features (pairs)
		"source_file": str(source_file),  # Original NPZ file path
	}
	if raw_ch_names is not None:
		# Save channel names unchanged from input NPZ
		kw["ch_names"] = raw_ch_names
	if extras:
		# Merge only approved keys to avoid collisions
		for k in ("events", "event_labels", "event_codes", "event_names", "meta_json"):
			if k in extras:
				kw[k] = extras[k]
	np.savez_compressed(out_file, **kw)


def write_pairs(out_dir: str, I: np.ndarray, J: np.ndarray, pairs_names: Optional[Sequence[Tuple[str, str]]]) -> Tuple[str, Optional[str]]:
	"""Write global pairs_idx.npy and pairs.json (if names available)."""
	idx_path = os.path.join(out_dir, "pairs_idx.npy")  # Path for index array
	to_save = np.stack([I, J], axis=0).astype(np.int64, copy=False)  # Shape (2, P)
	np.save(idx_path, to_save)  # Save as plain .npy for fast loading
	json_path = None  # Default: no JSON if names unavailable
	if pairs_names is not None:
		json_path = os.path.join(out_dir, "pairs.json")  # Path for channel name pairs
		with open(json_path, "w", encoding="utf-8") as f:
			json.dump([[a, b] for a, b in pairs_names], f, ensure_ascii=False, indent=2)  # Pretty JSON
	return idx_path, json_path  # Return paths for logging


def validate_ch_names(ref: Optional[Sequence[str]], current: Optional[Sequence[str]], file_path: str) -> None:
	"""Validate channel names are present and identical to the reference if provided.

	Raises ValueError with a clear message on mismatch.
	"""
	if ref is None or current is None:
		return  # Nothing to validate when either side is missing
	if len(ref) != len(current):
		# Immediate mismatch on channel count
		raise ValueError(
			f"Channel count mismatch vs reference: ref={len(ref)} current={len(current)} in '{file_path}'"
		)
	for i, (r, c) in enumerate(zip(ref, current)):
		if str(r) != str(c):
			# Names differ at specific index → fail with explicit message
			raise ValueError(
				f"Channel name mismatch at position {i}: ref='{r}' current='{c}' in '{file_path}'"
			)


def write_manifest(out_dir: str, rows: List[Tuple[str, str, int, int, int]]) -> str:
	"""Write manifest.csv with columns vec_file,source_file,E,C,P."""
	path = os.path.join(out_dir, "manifest.csv")  # Target path
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)  # Simple CSV writer (no quoting needed)
		writer.writerow(["vec_file", "source_file", "E", "C", "P"])  # Header row
		writer.writerows(rows)  # One row per processed input
	return path  # Return path for logging


def main(argv: Optional[Sequence[str]] = None) -> int:
	# Parse CLI arguments for input/output paths and options
	parser = argparse.ArgumentParser(description="Vectorize wSMI (E,C,C) to feature matrices (E,P) using upper triangle.")
	parser.add_argument("--input-dir", required=True, help="Directory containing input .npz files with wSMI arrays")
	parser.add_argument("--glob", default="*.npz", help="Glob pattern for input files (default: *.npz)")
	parser.add_argument(
		"--out-dir",
		required=True,
		help="Output directory (required)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting existing *_vec.npz and global outputs",
	)
	args = parser.parse_args(argv)  # Supports programmatic calls by passing argv

	setup_logging()  # Initialize logging once

	input_dir = os.path.abspath(args.input_dir)  # Normalize input path
	out_dir = os.path.abspath(args.out_dir)  # Required output path

	os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists

	files = list_npz_files(input_dir, args.glob)  # Collect input files to process
	logging.info(f"Found {len(files)} .npz files in '{input_dir}' (pattern='{args.glob}')")  # Report discovery
	if not files:
		logging.error("No input files found.")  # Early exit on empty input
		return 1

	# Reference metadata determined by the first successful file
	ref_C: Optional[int] = None  # Reference channel count
	ref_ch_names: Optional[List[str]] = None  # Reference channel names
	I: Optional[np.ndarray] = None  # Upper-triangle row indices
	J: Optional[np.ndarray] = None  # Upper-triangle col indices
	pairs_names: Optional[List[Tuple[str, str]]] = None  # Optional pairs name list

	manifest_rows: List[Tuple[str, str, int, int, int]] = []  # Records summary per output file
	total_E = 0  # Running sum of epochs over all processed inputs
	processed_count = 0  # Count of saved outputs (skipped files are not counted)

	for path in files:  # Process each candidate file
		base = os.path.splitext(os.path.basename(path))[0]  # Strip extension for output naming
		out_path = os.path.join(out_dir, f"{base}_vec.npz")  # Target vectorized file
		out_exists = os.path.exists(out_path)  # Detect prior output
		skip_save_only = out_exists and not args.overwrite  # Decide whether to skip saving

		t0 = time.time()  # Start timer for this file
		try:
			loaded = load_wsmi(path)  # Load data and optional channel names
			ecc, E, C = ensure_ecc(loaded.data)  # Normalize to (E, C, C)

			# Validate passthrough arrays against detected dimensions
			if "events" in loaded.extras:
				ev_arr = loaded.extras["events"]
				try:
					n_ev = int(ev_arr.shape[0]) if hasattr(ev_arr, "shape") else len(ev_arr)
				except Exception:
					n_ev = len(ev_arr)  # Best-effort fallback
				if n_ev != int(E):
					raise ValueError(f"events length mismatch: expected E={E}, got {n_ev}")
			# Validate raw channel names length when present
			if loaded.raw_ch_names is not None:
				try:
					n_ch = int(loaded.raw_ch_names.shape[0]) if hasattr(loaded.raw_ch_names, "shape") else len(loaded.raw_ch_names)
				except Exception:
					n_ch = len(loaded.raw_ch_names)
				if n_ch != int(C):
					raise ValueError(f"ch_names length mismatch: expected C={C}, got {n_ch}")

			# Initialize reference on first successful file
			if ref_C is None:
				ref_C = C  # Lock channel count
				ref_ch_names = loaded.ch_names[:] if loaded.ch_names is not None else None  # Lock channel names if present
				I, J, pairs_names = compute_pairs(C, ref_ch_names)  # Precompute index and name pairs

				# Write global pairs files (overwrite controlled by flag)
				if args.overwrite or not os.path.exists(os.path.join(out_dir, "pairs_idx.npy")):
					idx_path, json_path = write_pairs(out_dir, I, J, pairs_names)  # Save indices and name mapping
					logging.info(f"Wrote pairs indices -> {idx_path}")  # Report location
					if json_path:
						logging.info(f"Wrote pairs names   -> {json_path}")  # Report name mapping path
			else:
				# Validate consistent channel count and names for subsequent files
				if C != ref_C:
					logging.error(
						f"Channel count mismatch: expected C={ref_C}, got C={C} in '{path}'"
					)  # Hard failure to prevent mixing incompatible files
					return 2
				validate_ch_names(ref_ch_names, loaded.ch_names, path)  # Ensure names align exactly

			assert I is not None and J is not None  # Indices must be available now

			# Vectorize and validate
			X = vectorize_upper(ecc, I, J)  # Produce (E, P) feature matrix
			P = X.shape[1]  # Number of features per epoch
			if not (P == len(I) == len(J)):
				raise ValueError(
					f"Vectorization size mismatch: P={P}, len(I)={len(I)}, len(J)={len(J)}"
				)  # Defensive: ensure shapes align
			# Basic consistency checks
			if loaded.ch_names is not None and len(loaded.ch_names) != C:
				raise ValueError(
					f"Loaded channel names length does not match C: len(ch_names)={len(loaded.ch_names)}, C={C}"
				)  # Per-file internal consistency
			if ref_ch_names is not None and len(ref_ch_names) != C:
				raise ValueError(
					f"Channel names length does not match C: len(ch_names)={len(ref_ch_names)}, C={C}"
				)  # Reference consistency

			# Log preserved fields if any
			preserved_keys = [k for k in ("events", "event_labels", "event_codes", "event_names", "meta_json") if k in loaded.extras]
			if preserved_keys:
				logging.info(f"Preserved fields: {preserved_keys}")

			if skip_save_only:
				# Do not overwrite existing vec file; still record manifest info
				logging.info(
					f"Output exists, skipping save (use --overwrite to replace): {os.path.basename(out_path)}"
				)
				manifest_rows.append((os.path.basename(out_path), os.path.basename(path), int(E), int(C), int(P)))  # Track in manifest
			else:
				# Save new or overwritten vectorized output
				save_vectors(out_path, X, E, C, P, path, loaded.raw_ch_names, loaded.extras)
				dt = time.time() - t0  # Elapsed time for processing
				logging.info(
					f"Saved {os.path.basename(out_path)} with X.shape={X.shape} in {dt:.2f}s"
				)  # Report success and timing
				manifest_rows.append((os.path.basename(out_path), os.path.basename(path), int(E), int(C), int(P)))  # Append manifest row
				processed_count += 1  # Count as produced output

			total_E += int(E)  # Aggregate epochs across all files

		except Exception as e:
			# Fail fast: log and abort on first error instead of continuing
			logging.error(f"Aborting due to error in file '{os.path.basename(path)}': {e}")
			return 6

	if not manifest_rows:
		logging.error("No files were included in the manifest; nothing processed or all files failed.")  # Nothing useful produced
		return 3

	# Write manifest
	manifest_path = write_manifest(out_dir, manifest_rows)  # Persist manifest
	logging.info(f"Wrote manifest -> {manifest_path}")  # Report path

	# Post validation: all P identical and total E matches sum
	unique_P = {row[4] for row in manifest_rows}  # Collect unique P values
	if len(unique_P) != 1:
		logging.error(f"Inconsistent number of columns P across outputs: {sorted(unique_P)}")  # Mixed feature widths
		return 4
	sum_E = sum(row[2] for row in manifest_rows)  # Sum epochs from manifest
	if sum_E != total_E:
		logging.error(f"Internal epoch count mismatch: computed={total_E}, manifest_sum={sum_E}")  # Sanity check failed
		return 5

	logging.info(
		f"Done. Processed {processed_count} files. Total epochs={total_E}. Features per epoch (P)={unique_P.pop()}"
	)  # Final summary for quick inspection
	return 0  # Success


if __name__ == "__main__":  # Allow import without executing CLI
	sys.exit(main())  # Return appropriate exit code to the OS
