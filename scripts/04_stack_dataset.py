"""
Stack vectorized wSMI files (*_vec.npz) into a single dataset for clustering or LOSO.

Example:
	python scripts/stack_dataset.py --input-dir <vec_dir> --out-file <out_path>
"""

# Collect all recordings into a common dataset:
# - X (all epochs across subjects),
# - y_cond (condition per epoch: e.g. Familiar/Unfamiliar/Rest), groups (subject ID per epoch for LOSO),
# - eventually run_id/file_id.
#   Save as .npz

from __future__ import annotations  # Allow forward refs in type annotations

import argparse  # Command-line parsing
import logging  # INFO-level logging for progress
import os  # Filesystem operations
import re  # Subject extraction via regex
import sys  # Exit codes
from typing import Dict, List, Optional, Sequence, Tuple  # Type hints

import numpy as np  # Arrays and NPZ IO


def setup_logging() -> None:
	"""Configure a concise INFO logger."""
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def find_vec_files(input_dir: str, pattern: str) -> List[str]:
	"""Find *_vec.npz files in alphabetical order within the given directory.

	Fails if the directory is missing or no files match.
	"""
	import glob

	if not input_dir:
		raise SystemExit("--input-dir is required")
	if not os.path.isdir(input_dir):
		raise SystemExit(f"Input directory does not exist: {input_dir}")
	files = glob.glob(os.path.join(input_dir, pattern))
	files = [f for f in files if f.lower().endswith(".npz")]
	files.sort()
	return files


def load_vec(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Load one *_vec.npz and return (X, event_labels).

	Requires X with shape (E, P) and event_labels with shape (E,).
	Uses allow_pickle=False and fails on any mismatch.
	"""
	with np.load(path, allow_pickle=False) as npz:
		base = os.path.basename(path)
		if "X" not in npz:
			raise ValueError(f"Missing key 'X' in '{base}'")
		X = np.asarray(npz["X"], dtype=np.float32)
		if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
			raise ValueError(f"Invalid 'X' shape in '{base}': expected 2D (E,P) with E>0 and P>0")
		if "event_labels" not in npz:
			raise ValueError(f"Missing key 'event_labels' in '{base}'")
		event_labels = np.asarray(npz["event_labels"])  # expected dtype 'U'
		if event_labels.ndim != 1 or event_labels.shape[0] != X.shape[0]:
			raise ValueError(
				f"Invalid 'event_labels' shape in '{base}': expected (E,) matching X.shape[0], got {tuple(event_labels.shape)}"
			)
	return X, event_labels


def encode_to_ids(values: List[str]) -> Tuple[np.ndarray, List[str]]:
	"""Encode a list of strings to integer IDs based on first-seen order.

	Returns (ids, unique_values) where ids is int64 and unique_values preserves order of first appearance.
	"""
	mapping: Dict[str, int] = {}
	uniques: List[str] = []
	ids = np.empty(len(values), dtype=np.int64)
	for i, v in enumerate(values):
		if v not in mapping:
			mapping[v] = len(mapping)
			uniques.append(v)
		ids[i] = mapping[v]
	return ids, uniques


def find_pairs_paths(input_dir: str) -> Tuple[str, str]:
	"""Try to locate pairs_idx.npy and pairs.json in input_dir or its parent."""
	candidates = [input_dir, os.path.dirname(input_dir)]
	pairs_idx_path = ""
	pairs_json_path = ""
	for d in candidates:
		idx = os.path.join(d, "pairs_idx.npy")
		js = os.path.join(d, "pairs.json")
		if not pairs_idx_path and os.path.isfile(idx):
			pairs_idx_path = idx
		if not pairs_json_path and os.path.isfile(js):
			pairs_json_path = js
	return pairs_idx_path, pairs_json_path


def stack_all(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str], np.ndarray, np.ndarray, int, int, np.ndarray]:
	"""Load, validate, and stack all *_vec.npz files.

	Returns (X_all, y_cond, groups, subjects_list, conditions_list, files_out, file_index, file_ptr, P, N, event_labels_all).
	"""
	X_list: List[np.ndarray] = []
	files_out: List[str] = []
	file_index_list: List[int] = []
	file_ptr_vals: List[int] = [0]
	subj_per_row: List[str] = []
	labels_per_row: List[str] = []

	ref_P: Optional[int] = None

	subj_regex = re.compile(r"patient(\d+)")

	for fi, path in enumerate(files):
		base = os.path.basename(path)
		# Subject extraction from filename
		m = subj_regex.search(base)
		if not m:
			raise ValueError(f"Subject regex 'patient(\\d+)' did not match filename '{base}'")
		subj_code = f"P{int(m.group(1)):02d}"

		# Load arrays with strict validation
		X, ev_labels = load_vec(path)
		E, P_current = X.shape

		if ref_P is None:
			ref_P = P_current
		elif P_current != ref_P:
			raise ValueError(f"Feature size mismatch: expected P={ref_P}, got P={P_current} in '{base}'")

		# Append arrays and indexing info
		X_list.append(X)
		files_out.append(os.path.abspath(path))
		file_index_list.extend([fi] * E)
		file_ptr_vals.append(file_ptr_vals[-1] + E)

		# Extend per-row subjects and event labels
		subj_per_row.extend([subj_code] * E)
		labels_per_row.extend([str(x) for x in ev_labels.tolist()])

	if not X_list:
		raise ValueError("No input *_vec.npz files found")

	X_all = np.vstack(X_list).astype(np.float32, copy=False)
	if X_all.dtype != np.float32:
		raise ValueError("X dtype must be float32")
	N, P = int(X_all.shape[0]), int(ref_P)  # type: ignore[arg-type]

	# Encode to integer IDs
	y_cond, conditions_list = encode_to_ids(labels_per_row)
	groups, subjects_list = encode_to_ids(subj_per_row)

	# Build event_labels_all as Unicode array for downstream inspection
	event_labels_all = np.asarray(labels_per_row, dtype=np.unicode_)

	return (
		X_all,
		y_cond,
		groups,
		subjects_list,
		conditions_list,
		files_out,
		np.asarray(file_index_list, dtype=np.int64),
		np.asarray(file_ptr_vals, dtype=np.int64),
		P,
		N,
		event_labels_all,
	)


def save_npz(out_path: str,
			 X: np.ndarray,
			 y_cond: np.ndarray,
			 groups: np.ndarray,
			 subjects: List[str],
			 conditions: List[str],
			 files: List[str],
			 file_index: np.ndarray,
			 file_ptr: np.ndarray,
			 P: int,
			 N: int,
			 pairs_idx_path: str,
			 pairs_json_path: str,
			 event_labels_all: np.ndarray) -> None:
	"""Save the stacked dataset as a compressed NPZ."""
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	np.savez_compressed(
		out_path,
		X=X,
		y_cond=y_cond,
		groups=groups,
		# Store text arrays as Unicode (dtype 'U') to allow allow_pickle=False on load
		subjects=np.asarray(subjects, dtype=np.unicode_),
		conditions=np.asarray(conditions, dtype=np.unicode_),
		files=np.asarray(files, dtype=np.unicode_),
		file_index=file_index,
		file_ptr=file_ptr,
		P=int(P),
		N=int(N),
		pairs_idx_path=str(pairs_idx_path),
		pairs_json_path=str(pairs_json_path),
		event_labels_all=event_labels_all,
	)
	logging.info(f"Saved dataset -> {out_path}")


def default_out_file(input_dir: str) -> str:
	"""Construct default output path under data/processed/datasets/<dir>_stack.npz."""
	last = os.path.basename(os.path.normpath(input_dir))
	return os.path.join("data", "processed", "datasets", f"{last}_stack.npz")


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Stack vectorized wSMI *_vec.npz into one dataset.")
	parser.add_argument("--input-dir", required=True, help="Folder with *_vec.npz files")
	parser.add_argument("--glob", default="*_vec.npz", help="Glob for selecting input files (default: *_vec.npz)")
	parser.add_argument("--out-file", default=None, help="Output .npz path (default: data/processed/datasets/<lastdir>_stack.npz)")
	args = parser.parse_args(argv)

	setup_logging()

	# Locate vec files
	files = find_vec_files(args.input_dir, args.glob)
	if not files:
		logging.error("No input files found")
		return 1
	logging.info(f"Files discovered: {len(files)}")

	# Stack all and get pairs paths
	try:
		X_all, y_cond, groups, subjects_list, conditions_list, files_out, file_index, file_ptr, P, N, event_labels_all = stack_all(files)
	except Exception as e:
		logging.error(str(e))
		return 2

	in_dir = args.input_dir
	pairs_idx_path, pairs_json_path = find_pairs_paths(in_dir)

	# Determine output path
	out_path = args.out_file if args.out_file else default_out_file(in_dir)

	# Minimal summary and save
	logging.info(f"Files processed: {len(files_out)} | Total rows (N): {N} | Features (P): {P}")
	save_npz(
		out_path,
		X_all,
		y_cond,
		groups,
		subjects_list,
		conditions_list,
		files_out,
		file_index,
		file_ptr,
		P,
		N,
		pairs_idx_path,
		pairs_json_path,
		event_labels_all,
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())