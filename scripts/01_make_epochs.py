#!/usr/bin/env python3
"""
Create MNE epochs from raw EEG files with configurable parameters and save them under data/epochs.

This script supports two modes:
- events: event-locked epochs from annotations (e.g., Resting, Familiar voice, Medical staff)
- sliding: fixed-length overlapping windows across continuous data

The script mirrors key preprocessing choices used in prior work:
- Channel renaming to standard 10-20 labels (optional)
- Optional band-pass and notch filtering
- Resampling to a target sampling frequency
- Optional removal of spans annotated as bad (e.g., 'BAD_')

Notes:
- Default event window reflects prior usage (tmin=-0.2, tmax=15.0).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import mne
from tqdm import tqdm
import re


# -------------------------
# Utilities
# -------------------------

def _ensure_dir(p: Path) -> None:
    """Create directory if missing (parents included)."""
    p.mkdir(parents=True, exist_ok=True)


def _maybe_rename_edf_channels(raw: mne.io.BaseRaw, *, apply: bool) -> None:
    """Optionally normalize EDF channel names by removing 'EEG ' prefix and '-REF' suffix.

    This mirrors typical EDF imports encountered in similar pipelines and helps standardize names
    before applying 10-20 montage.
    """
    if not apply:
        return
    chs = raw.info["ch_names"]
    mapping = {}
    for ch in chs:
        new_name = ch
        if new_name.startswith("EEG "):
            new_name = new_name.replace("EEG ", "", 1)
        if new_name.endswith("-REF"):
            new_name = new_name[:-4]
        mapping[ch] = new_name
    if mapping:
        mne.rename_channels(raw.info, mapping)


def _set_montage_and_types(raw: mne.io.BaseRaw) -> None:
    """Assign standard 10-20 montage and set ECG type when an ECG channel is present."""
    # Mark commonly-used ECG name if present
    ch_types = {}
    for ch in raw.info["ch_names"]:
        if ch.upper() in {"ECG", "ECG EKG", "EKG"}:
            ch_types[ch] = "ecg"
    if ch_types:
        raw.set_channel_types(ch_types, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")


def _filter_resample(
    raw: mne.io.BaseRaw,
    *,
    l_freq: Optional[float],
    h_freq: Optional[float],
    notch: Optional[Iterable[float]],
    sfreq: Optional[float],
    picks: str | list | None,
    n_jobs: int = 1, # number of parallel jobs; recommended to be -2 to leave one core free for system use etc.
) -> mne.io.BaseRaw:
    """Apply optional notch, band-pass, and resampling in a stable order.

    MNE requires data to be loaded in memory for filtering/resampling. Only load when needed.
    """
    need_proc = bool(notch) or (l_freq is not None or h_freq is not None) or (
        sfreq is not None and raw.info.get("sfreq") and raw.info["sfreq"] != sfreq
    )
    if not need_proc:
        return raw

    r = raw.copy().load_data()
    if notch:
        r = r.notch_filter(freqs=list(notch), picks=picks, n_jobs=n_jobs, verbose="ERROR")
    if l_freq is not None or h_freq is not None:
        r = r.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, n_jobs=n_jobs, verbose="ERROR")
    if sfreq is not None and r.info.get("sfreq") and r.info["sfreq"] != sfreq:
        r = r.resample(sfreq=sfreq, n_jobs=n_jobs)
    return r


def _reject_bad_spans(raw: mne.io.BaseRaw, *, annot_name: Optional[str]) -> mne.io.BaseRaw:
    """Remove spans annotated with the given name by concatenating remaining segments.

    This matches the idea of excluding e.g., 'BAD_' labeled time periods prior to epoching.
    """
    if not annot_name:
        return raw

    # If no annotations, nothing to reject
    if raw.annotations is None or len(raw.annotations) == 0:
        return raw

    # Build a list of non-bad segments
    segments: List[mne.io.BaseRaw] = []
    t = 0.0
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc == annot_name:
            # Append segment from current t to just before the bad onset
            if onset > t:
                segments.append(raw.copy().crop(tmin=t, tmax=onset, include_tmax=False))
            # Jump past the bad span
            t = onset + duration
    # Append tail segment if any
    if t < raw.times[-1]:
        segments.append(raw.copy().crop(tmin=t))

    if not segments:
        return raw.copy().crop(tmin=0.0, tmax=0.0)  # return an empty Raw if everything was rejected

    return mne.concatenate_raws(segments, verbose=False)


def _sanitize_tag(s: str) -> str:
    """Make a string safe for filesystem paths by replacing non-alnum characters.

    Collapses sequences of non-alphanumerics into single dashes and trims leading/trailing dashes.
    """
    s2 = re.sub(r"[^0-9A-Za-z]+", "-", s)
    return s2.strip("-")


def _fmt_num(n: float) -> str:
    """Format number for compact folder naming: 16 -> 16, -0.2 -> -0p2."""
    s = f"{float(n):g}"
    return s.replace(".", "p")


def _build_spec_subdir(
    *,
    args: argparse.Namespace,
    events_list: Optional[List[str]],
    notch: Optional[Iterable[float]],
    tmin_val: Optional[float] = None,
    tmax_val: Optional[float] = None,
    include_tmin: bool = False,
    include_tmax: bool = False,
) -> Path:
    # Build a single-level directory name summarizing key criteria
    tokens: List[str] = []
    mode_prefix = "events" if args.mode == "events" else "sliding"
    if args.mode == "sliding":
        tokens.append(f"window{_fmt_num(args.window)}s")
        tokens.append(f"stride{_fmt_num(args.stride)}s")
        # Optional crop offsets
        if args.start_offset and args.start_offset != 0.0:
            tokens.append(f"start{_fmt_num(args.start_offset)}s")
        if args.stop_offset is not None:
            tokens.append(f"stop{_fmt_num(args.stop_offset)}s")
    else:  # events
        # Only include tmin/tmax in folder name if the user explicitly provided them
        if include_tmin and tmin_val is not None:
            tokens.append(f"tmin{_fmt_num(tmin_val)}")
        if include_tmax and tmax_val is not None:
            tokens.append(f"tmax{_fmt_num(tmax_val)}")

    # Compose final directory name
    if tokens:
        dirname = f"{mode_prefix}_" + "_".join(tokens)
    else:
        dirname = mode_prefix
    return Path(dirname)


# -------------------------
# Epoch builders
# -------------------------

def build_event_epochs(
    raw: mne.io.BaseRaw,
    *,
    tmin: float,
    tmax: float,
    events_wanted: Optional[List[str]],
    picks: str | list | None,
) -> mne.Epochs:
    """Create event-locked epochs from annotations, optionally filtering by description names."""
    events_from_annot, event_dict = mne.events_from_annotations(raw=raw, verbose="ERROR")
    if events_wanted:
        # Filter event_dict to only include requested labels
        labels = [lab for lab in events_wanted if lab in event_dict]
        event_id = {k: event_dict[k] for k in labels}
        if not event_id:
            raise RuntimeError("None of the requested events are present in this recording.")
    else:
        event_id = event_dict
    if not event_id:
        raise RuntimeError("No recognizable events were found in annotations.")

    epochs = mne.Epochs(
        raw,
        events=events_from_annot,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        event_repeated="merge",
        on_missing="warn",
        verbose="ERROR",
    )
    return epochs


def build_sliding_epochs(
    raw: mne.io.BaseRaw,
    *,
    window: float,
    stride: float,
    start_offset: float,
    stop_offset: Optional[float],
    picks: str | list | None,
) -> mne.Epochs:
    """Create fixed-length overlapping epochs across continuous data using a sliding window."""
    duration = window
    overlap = max(0.0, window - stride)

    # Crop the raw if offsets are requested
    r = raw
    if start_offset or stop_offset:
        tmin = start_offset or raw.times[0]
        tmax = None if stop_offset is None else stop_offset
        r = raw.copy().crop(tmin=tmin, tmax=tmax, verbose="ERROR")

    # Some MNE versions don't support `picks` in make_fixed_length_epochs; pick channels beforehand.
    r2 = r.copy()
    if picks is not None:
        try:
            r2.pick(picks, verbose="ERROR")
        except Exception:
            # Fall back silently if picking fails; better to raise later with a clear message
            pass

    epochs = mne.make_fixed_length_epochs(
        r2,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose="ERROR",
    )
    return epochs


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create epochs under data/epochs with configurable parameters.")

    ap.add_argument("--input-dir", type=str, default="EDF filer", help="Folder with input EDF/FIF files.")
    ap.add_argument("--glob", type=str, default="*.edf", help="Glob pattern to select files.")
    ap.add_argument("--out-dir", type=str, default="data/epochs", help="Output directory for .fif epochs.")

    # Preprocessing
    ap.add_argument("--rename-eeg", action="store_true", help="Normalize EDF channel names (remove 'EEG ' and '-REF').")
    ap.add_argument("--sfreq", type=float, default=None, help="Resample to this sampling frequency (Hz). Omit to keep as-is.")
    ap.add_argument("--l-freq", type=float, default=None, help="High-pass cutoff in Hz (None to skip; default keeps as-is).")
    ap.add_argument("--h-freq", type=float, default=None, help="Low-pass cutoff in Hz (None to skip; default keeps as-is).")
    ap.add_argument("--use-notch", action="store_true", help="Enable notch filtering if --notch-freqs provided.")
    ap.add_argument("--notch-freqs", type=str, default=None, help="Comma-separated notch freqs in Hz, e.g. '50,100'.")
    ap.add_argument("--reject-annot", type=str, default=None, help="Annotation description to reject entirely (e.g., 'BAD_').")
    ap.add_argument("--picks", type=str, default="eeg", help="MNE picks selector (e.g., 'eeg').")

    # Modes
    ap.add_argument("--mode", choices=["events", "sliding"], default="events", help="Epoching mode: events or sliding.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files (default: skip existing).")

    # Events mode
    ap.add_argument("--tmin", type=float, default=None, help="Start time relative to event (s). Omit to use default (-0.2) without naming it in the output folder.")
    ap.add_argument("--tmax", type=float, default=None, help="End time relative to event (s). Omit to use default (15.0) without naming it in the output folder.")
    ap.add_argument(
        "--events",
        type=str,
        default=None,
        help="Comma-separated annotation labels to include (defaults to all present).",
    )

    # Sliding mode
    ap.add_argument("--window", type=float, default=2.5, help="Window length in seconds (sliding mode).")
    ap.add_argument("--stride", type=float, default=2.5, help="Stride in seconds (sliding mode).")
    ap.add_argument("--start-offset", type=float, default=0.0, help="Crop start (s) before sliding windowing.")
    ap.add_argument("--stop-offset", type=float, default=None, help="Crop stop (s) before sliding windowing.")

    # Performance
    ap.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (1=sequential, -1=all cores, -2=all but one core).")

    return ap.parse_args()


def _parse_notch(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return vals or None


def main() -> None:
    args = parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    notch = _parse_notch(args.notch_freqs) if args.use_notch else None

    paths = sorted(in_dir.glob(args.glob))
    n_saved = 0
    for path in tqdm(paths, desc="Epoching files", unit="file"):
        # Load raw file (EDF or FIF)
        try:
            if path.suffix.lower() == ".edf":
                raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            elif path.suffix.lower() == ".fif":
                raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
            else:
                continue  # skip unsupported file types
        except Exception as e:  # pragma: no cover
            tqdm.write(f"[WARN] Skipping {path.name}: {e}")
            continue

        # Optional normalization of channel names and basic metadata
        _maybe_rename_edf_channels(raw, apply=bool(args.rename_eeg))
        _set_montage_and_types(raw)

        # Preprocessing (filter → resample)
        raw_pp = _filter_resample(
            raw,
            l_freq=None if args.l_freq is None else float(args.l_freq),
            h_freq=None if args.h_freq is None else float(args.h_freq),
            notch=notch,
            sfreq=float(args.sfreq) if args.sfreq else None,
            picks=args.picks,
            n_jobs=args.n_jobs,
        )

        # Remove explicitly bad-annotated spans if requested
        raw_pp = _reject_bad_spans(raw_pp, annot_name=args.reject_annot)

        # Build epochs per mode
        try:
            if args.mode == "events":
                # Determine used tmin/tmax and whether user set them explicitly
                user_set_tmin = args.tmin is not None
                user_set_tmax = args.tmax is not None
                used_tmin = float(args.tmin) if user_set_tmin else -0.2 # C&K default
                used_tmax = float(args.tmax) if user_set_tmax else 15.0 # C&K default

                events = None if args.events is None else [x.strip() for x in args.events.split(",") if x.strip()]
                epochs = build_event_epochs(
                    raw_pp, tmin=used_tmin, tmax=used_tmax, events_wanted=events, picks=args.picks
                )
            else:
                epochs = build_sliding_epochs(
                    raw_pp,
                    window=float(args.window),
                    stride=float(args.stride),
                    start_offset=float(args.start_offset),
                    stop_offset=None if args.stop_offset is None else float(args.stop_offset),
                    picks=args.picks,
                )
        except Exception as e:  # pragma: no cover
            tqdm.write(f"[WARN] No epochs saved for {path.name}: {e}")
            continue

        # Construct output directory and filename based on specs
        used_events = None if args.mode != "events" else (None if args.events is None else [x.strip() for x in args.events.split(",") if x.strip()])
        if args.mode == "events":
            spec_dir = _build_spec_subdir(
                args=args,
                events_list=used_events,
                notch=notch,
                tmin_val=used_tmin,
                tmax_val=used_tmax,
                include_tmin=user_set_tmin,
                include_tmax=user_set_tmax,
            )
        else:
            spec_dir = _build_spec_subdir(args=args, events_list=used_events, notch=notch)
        out_spec_dir = out_dir / spec_dir
        _ensure_dir(out_spec_dir)

        base = path.stem
        out_path = out_spec_dir / f"{base}-epo.fif"

        # Save epochs
        try:
            if out_path.exists() and not args.overwrite:
                tqdm.write(f"[SKIP] Exists → {out_path}")
            else:
                epochs.save(out_path, overwrite=bool(args.overwrite))
                n_saved += 1
                tqdm.write(f"[OK] Saved {len(epochs)} epochs → {out_path}")
        except Exception as e:  # pragma: no cover
            tqdm.write(f"[WARN] Failed to save epochs for {path.name}: {e}")
            continue

    tqdm.write(f"[DONE] Saved epochs for {n_saved} file(s) to {out_dir}")


if __name__ == "__main__":
    main()
