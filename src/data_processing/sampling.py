"""Draws the exact train/val row counts for one intersection type from its raw CSVs."""

import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.dataset_files import count_samples

CHUNK_SIZE = 10_000
Paths = Tuple[Optional[str], Optional[str]]


def iter_chunks(files: List[str], chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Chunks of several CSVs read back to back, as if they were one file."""
    for path in files:
        yield from pd.read_csv(path, chunksize=chunksize)


def _append(frame: pd.DataFrame, path: str) -> int:
    if not frame.empty:
        frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(frame)


def allocate_counts(
    train_total: int,
    val_total: int,
    distributions: Dict[str, float],
    available_types: Iterable[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Split train/val totals across intersection types by percentage.

    Types are processed from largest share to smallest; the last configured one absorbs the
    rounding remainder so the totals are exact. Types without a configured share get 0.
    """
    total_pct = sum(distributions.values())
    if abs(total_pct - 100.0) > 0.001:
        raise ValueError(f"Intersection distributions must sum to 100%, got {total_pct}%")

    available = list(available_types)
    train_targets = {t: 0 for t in available}
    val_targets = {t: 0 for t in available}

    configured = sorted((t for t in available if t in distributions), key=lambda t: distributions[t], reverse=True)
    allocated_train = allocated_val = 0
    for i, t in enumerate(configured):
        if i == len(configured) - 1:
            train_targets[t] = train_total - allocated_train
            val_targets[t] = val_total - allocated_val
        else:
            share = distributions[t] / 100.0
            train_targets[t] = int(train_total * share)
            val_targets[t] = int(val_total * share)
            allocated_train += train_targets[t]
            allocated_val += val_targets[t]

    if sum(train_targets.values()) != train_total or sum(val_targets.values()) != val_total:
        missing = sorted(set(distributions) - set(available))
        raise ValueError(
            f"Sample allocation error: got {sum(train_targets.values())}+{sum(val_targets.values())}, "
            f"expected {train_total}+{val_total}"
            + (f" (no data found for configured types: {missing})" if missing else "")
        )
    return train_targets, val_targets


def sample_regular(files: List[str], train_target: int, val_target: int, path_prefix: str) -> Paths:
    """First rows (after a seeded per-chunk shuffle) of the files: train first, then val.

    Asks for fewer rows than requested when the files are too small; `fit_file_to_count` tops up.
    """
    available = sum(count_samples(f) for f in files)
    needed = train_target + val_target
    if needed > available:
        train_target, val_target = int(train_target * available / needed), int(val_target * available / needed)
        print(f"    Scaled down to {train_target} train + {val_target} val samples")

    train_path, val_path = f"{path_prefix}_train.csv", f"{path_prefix}_val.csv"
    n_train = n_val = 0
    for chunk in iter_chunks(files):
        if n_train >= train_target and n_val >= val_target:
            break
        chunk = chunk.sample(frac=1, random_state=42)
        train_part = chunk.iloc[:max(train_target - n_train, 0)]
        val_part = chunk.iloc[len(train_part):len(train_part) + max(val_target - n_val, 0)]
        n_train += _append(train_part, train_path)
        n_val += _append(val_part, val_path)

    return (train_path if n_train else None), (val_path if n_val else None)


def _water_fill(counts: np.ndarray, total: int) -> np.ndarray:
    """Per-bin quotas as equal as possible, capped by availability, summing to `total`."""
    quota = np.zeros_like(counts)
    while (rest := total - quota.sum()) > 0:
        open_bins = np.flatnonzero(quota < counts)
        share = rest // len(open_bins)
        if share == 0:
            quota[open_bins[:rest]] += 1
        else:
            quota[open_bins] = np.minimum(counts[open_bins], quota[open_bins] + share)
    return quota


def sample_log_uniform(files: List[str], train_target: int, val_target: int, path_prefix: str,
                       volume_range: Tuple[float, float], n_bins: int) -> Paths:
    """Draws rows so IntersectionVolume is as close to log-uniform over `volume_range` as the data allows.

    Bins holding fewer rows than an equal share keep all of them and pass the deficit to the other bins;
    nothing is synthesised, and every row is used at most once, so train and val never overlap.
    """
    vol_min, vol_max = volume_range
    rng = np.random.default_rng(42)

    cols = pd.concat(pd.read_csv(f, usecols=["IntersectionVolume", "HasIntersection"]) for f in files)
    volume = cols["IntersectionVolume"].values
    valid = (cols["HasIntersection"].values == 1) & (volume >= vol_min) & (volume <= vol_max)
    edges = np.logspace(np.log10(vol_min), np.log10(vol_max), n_bins + 1)
    bin_id = np.clip(np.searchsorted(edges, volume, side="right") - 1, 0, n_bins - 1)

    total = int(min(train_target + val_target, valid.sum()))
    counts = np.bincount(bin_id[valid], minlength=n_bins)
    quota = _water_fill(counts, total)
    print(f"    {int((counts < total / n_bins).sum())}/{n_bins} volume bins hold fewer rows than an equal share")

    chosen = np.zeros(len(volume), dtype=bool)
    for b in np.flatnonzero(quota):
        chosen[rng.choice(np.flatnonzero(valid & (bin_id == b)), quota[b], replace=False)] = True

    rows, offset = [], 0
    for chunk in iter_chunks(files):
        rows.append(chunk[chosen[offset:offset + len(chunk)]])
        offset += len(chunk)
    frame = pd.concat(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    n_train = total * train_target // (train_target + val_target)
    train_path, val_path = f"{path_prefix}_train.csv", f"{path_prefix}_val.csv"
    return (
        train_path if _append(frame.iloc[:n_train], train_path) else None,
        val_path if _append(frame.iloc[n_train:], val_path) else None,
    )


def fit_file_to_count(source_file: str, target_count: int, output_file: str) -> str:
    """Write `output_file` with exactly `target_count` rows of `source_file`.

    Extra rows are dropped from the end; missing rows are made up by resampling existing ones.
    Returns the path to use afterwards (the source itself when it already matches).
    """
    current_count = count_samples(source_file)
    if current_count == target_count:
        return source_file
    if current_count == 0:
        raise ValueError(f"Cannot reach {target_count} rows: {source_file} is empty")

    if current_count > target_count:
        print(f"    Truncating from {current_count} to {target_count}")
        pd.read_csv(source_file, nrows=target_count).to_csv(output_file, index=False)
    else:
        print(f"    Oversampling from {current_count} to {target_count}")
        data = pd.read_csv(source_file)
        extra = data.sample(n=target_count - current_count, replace=target_count - current_count > len(data), random_state=42)
        pd.concat([data, extra], ignore_index=True).to_csv(output_file, index=False)

    final_count = count_samples(output_file)
    if final_count != target_count:
        raise ValueError(f"Failed to create exact count file: got {final_count}, wanted {target_count}")
    return output_file
