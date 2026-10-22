import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.data_processing.augmentation import PERMUTE, SWAP, augment_record
from src.data_processing.config import ProcessorConfig
from src.data_processing.normalization import normalize_types
from src.data_processing.sampling.base import SampleResult
from src.data_processing.sampling.regular import CHUNK_SIZE, RegularSampler


@dataclass
class _Bins:
    """Per-bin sample lists (train/val) plus the volume edges and per-bin targets."""

    edges: np.ndarray
    targets: List[tuple]
    train: List[list]
    val: List[list]

    @property
    def n_bins(self) -> int:
        return len(self.targets)

    def pooled(self, idx: int) -> list:
        return self.train[idx] + self.val[idx]

    def is_full(self) -> bool:
        return all(
            len(self.train[i]) >= t and len(self.val[i]) >= v
            for i, (t, v) in enumerate(self.targets)
        )

    def needs_samples(self, idx: int) -> bool:
        train_target, val_target = self.targets[idx]
        return len(self.train[idx]) < train_target or len(self.val[idx]) < val_target

    def label(self, idx: int, fmt: str = ".2e") -> str:
        return f"[{self.edges[idx]:{fmt}}, {self.edges[idx + 1]:{fmt}}]"


def _choose(n_available: int, n_wanted: int) -> np.ndarray:
    return np.random.choice(n_available, n_wanted, replace=n_wanted > n_available)


class LogUniformVolumeSampler:
    """Fills equal-width log-volume bins with intersecting samples, topping up with augmented ones.

    Real samples are always preferred; augmentation only fills the gaps in under-populated bins.
    """

    def __init__(self, config: ProcessorConfig):
        self.config = config

    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir) -> SampleResult:
        print("      Applying LOG-UNIFORM volume distribution sampling")

        if not self.config.volume_range:
            print("        No volume_range specified, falling back to regular sampling")
            return RegularSampler(self.config).sample(source_file, intersection_type, train_target, val_target, temp_dir)

        n_bins = self.config.n_volume_bins
        vol_min, vol_max = self.config.volume_range
        edges = np.logspace(np.log10(vol_min), np.log10(vol_max), n_bins + 1)

        print(f"        Volume range: [{vol_min:.8e}, {vol_max:.8e}]")
        print(f"        Spans {np.log10(vol_max / vol_min):.1f} decades across {n_bins} log-uniform bins")
        print(f"        Targets: {train_target} train + {val_target} val")

        bins = _Bins(
            edges=edges,
            targets=self._bin_targets(train_target, val_target, n_bins),
            train=[[] for _ in range(n_bins)],
            val=[[] for _ in range(n_bins)],
        )
        for i, (t, v) in enumerate(bins.targets):
            print(f"          Bin {i + 1:2d} {bins.label(i, '.1e')}: {t} train + {v} val")

        print("        Phase 1: Collecting real samples...")
        empty_bins = self._collect_real_samples(source_file, bins, vol_min, vol_max)
        real_count = sum(len(bins.pooled(i)) for i in range(n_bins))

        print("        Phase 2: Geometric augmentation...")
        self._fill_gaps(bins, empty_bins)

        print("        Phase 3: Writing samples...")
        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        train_count = self._write(bins.train, train_file)
        val_count = self._write(bins.val, val_file)

        total = train_count + val_count
        real_pct = real_count / total * 100 if total else 0
        print(f"        Final result: {train_count} train + {val_count} val "
              f"({real_count} real = {real_pct:.1f}%, {total - real_count} augmented)")

        return SampleResult(
            train_file if train_count > 0 else None,
            val_file if val_count > 0 else None,
            train_count,
            val_count,
        )

    @staticmethod
    def _bin_targets(train_target, val_target, n_bins):
        total = train_target + val_target
        per_bin, remainder = divmod(total, n_bins)
        train_ratio = train_target / total if total > 0 else 0.9

        targets = []
        for i in range(n_bins):
            in_bin = per_bin + (1 if i < remainder else 0)
            train_in_bin = int(in_bin * train_ratio)
            targets.append((train_in_bin, in_bin - train_in_bin))
        return targets

    # ---- Phase 1 -------------------------------------------------------------------------

    def _collect_real_samples(self, source_file, bins: _Bins, vol_min, vol_max) -> List[int]:
        """Stream the file, assigning intersecting samples to bins. Returns indices of bins that stayed empty."""
        upper_edges = bins.edges[1:]

        for chunk in pd.read_csv(source_file, chunksize=CHUNK_SIZE):
            if "HasIntersection" not in chunk.columns or "IntersectionVolume" not in chunk.columns:
                continue

            volume = chunk["IntersectionVolume"]
            valid = chunk[(chunk["HasIntersection"] == 1) & (volume >= vol_min) & (volume <= vol_max)]
            if valid.empty:
                continue

            valid = valid.sample(frac=1, random_state=42).reset_index(drop=True)
            bin_indices = np.minimum(np.searchsorted(upper_edges, valid["IntersectionVolume"].values, side="right"), bins.n_bins - 1)

            for idx, record in zip(bin_indices, valid.to_dict("records")):
                train_target, val_target = bins.targets[idx]
                if len(bins.train[idx]) < train_target:
                    bins.train[idx].append(record)
                elif len(bins.val[idx]) < val_target:
                    bins.val[idx].append(record)

            if bins.is_full():
                print("        All log-bins filled with real data!")
                break

        empty_bins = [i for i in range(bins.n_bins) if bins.needs_samples(i) and not bins.pooled(i)]
        n_short = sum(bins.needs_samples(i) for i in range(bins.n_bins))
        print(f"        Collected {sum(len(bins.pooled(i)) for i in range(bins.n_bins))} real samples; "
              f"{n_short}/{bins.n_bins} bins need augmentation ({len(empty_bins)} empty)")
        return empty_bins

    # ---- Phase 2 -------------------------------------------------------------------------

    def _fill_gaps(self, bins: _Bins, empty_bins: List[int]):
        for idx in range(bins.n_bins):
            train_target, val_target = bins.targets[idx]
            train_needed = train_target - len(bins.train[idx])
            val_needed = val_target - len(bins.val[idx])
            needed = train_needed + val_needed
            if train_needed <= 0 and val_needed <= 0:
                continue

            print(f"          Bin {idx + 1} {bins.label(idx)} needs {train_needed} train + {val_needed} val")
            new_samples = []

            own = bins.pooled(idx)
            if own:
                new_samples += self._variants(own, min(needed, len(own) * 2))

            if len(new_samples) < needed:
                new_samples += self._borrow_from_neighbours(bins, idx, needed - len(new_samples))

            if idx in empty_bins and len(new_samples) < needed:
                new_samples += self._oversample_nearest(bins, idx, needed - len(new_samples))

            if len(new_samples) < needed:
                new_samples += self._cross_bin(bins, idx, needed - len(new_samples))

            for i, sample in enumerate(new_samples[:needed]):
                (bins.train if i < train_needed else bins.val)[idx].append(sample)
            print(f"            Added {min(len(new_samples), needed)} samples")

    @staticmethod
    def _variants(samples, count):
        """Randomly swapped/permuted copies of `samples` (60% swap, 40% permute)."""
        if not samples or count <= 0:
            return []

        variants = []
        for i in _choose(len(samples), count):
            kind = str(np.random.choice([SWAP, PERMUTE], p=[0.6, 0.4]))
            variants.append(augment_record(samples[i], kind))
        return variants

    @staticmethod
    def _borrow_from_neighbours(bins: _Bins, target_idx, count):
        """Copy samples from the log-volume-closest bins, taking at most ~30% of a bin scaled by similarity."""
        center = np.sqrt(bins.edges[target_idx] * bins.edges[target_idx + 1])

        candidates = []
        for offset in range(1, bins.n_bins):
            for direction in (-1, 1):
                neighbour = target_idx + direction * offset
                if 0 <= neighbour < bins.n_bins:
                    neighbour_center = np.sqrt(bins.edges[neighbour] * bins.edges[neighbour + 1])
                    similarity = 1.0 / (1.0 + abs(np.log10(neighbour_center / center)))
                    candidates.append((neighbour, similarity))
        candidates.sort(key=lambda c: c[1], reverse=True)

        borrowed = []
        for neighbour, similarity in candidates:
            if len(borrowed) >= count:
                break
            available = bins.pooled(neighbour)
            if not available:
                continue

            n_take = min(max(1, int(len(available) * similarity * 0.3)), count - len(borrowed), len(available))
            for i in np.random.choice(len(available), n_take, replace=False):
                borrowed.append(dict(available[i]))
        return borrowed

    @staticmethod
    def _oversample_nearest(bins: _Bins, target_idx, count):
        """For an empty bin: permuted copies of samples from the nearest populated bin."""
        populated = [i for i in range(bins.n_bins) if i != target_idx and bins.pooled(i)]
        if not populated:
            print("            Warning: No populated bins found for oversampling")
            return []

        nearest = min(populated, key=lambda i: abs(i - target_idx))
        source = bins.pooled(nearest)
        return [augment_record(source[i], PERMUTE) for i in _choose(len(source), count)]

    @staticmethod
    def _cross_bin(bins: _Bins, target_idx, count):
        """Last resort: randomly swapped/permuted copies of samples from every other bin."""
        pool = [s for i in range(bins.n_bins) if i != target_idx for s in bins.pooled(i)]
        if not pool:
            return []

        augmented = []
        for i in _choose(len(pool), count):
            kind = str(np.random.choice([SWAP, PERMUTE]))
            augmented.append(augment_record(pool[i], kind))
        return augmented

    # ---- Phase 3 -------------------------------------------------------------------------

    def _write(self, bin_samples: List[list], output_file) -> int:
        samples = [s for bin_list in bin_samples for s in bin_list]
        if not samples:
            return 0

        df = normalize_types(pd.DataFrame(samples), self.config.volume_scale_factor)
        df.to_csv(output_file, index=False, float_format="%.17g")
        return len(samples)
