import os
import tempfile
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd

from src.data_processing.allocation import allocate_counts
from src.data_processing.augmentation import augment_data
from src.data_processing.config import ProcessorConfig
from src.data_processing.exact_count import fit_file_to_count
from src.data_processing.normalization import normalize_types
from src.data_processing.reporting import report_distribution
from src.data_processing.sampling import create_sampler
from src.data_processing.transformations import transform_data
from src.utils.file_management import (
    INTERSECTION_TYPES,
    combine_files_streaming,
    count_raw_data,
    count_samples,
    discover_raw_files,
    processed_split_path,
)

CHUNK_SIZE = 10_000
COMBINE_CHUNK_SIZE = 50_000


class CDataProcessor:
    """Turns raw per-intersection-type CSVs into train/val CSVs.

    Flow: gather raw files -> sample exact per-type counts -> augment -> transform -> write.
    """

    def __init__(self, processor_config: Union[dict, ProcessorConfig]):
        if isinstance(processor_config, ProcessorConfig):
            self.config = processor_config
        else:
            self.config = ProcessorConfig.from_dict(processor_config)

    def process(self) -> None:
        print("-- Processing Data --")
        cfg = self.config

        total_raw = count_raw_data(cfg.raw_data_path)
        total_target = cfg.num_train_samples + cfg.num_val_samples
        if total_raw < total_target:
            print(f"WARNING: Raw data ({total_raw:,}) insufficient for target ({total_target:,})")
        print(f"Target: {cfg.num_train_samples} train, {cfg.num_val_samples} val samples")

        with tempfile.TemporaryDirectory(prefix="data_processor_") as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            print("\n=== Stage 1: Gathering raw files ===")
            raw_files = self._collect_raw_files(temp_dir)

            print("\n=== Stage 2: Sampling ===")
            train_files, val_files = self._sample(raw_files, temp_dir)

            print("\n=== Stage 3: Augmentation & transformation ===")
            train_files, val_files = self._augment_and_transform(train_files, val_files, temp_dir)

            print("\n=== Stage 4: Writing output ===")
            self._write_output(train_files, val_files)

        print("-- Data Processing Complete --")

    # ---- Stage 1 -------------------------------------------------------------------------

    def _collect_raw_files(self, temp_dir: str) -> Dict[str, str]:
        """One raw CSV path per intersection type (several source files are merged into one)."""
        raw_files = {}

        for intersection_type in INTERSECTION_TYPES:
            files = discover_raw_files(self.config.raw_data_path, intersection_type)
            if not files:
                print(f"  Warning: No files found for {intersection_type}")
            elif len(files) == 1:
                raw_files[intersection_type] = files[0]
                print(f"  Found {intersection_type}: {os.path.basename(files[0])}")
            else:
                print(f"  Found {len(files)} files for {intersection_type}, combining...")
                combined = self._combine_raw_files(files, intersection_type, temp_dir)
                if combined:
                    raw_files[intersection_type] = combined

        if not raw_files:
            raise FileNotFoundError("No intersection files found for processing.")
        return raw_files

    def _combine_raw_files(self, files: List[str], intersection_type: str, temp_dir: str):
        combined_file = os.path.join(temp_dir, f"{intersection_type}_combined_raw.csv")
        total = 0

        try:
            for path in files:
                file_total = 0
                for chunk in pd.read_csv(path, chunksize=COMBINE_CHUNK_SIZE):
                    chunk = normalize_types(chunk)  # volume scaling happens later, in the samplers
                    if len(chunk) > 0:
                        chunk.to_csv(combined_file, mode="a", header=total == 0, index=False, float_format="%.17g")
                        total += len(chunk)
                        file_total += len(chunk)
                print(f"    {os.path.basename(path)}: {file_total} samples")
        except Exception as e:
            print(f"    Error combining files: {e}")
            return None

        print(f"    Total combined: {total} samples")
        return combined_file if total > 0 else None

    # ---- Stage 2 -------------------------------------------------------------------------

    def _sample(self, raw_files: Dict[str, str], temp_dir: str) -> Tuple[List[str], List[str]]:
        cfg = self.config

        distributions = dict(cfg.intersection_distributions)
        if not distributions:
            print("  No intersection_distributions found, using equal distribution")
            distributions = {t: 100.0 / len(raw_files) for t in raw_files}

        train_targets, val_targets = allocate_counts(
            cfg.num_train_samples, cfg.num_val_samples, distributions, raw_files
        )
        for t in raw_files:
            print(f"  {t}: {train_targets[t]} train + {val_targets[t]} val "
                  f"(configured {distributions.get(t, 0)}%)")

        train_files, val_files = [], []
        for intersection_type, source_file in raw_files.items():
            train_target, val_target = train_targets[intersection_type], val_targets[intersection_type]
            if train_target == 0 and val_target == 0:
                continue

            print(f"\n  Processing {intersection_type}: {train_target} train + {val_target} val required")
            result = create_sampler(intersection_type, cfg).sample(
                source_file, intersection_type, train_target, val_target, temp_dir
            )
            print(f"    Sampled {result.train_count} train + {result.val_count} val")

            for split, file, target, files in (
                ("train", result.train_file, train_target, train_files),
                ("val", result.val_file, val_target, val_files),
            ):
                if file:
                    exact_path = os.path.join(temp_dir, f"{intersection_type}_{split}_exact.csv")
                    files.append(fit_file_to_count(file, target, exact_path))

        train_total = sum(count_samples(f) for f in train_files)
        val_total = sum(count_samples(f) for f in val_files)
        print(f"\n  Sampled {train_total} train + {val_total} val (target {cfg.num_train_samples} + {cfg.num_val_samples})")
        if train_total != cfg.num_train_samples or val_total != cfg.num_val_samples:
            raise ValueError(f"Failed to achieve exact sample counts: got {train_total}+{val_total}")

        return train_files, val_files

    # ---- Stage 3 -------------------------------------------------------------------------

    def _augment_and_transform(self, train_files, val_files, temp_dir):
        cfg = self.config
        if not cfg.has_augmentation and not cfg.transformations:
            print("Skipping augmentation and transformation (not configured)")
            return train_files, val_files

        if cfg.has_augmentation:
            print(f"  Augmentation: point-wise {cfg.point_wise_pct}%, tetrahedron-wise {cfg.tetrahedron_wise_pct}%")
        if cfg.transformations:
            print(f"  Transformations: {', '.join(cfg.transformations)}")

        def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            if cfg.has_augmentation:
                chunk = augment_data(chunk, cfg.point_wise_pct, cfg.tetrahedron_wise_pct)
            if cfg.transformations:
                chunk = transform_data(chunk, cfg.transformations)
            return chunk

        return (
            self._map_chunks(train_files, temp_dir, "train", process_chunk),
            self._map_chunks(val_files, temp_dir, "val", process_chunk),
        )

    @staticmethod
    def _map_chunks(files: List[str], temp_dir: str, split: str,
                    fn: Callable[[pd.DataFrame], pd.DataFrame]) -> List[str]:
        """Apply `fn` to each chunk of each file, writing one new temp file per input file."""
        outputs = []
        for idx, path in enumerate(files):
            output = os.path.join(temp_dir, f"{split}_processed_{idx}.csv")
            for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE)):
                fn(chunk).to_csv(output, mode="a", header=i == 0, index=False)
            outputs.append(output)
        return outputs

    # ---- Stage 4 -------------------------------------------------------------------------

    def _write_output(self, train_files: List[str], val_files: List[str]) -> None:
        if not train_files and not val_files:
            print("  No processed files to save!")
            return

        for split, label, files in (("train", "Training", train_files), ("val", "Validation", val_files)):
            output = processed_split_path(self.config.processed_data_path, split)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if not files:
                continue

            print(f"Combining {len(files)} {split} files...")
            combine_files_streaming(files, output, chunk_size=COMBINE_CHUNK_SIZE)
            print(f"{label} data saved: {count_samples(output)} samples")
            report_distribution(output, label)
