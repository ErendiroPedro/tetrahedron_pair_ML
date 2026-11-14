import os
import tempfile
from typing import Dict, List, Tuple

from src.data_processing.config import ProcessorConfig
from src.data_processing.sampling import (
    allocate_counts,
    fit_file_to_count,
    iter_chunks,
    sample_log_uniform,
    sample_regular,
)
from src.data_processing.transformations import transform_data
from src.utils.dataset_files import (
    INTERSECTION_TYPES,
    count_samples,
    discover_csv_files,
    processed_split_path,
)


class DataProcessor:
    """Turns raw per-intersection-type CSVs into train/val CSVs.

    Flow: gather raw files -> sample exact per-type counts -> scale, transform and write.
    """

    def __init__(self, processor_config: dict):
        self.config = ProcessorConfig.from_dict(processor_config)

    def process(self) -> None:
        print("-- Processing Data --")
        config = self.config
        print(f"Target: {config.num_train_samples} train, {config.num_val_samples} val samples")

        with tempfile.TemporaryDirectory(prefix="data_processor_") as temp_dir:
            print("\n=== Stage 1: Gathering raw files ===")
            raw_files = self._collect_raw_files()

            print("\n=== Stage 2: Sampling ===")
            train_files, val_files = self._sample(raw_files, temp_dir)

            print("\n=== Stage 3: Scaling, transformation & writing output ===")
            self._write_output(train_files, val_files)

        print("-- Data Processing Complete --")

    def _collect_raw_files(self) -> Dict[str, List[str]]:
        raw_files = {}
        for intersection_type in INTERSECTION_TYPES:
            files = discover_csv_files(os.path.join(self.config.raw_data_path, intersection_type))
            if files:
                raw_files[intersection_type] = files
                print(f"  Found {len(files)} file(s) for {intersection_type}")
            else:
                print(f"  Warning: No files found for {intersection_type}")

        if not raw_files:
            raise FileNotFoundError("No intersection files found for processing.")
        return raw_files

    def _sample(self, raw_files: Dict[str, List[str]], temp_dir: str) -> Tuple[List[str], List[str]]:
        config = self.config

        distributions = dict(config.intersection_distributions)
        if not distributions:
            print("  No intersection_distributions found, using equal distribution")
            distributions = {t: 100.0 / len(raw_files) for t in raw_files}

        train_targets, val_targets = allocate_counts(
            config.num_train_samples, config.num_val_samples, distributions, raw_files
        )

        train_files, val_files = [], []
        for intersection_type, source_files in raw_files.items():
            train_target, val_target = train_targets[intersection_type], val_targets[intersection_type]
            if train_target == 0 and val_target == 0:
                continue

            print(f"  {intersection_type}: {train_target} train + {val_target} val "
                  f"(configured {distributions.get(intersection_type, 0)}%)")
            prefix = os.path.join(temp_dir, intersection_type)
            if intersection_type == "polyhedron_intersection" and config.volume_range:
                sampled = sample_log_uniform(source_files, train_target, val_target, prefix,
                                             config.volume_range, config.n_volume_bins)
            else:
                sampled = sample_regular(source_files, train_target, val_target, prefix)

            for split, sampled_file, target, split_files in (
                ("train", sampled[0], train_target, train_files),
                ("val", sampled[1], val_target, val_files),
            ):
                if sampled_file:
                    split_files.append(fit_file_to_count(sampled_file, target, f"{prefix}_{split}_exact.csv"))

        train_total = sum(count_samples(f) for f in train_files)
        val_total = sum(count_samples(f) for f in val_files)
        print(f"\n  Sampled {train_total} train + {val_total} val (target {config.num_train_samples} + {config.num_val_samples})")
        if train_total != config.num_train_samples or val_total != config.num_val_samples:
            raise ValueError(f"Failed to achieve exact sample counts: got {train_total}+{val_total}")

        return train_files, val_files

    def _write_output(self, train_files: List[str], val_files: List[str]) -> None:
        config = self.config
        if config.transformations:
            print(f"  Transformations: {', '.join(config.transformations)}")

        for split, files in (("train", train_files), ("val", val_files)):
            output = processed_split_path(config.processed_data_path, split)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if not files:
                continue

            written = 0
            for chunk in iter_chunks(files):
                chunk["IntersectionVolume"] *= config.volume_scale_factor
                chunk = transform_data(chunk, config.transformations)
                chunk.to_csv(output, mode="a" if written else "w", header=not written,
                             index=False, float_format="%.17g")
                written += len(chunk)
            print(f"  {split} data saved: {written} samples -> {output}")
