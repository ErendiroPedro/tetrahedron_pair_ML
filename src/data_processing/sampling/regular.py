import os

import pandas as pd

from src.data_processing.config import ProcessorConfig
from src.data_processing.normalization import normalize_types
from src.data_processing.sampling.base import SampleResult
from src.utils.file_management import count_samples

CHUNK_SIZE = 10_000


class RegularSampler:
    """Takes the first rows (after a seeded per-chunk shuffle) of a file as train, then val."""

    def __init__(self, config: ProcessorConfig):
        self.config = config

    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir) -> SampleResult:
        total_available = count_samples(source_file)
        if total_available == 0:
            return SampleResult(None, None, 0, 0)

        total_needed = train_target + val_target
        if total_needed > total_available:
            scale = total_available / total_needed
            train_target = int(train_target * scale)
            val_target = int(val_target * scale)
            print(f"  Scaled down to {train_target} train + {val_target} val samples")

        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        train_count, val_count = self._stream(source_file, train_file, val_file, train_target, val_target)

        return SampleResult(
            train_file if train_count > 0 else None,
            val_file if val_count > 0 else None,
            train_count,
            val_count,
        )

    def _stream(self, source_file, train_file, val_file, train_target, val_target):
        train_count = val_count = 0

        for chunk in pd.read_csv(source_file, chunksize=CHUNK_SIZE):
            if train_count >= train_target and val_count >= val_target:
                break

            chunk = normalize_types(chunk, self.config.volume_scale_factor)
            chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

            n_train = min(max(train_target - train_count, 0), len(chunk))
            n_val = min(max(val_target - val_count, 0), len(chunk) - n_train)

            if n_train > 0:
                chunk.iloc[:n_train].to_csv(train_file, mode="a", header=train_count == 0, index=False, float_format="%.17g")
                train_count += n_train
            if n_val > 0:
                chunk.iloc[n_train:n_train + n_val].to_csv(val_file, mode="a", header=val_count == 0, index=False, float_format="%.17g")
                val_count += n_val

        return train_count, val_count
