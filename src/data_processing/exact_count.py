import pandas as pd

from src.utils.file_management import count_samples

TRUNCATE_CHUNK_SIZE = 50_000


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
        print(f"        Truncating from {current_count} to {target_count}")
        _truncate(source_file, target_count, output_file)
    else:
        print(f"        Oversampling from {current_count} to {target_count}")
        _oversample(source_file, target_count - current_count, output_file)

    final_count = count_samples(output_file)
    if final_count != target_count:
        raise ValueError(f"Failed to create exact count file: got {final_count}, wanted {target_count}")
    return output_file


def _truncate(source_file, target_count, output_file):
    written = 0
    for chunk in pd.read_csv(source_file, chunksize=min(target_count, TRUNCATE_CHUNK_SIZE)):
        if written >= target_count:
            break
        chunk = chunk.iloc[:target_count - written]
        chunk.to_csv(output_file, mode="a", header=written == 0, index=False, float_format="%.17g")
        written += len(chunk)


def _oversample(source_file, n_extra, output_file):
    data = pd.read_csv(source_file)
    extra = data.sample(n=n_extra, replace=n_extra > len(data), random_state=42)
    pd.concat([data, extra], ignore_index=True).to_csv(output_file, index=False, float_format="%.17g")
