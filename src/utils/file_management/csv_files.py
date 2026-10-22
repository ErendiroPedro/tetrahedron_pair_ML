import gc
import os
import subprocess
from typing import List

import pandas as pd

EXPECTED_COLUMNS = [
    'T1_v1_x', 'T1_v1_y', 'T1_v1_z', 'T1_v2_x', 'T1_v2_y', 'T1_v2_z',
    'T1_v3_x', 'T1_v3_y', 'T1_v3_z', 'T1_v4_x', 'T1_v4_y', 'T1_v4_z',
    'T2_v1_x', 'T2_v1_y', 'T2_v1_z', 'T2_v2_x', 'T2_v2_y', 'T2_v2_z',
    'T2_v3_x', 'T2_v3_y', 'T2_v3_z', 'T2_v4_x', 'T2_v4_y', 'T2_v4_z',
    'IntersectionVolume', 'HasIntersection',
]


def discover_csv_files(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".csv")]


def count_samples(file_path: str) -> int:
    """Number of data rows (header excluded); 0 if the file is missing or unreadable."""
    if not os.path.exists(file_path):
        return 0

    try:
        if os.name == 'posix':
            try:
                result = subprocess.run(['wc', '-l', file_path],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return max(0, int(result.stdout.split()[0]) - 1)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass

        return sum(len(chunk) for chunk in pd.read_csv(file_path, chunksize=10_000))

    except Exception as e:
        print(f"Error counting samples in {file_path}: {e}")
        return 0


def count_samples_in_files(file_list: List[str]) -> int:
    total = 0
    for file_path in file_list:
        if os.path.exists(file_path):
            count = count_samples(file_path)
            total += count
            print(f"    {os.path.basename(file_path)}: {count} samples")
    return total


def count_raw_data(raw_data_path: str) -> int:
    total = 0
    print("Counting raw data files:")

    for intersection_type in sorted(os.listdir(raw_data_path)):
        folder_path = os.path.join(raw_data_path, intersection_type)
        if os.path.isdir(folder_path):
            folder_total = sum(count_samples(f) for f in discover_csv_files(folder_path))
            total += folder_total
            print(f"  {intersection_type} total: {folder_total:,} rows")

    print(f"Total raw data: {total:,} rows")
    return total


def _validate_and_clean_chunk(chunk_df: pd.DataFrame, expected_count: int) -> pd.DataFrame:
    """Trim extra columns, coerce coordinates/volume to float64 and the label to int32, zero-fill NaNs."""
    if chunk_df.empty:
        return chunk_df

    try:
        actual_count = len(chunk_df.columns)

        if actual_count != expected_count:
            print(f"        Column mismatch: expected {expected_count}, got {actual_count}")
            if actual_count > expected_count:
                chunk_df = chunk_df.iloc[:, :expected_count]
                print(f"        Trimmed to first {expected_count} columns")
            else:
                print(f"        WARNING: Missing columns, keeping {actual_count} columns")

        chunk_df = chunk_df.dropna(how='all').copy()

        for col in chunk_df.columns[:-2]:
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').astype('float64')

        if len(chunk_df.columns) >= 2:
            volume_col = chunk_df.columns[-2]
            chunk_df[volume_col] = pd.to_numeric(chunk_df[volume_col], errors='coerce').astype('float64')

        if len(chunk_df.columns) >= 1:
            label_col = chunk_df.columns[-1]
            chunk_df[label_col] = pd.to_numeric(chunk_df[label_col], errors='coerce').astype('int32')

        chunk_df = chunk_df.fillna(0)
        return chunk_df.dropna(how='all')

    except Exception as e:
        print(f"        Error validating chunk: {e}")
        return pd.DataFrame()


def combine_files_streaming(source_files: List[str], output_file: str, chunk_size: int = 50_000) -> bool:
    """Concatenate CSVs into `output_file` in chunks, then re-validate the result."""
    expected_samples = count_samples_in_files(source_files)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"  Removed existing output file: {output_file}")

    print(f"  Combining {len(source_files)} files into {os.path.basename(output_file)}")

    total_written = 0
    first_file = True
    expected_column_count = None

    for file_idx, source_file in enumerate(source_files):
        try:
            if not os.path.exists(source_file):
                print(f"    Skipping missing file: {source_file}")
                continue

            print(f"    Processing file {file_idx + 1}/{len(source_files)}: {os.path.basename(source_file)}")

            file_written = 0
            for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                if first_file and expected_column_count is None:
                    expected_column_count = len(chunk_df.columns)
                    print(f"      Expected columns: {expected_column_count}")

                chunk_df = _validate_and_clean_chunk(chunk_df, expected_column_count)

                if len(chunk_df) > 0:
                    chunk_df.to_csv(output_file, mode='a', header=first_file, index=False,
                                    float_format='%.17g')
                    first_file = False
                    file_written += len(chunk_df)
                    total_written += len(chunk_df)

                del chunk_df
                gc.collect()

            print(f"      Added {file_written} rows from {os.path.basename(source_file)}")

        except Exception as e:
            print(f"    Error processing {source_file}: {e}")
            continue

    print(f"  Final combined file: {total_written} total rows")

    if total_written > 0 and os.path.exists(output_file):
        validate_and_fix_file_consistency(output_file)
        final_count = count_samples(output_file)
        print(f"  Verification: {final_count} rows in final file")

        if abs(final_count - expected_samples) > 100:
            print(f"  WARNING: Expected ~{expected_samples} but got {final_count}")
    else:
        print(f"  ERROR: No data was written to {output_file}")
        return False

    return True


def validate_and_fix_file_consistency(file_path: str, chunk_size: int = 10_000) -> bool:
    """Rewrite `file_path` in place with cleaned types and the canonical column names."""
    if not os.path.exists(file_path):
        print(f"  File does not exist: {file_path}")
        return False

    try:
        print(f"Validating data consistency in {os.path.basename(file_path)}")
        print(f"  Expected columns: {len(EXPECTED_COLUMNS)}")

        temp_output = file_path + "_temp"
        total_processed = 0
        first_chunk = True

        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
            validated_chunk = _validate_and_clean_chunk(chunk_df, len(EXPECTED_COLUMNS))

            if len(validated_chunk) > 0:
                if list(validated_chunk.columns) != EXPECTED_COLUMNS:
                    if len(validated_chunk.columns) == len(EXPECTED_COLUMNS):
                        validated_chunk.columns = EXPECTED_COLUMNS

                validated_chunk.to_csv(temp_output, mode='a', header=first_chunk,
                                       index=False, float_format='%.17g')
                first_chunk = False
                total_processed += len(validated_chunk)

            del chunk_df, validated_chunk
            gc.collect()

        if total_processed > 0:
            os.replace(temp_output, file_path)
            print(f"  Validated and fixed {total_processed} rows")
            return True

        print("  No output file created during validation")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False

    except Exception as e:
        print(f"  Error during validation: {e}")
        return False
