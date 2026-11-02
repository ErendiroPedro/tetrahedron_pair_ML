"""Dataset layout on disk: raw/<intersection_type>/*.csv and processed/{train,val}/<split>_data.csv."""

import os
from typing import List

INTERSECTION_TYPES = (
    "no_intersection",
    "point_intersection",
    "segment_intersection",
    "polygon_intersection",
    "polyhedron_intersection",
)

EXPECTED_COLUMNS = [f"T{t}_v{v}_{axis}" for t in (1, 2) for v in (1, 2, 3, 4) for axis in "xyz"] + [
    "IntersectionVolume",
    "HasIntersection",
]


def discover_csv_files(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".csv")]


def processed_split_path(processed_data_path: str, split: str) -> str:
    return os.path.join(processed_data_path, split, f"{split}_data.csv")


def count_samples(file_path: str) -> int:
    """Number of data rows (header excluded); also counts a last row that has no trailing newline."""
    newlines, last_byte = 0, b"\n"
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return max(newlines + (last_byte != b"\n") - 1, 0)
