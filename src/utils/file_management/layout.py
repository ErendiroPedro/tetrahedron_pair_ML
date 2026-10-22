"""Where datasets live on disk.

raw/<intersection_type>/*.csv       one or more source CSVs per intersection type
processed/{train,val}/<split>_data.csv
"""

import os
from typing import List

from src.utils.file_management.csv_files import discover_csv_files

INTERSECTION_TYPES = (
    "no_intersection",
    "point_intersection",
    "segment_intersection",
    "polygon_intersection",
    "polyhedron_intersection",
)


def discover_raw_files(raw_data_path: str, intersection_type: str) -> List[str]:
    folder = os.path.join(raw_data_path, intersection_type)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Raw data folder not found: {folder}")
    return discover_csv_files(folder)


def processed_split_path(processed_data_path: str, split: str) -> str:
    return os.path.join(processed_data_path, split, f"{split}_data.csv")
