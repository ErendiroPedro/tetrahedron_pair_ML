import numpy as np
import pandas as pd
import torch

import src.utils.geometry as gu

SWAP = "swap"
PERMUTE = "permute"
_OPERATIONS = {
    SWAP: gu.swap_tetrahedrons,
    PERMUTE: gu.permute_points_within_tetrahedrons,
}


def augment_coordinates(coords: np.ndarray, kind: str) -> np.ndarray:
    """Apply a geometric augmentation to one flat 24-value tetrahedron-pair vector."""
    batch = torch.tensor(coords, dtype=torch.float64).unsqueeze(0)
    return _OPERATIONS[kind](batch).squeeze(0).numpy()


def augment_record(record: dict, kind: str) -> dict:
    """Same as augment_coordinates, for a row dict with T1_*/T2_* coordinate keys."""
    result = dict(record)
    coordinate_keys = [k for k in record if k.startswith(("T1_", "T2_"))]
    if len(coordinate_keys) != 24:
        return result

    try:
        coords = np.array([record[k] for k in coordinate_keys])
        for key, value in zip(coordinate_keys, augment_coordinates(coords, kind)):
            result[key] = value
    except Exception as e:
        print(f"            Warning: {kind} augmentation failed: {e}, keeping original")
    return result


def augment_data(data: pd.DataFrame, point_wise_pct: float, tetrahedron_wise_pct: float) -> pd.DataFrame:
    """Append augmented copies of randomly chosen rows (percentages of the input size)."""
    if data.empty:
        return data

    try:
        new_rows = []
        for kind, pct in ((PERMUTE, point_wise_pct), (SWAP, tetrahedron_wise_pct)):
            n_samples = int(len(data) * (pct / 100.0))
            if n_samples > 0:
                new_rows.extend(_augmented_rows(data, n_samples, kind))

        if not new_rows:
            return data
        return pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    except Exception as e:
        print(f"Warning: Data augmentation failed: {e}")
        return data


def _augmented_rows(data: pd.DataFrame, n_samples: int, kind: str) -> list:
    indices = np.random.choice(len(data), n_samples, replace=n_samples >= len(data))
    coordinate_columns = data.columns[:-2]

    rows = []
    for idx in indices:
        row = data.iloc[idx].copy()
        try:
            row[coordinate_columns] = augment_coordinates(row[coordinate_columns].values, kind)
        except Exception as e:
            print(f"      Warning: {kind} augmentation failed for sample {idx}: {e}")
        rows.append(row.to_dict())
    return rows
