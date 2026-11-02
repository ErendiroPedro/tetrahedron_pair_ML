import pandas as pd
import torch

from src.utils.geometry import principal_axis_transform, unitary_tetrahedron_transform

_UNITARY_COLUMNS = [f"T1_v{v}_{c}" for v in range(1, 5) for c in "xyz"]


def transform_data(data: pd.DataFrame, transformations) -> pd.DataFrame:
    """Apply the named transformations in order to a chunk of raw rows."""
    if data.empty or not transformations:
        return data

    coordinate_columns = data.columns[:-2]
    numeric = data[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    valid = ~numeric.isna().any(axis=1)
    if not valid.all():
        print(f"    Removing {(~valid).sum()} samples with invalid coordinates")
    data = data[valid].reset_index(drop=True)
    data[coordinate_columns] = numeric[valid].reset_index(drop=True)

    for name in transformations:
        # Column count can change (unitary: 24 -> 12), so re-derive each time.
        data = TRANSFORMATIONS[name](data, data.columns[:-2])
    return data


def _apply_unitary_tetrahedron(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    """Unitary-tetrahedron transform: 24 coordinate columns -> 12 (T1 only)."""
    pair_coordinates = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
    transformed = unitary_tetrahedron_transform(pair_coordinates)[:, :12]
    transformed_frame = pd.DataFrame(transformed.cpu().numpy(), columns=_UNITARY_COLUMNS, index=data.index)
    return pd.concat([transformed_frame, data.drop(columns=coordinate_columns)], axis=1)


def _apply_principal_axis(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    """Principal-axis transform: rewrites the 24 coordinate columns in place."""
    pair_coordinates = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
    data[coordinate_columns] = principal_axis_transform(pair_coordinates).cpu().numpy()
    return data


TRANSFORMATIONS = {
    "unitary_tetrahedron": _apply_unitary_tetrahedron,
    "principal_axis": _apply_principal_axis,
}
