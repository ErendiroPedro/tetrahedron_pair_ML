import pandas as pd
import torch

import src.utils.geometry as gu

_UNITARY_COLUMNS = [f"T1_v{v}_{c}" for v in range(1, 5) for c in "xyz"]


def transform_data(data: pd.DataFrame, transformations) -> pd.DataFrame:
    """Apply the named transformations in order to a chunk of raw rows."""
    if data.empty or not transformations:
        return data

    coords = data.columns[:-2]
    numeric = data[coords].apply(pd.to_numeric, errors="coerce")
    valid = ~numeric.isna().any(axis=1)
    if not valid.all():
        print(f"    Removing {(~valid).sum()} samples with invalid coordinates")
    data = data[valid].reset_index(drop=True)
    data[coords] = numeric[valid].reset_index(drop=True)

    for name in transformations:
        # Column count can change (unitary: 24 -> 12), so re-derive each time.
        data = TRANSFORMATIONS[name](data, data.columns[:-2])
    return data


def _apply_unitary(data: pd.DataFrame, coords) -> pd.DataFrame:
    """Unitary-tetrahedron transform: 24 coordinate columns -> 12 (T1 only)."""
    tensor = torch.tensor(data[coords].values, dtype=torch.float64)
    transformed = gu.apply_unitary_tetrahedron_transformation_batch(tensor)[:, :12]
    df = pd.DataFrame(transformed.cpu().numpy(), columns=_UNITARY_COLUMNS, index=data.index)
    return pd.concat([df, data.drop(columns=coords)], axis=1)


def _apply_principal(data: pd.DataFrame, coords) -> pd.DataFrame:
    """Principal-axis transform: rewrites the 24 coordinate columns in place."""
    tensor = torch.tensor(data[coords].values, dtype=torch.float64)
    data[coords] = gu.apply_principal_axis_transformation_batch(tensor).cpu().numpy()
    return data


TRANSFORMATIONS = {
    "unitary_tetrahedron": _apply_unitary,
    "principal_axis": _apply_principal,
}
