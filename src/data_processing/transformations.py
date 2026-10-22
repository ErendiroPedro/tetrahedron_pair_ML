import numpy as np
import pandas as pd
import torch

import src.utils.geometry as gu

_UNITARY_COLUMNS = [f"T1_v{v}_{c}" for v in range(1, 5) for c in "xyz"]


def transform_data(data: pd.DataFrame, transformations) -> pd.DataFrame:
    """Apply the named transformations in order to a chunk of raw rows."""
    if data.empty or not transformations:
        return data

    try:
        coordinate_columns = data.columns[:-2]
        numeric = data[coordinate_columns].apply(pd.to_numeric, errors="coerce")
        invalid = numeric.isna().any(axis=1)
        if invalid.any():
            print(f"    Removing {invalid.sum()} samples with invalid coordinates")
            data = data[~invalid].reset_index(drop=True)
            numeric = numeric[~invalid].reset_index(drop=True)
        data[coordinate_columns] = numeric

        for name in transformations:
            # Column count can change (unitary: 24 -> 12), so re-derive each time.
            data = TRANSFORMATIONS[name](data, data.columns[:-2])
        return data

    except Exception as e:
        print(f"Warning: Transformation failed: {e}")
        return data


def _apply_unitary(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    """Unitary-tetrahedron transform: 24 coordinate columns -> 12 (T1 only)."""
    try:
        tensor = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
        transformed = gu.apply_unitary_tetrahedron_transformation_batch(tensor)

        if transformed.shape[1] == 24:
            transformed = transformed[:, :12]
        elif transformed.shape[1] != 12:
            raise ValueError(f"Unexpected output dimension: {transformed.shape[1]}")

        transformed_df = pd.DataFrame(transformed.cpu().numpy(), columns=_UNITARY_COLUMNS, index=data.index)
        return pd.concat([transformed_df, data.drop(columns=coordinate_columns)], axis=1)

    except Exception as e:
        print(f"    Vectorized failed ({e}), using fallback")
        return _apply_unitary_fallback(data, coordinate_columns)


def _apply_unitary_fallback(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    failed_rows = 0
    transformed_rows = []

    for idx in data.index:
        try:
            row = data.loc[idx, coordinate_columns]
            transformed_row = gu.apply_unitary_tetrahedron_transformation(row)
            values = transformed_row.values if hasattr(transformed_row, "values") else transformed_row

            if len(values) in (12, 24):
                transformed_rows.append(values[:12])
            else:
                transformed_rows.append(np.zeros(12))
                failed_rows += 1
        except Exception as e:
            print(f"    Warning: Row {idx} transformation failed: {e}")
            failed_rows += 1
            transformed_rows.append(np.zeros(12))

    transformed_df = pd.DataFrame(transformed_rows, columns=_UNITARY_COLUMNS, index=data.index)
    if failed_rows > 0:
        print(f"    {failed_rows} rows failed transformation (kept as zeros)")
    return pd.concat([transformed_df, data.drop(columns=coordinate_columns)], axis=1)


def _apply_principal(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    """Principal-axis transform: rewrites the 24 coordinate columns in place."""
    try:
        tensor = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
        data[coordinate_columns] = gu.apply_principal_axis_transformation_batch(tensor).cpu().numpy()
        return data
    except Exception as e:
        print(f"    Vectorized failed ({e}), using fallback")
        return _apply_principal_fallback(data, coordinate_columns)


def _apply_principal_fallback(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
    failed_rows = 0
    for idx in data.index:
        try:
            row_tensor = torch.tensor(data.loc[idx, coordinate_columns].values, dtype=torch.float64)
            data.loc[idx, coordinate_columns] = gu.apply_principal_axis_transformation(row_tensor).cpu().numpy().flatten()
        except Exception:
            failed_rows += 1

    if failed_rows > 0:
        print(f"    {failed_rows} rows failed transformation (kept original)")
    return data


TRANSFORMATIONS = {
    "unitary_tetrahedron": _apply_unitary,
    "principal_axis": _apply_principal,
}
