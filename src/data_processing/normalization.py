import pandas as pd


def normalize_types(df: pd.DataFrame, volume_scale_factor: float = 1.0) -> pd.DataFrame:
    """Coerce coordinates/volume to float64 and label to int32, drop empty rows, zero-fill NaN.

    The volume column is multiplied by `volume_scale_factor` (skipped when it is 1.0).
    """
    if df.empty:
        return df

    df = df.copy()
    try:
        for col in df.columns[:-2]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        if "IntersectionVolume" in df.columns:
            df["IntersectionVolume"] = pd.to_numeric(df["IntersectionVolume"], errors="coerce").astype("float64")
            if volume_scale_factor != 1.0:
                df["IntersectionVolume"] = df["IntersectionVolume"] * volume_scale_factor

        if "HasIntersection" in df.columns:
            df["HasIntersection"] = pd.to_numeric(df["HasIntersection"], errors="coerce").astype("int32")

        df = df[~df.isna().all(axis=1)]
        return df.fillna(0)
    except Exception as e:
        print(f"  Warning: Type conversion error: {e}")
        return pd.DataFrame()
