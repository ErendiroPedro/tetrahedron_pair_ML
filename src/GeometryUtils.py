import torch
import numpy as np
import pandas as pd
import pymorton


def swap_tetrahedrons(X):
    """
    Swap the order of tetrahedrons in the input tensor.
    Args:
        X (torch.Tensor): Input tensor of shape (batch_size, 24).
    Returns:
        torch.Tensor: Tensor with swapped tetrahedrons.
    """
    return torch.cat([X[:, 12:], X[:, :12]], dim=1)

def permute_points_within_tetrahedrons(X: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute points within each tetrahedron
    """
    batch_size = X.size(0)
    device = X.device

    # Split into two tetrahedrons (12 features each)
    tetra1 = X[:, :12].view(batch_size, 4, 3)  # (B, 4, 3)
    tetra2 = X[:, 12:].view(batch_size, 4, 3)  # (B, 4, 3)

    # Generate permutations for both tetrahedrons
    perm1 = torch.stack([torch.randperm(4) for _ in range(batch_size)]).to(device)
    perm2 = torch.stack([torch.randperm(4) for _ in range(batch_size)]).to(device)

    # Apply permutations using gather
    permuted_tetra1 = torch.gather(tetra1, 1, perm1.unsqueeze(-1).expand(-1, -1, 3))
    permuted_tetra2 = torch.gather(tetra2, 1, perm2.unsqueeze(-1).expand(-1, -1, 3))

    # Reconstruct features
    return torch.cat([
        permuted_tetra1.view(batch_size, 12),
        permuted_tetra2.view(batch_size, 12)
    ], dim=1)

def volume_reordering(data: pd.DataFrame, larger=True) -> pd.DataFrame:
    """
    Reorder the tetrahedrons in the dataset based on their volumes.
    """
    features = data.iloc[:, :-2]
    labels = data.iloc[:, -2:]

    def reorder_row(row):
        X = row.values.astype(np.float32)
        
        # Split into two tetrahedrons
        T1, T2 = X[:12], X[12:]
        
        # Calculate volumes
        vol1 = calculate_tetrahedron_volume(T1)
        vol2 = calculate_tetrahedron_volume(T2)
        
        # Reorder if necessary
        if (vol2 > vol1 and larger) or (vol1 > vol2 and not larger):
            X = np.concatenate([T2, T1])
        
        return X

    # Apply reordering to each row
    features_processed = features.apply(reorder_row, axis=1, result_type='expand')
    
    # Recombine features and labels
    data_processed = pd.concat([features_processed, labels.reset_index(drop=True)], axis=1)
    
    return data_processed

def sort_by_x_coordinate(data: pd.DataFrame, column_name: str = "T1_v1_x") -> pd.DataFrame:
    """
    Sorts the given DataFrame by the specified column.
    
    Args:
        tetrahedron_pair_vertices_flat (pd.DataFrame): The input dataset containing the column to sort by.
        column_name (str): The name of the column to sort the DataFrame by.
    
    Returns:
        pd.DataFrame: Sorted DataFrame by the specified column in ascending order.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    return data.sort_values(by=column_name, ascending=True).reset_index(drop=True)

def sort_by_morton_code(data: pd.DataFrame) -> pd.DataFrame:

    scale_factor = 1e18 # to convert high precision coordinates into integers.

    morton_codes = []

    for idx, row in data.iterrows():

        # Get tetrahedron vertices
        coords = row.iloc[2:].to_numpy(dtype=float)
        tetra1 = coords[:12].reshape((4, 3))
        tetra2 = coords[12:].reshape((4, 3))

        # Compute centroids of each tetrahedron
        centroid1 = tetra1.mean(axis=0)
        centroid2 = tetra2.mean(axis=0)

        # Calculate the average (combined) centroid
        avg_centroid = (centroid1 + centroid2) / 2.0

        # Scale and convert each coordinate to an integer.
        x, y, z = [int(round(coord * scale_factor)) for coord in avg_centroid]

        # Compute the Morton code (Z-order)
        code = pymorton.interleave(x, y, z)
        morton_codes.append(code)

    # Sort by the Morton code and drop the temporary column before returning.
    data['_morton_code'] = morton_codes
    sorted_df = data.sort_values(by='_morton_code').drop(columns=['_morton_code']).reset_index(drop=True)

    return sorted_df

def sort_by_x_coordinate_alt(data: pd.DataFrame) -> pd.DataFrame:

    t1_columns = [col for col in data.columns if col.startswith("T1_")]
    t2_columns = [col for col in data.columns if col.startswith("T2_")]

    metadata_columns = [col for col in data.columns if col not in t1_columns + t2_columns]
    
    def sort_tetrahedron_vertices(row, tetrahedron_columns):
        vertices = row[tetrahedron_columns].values.reshape(4, 3)  # 4 vertices x 3 coordinates
        sorted_vertices = vertices[vertices[:, 0].argsort()] # Sort by x-coordinate (first column)
        return pd.Series(sorted_vertices.flatten(), index=tetrahedron_columns)
    
    t1_sorted = data.apply(sort_tetrahedron_vertices, axis=1, tetrahedron_columns=t1_columns)
    t2_sorted = data.apply(sort_tetrahedron_vertices, axis=1, tetrahedron_columns=t2_columns)

    sorted_data = pd.concat([t1_sorted, t2_sorted, data[metadata_columns]], axis=1)
    
    return sorted_data

def sort_by_morton_code_alt(data: pd.DataFrame, scale_factor: float = 1e18) -> pd.DataFrame:
    t1_columns = [col for col in data.columns if col.startswith("T1_")]
    t2_columns = [col for col in data.columns if col.startswith("T2_")]
    metadata_columns = [col for col in data.columns if col not in t1_columns + t2_columns]

    def sort_tetrahedron(row, tetra_columns):
        vertices = row[tetra_columns].values.reshape(4, 3)
        morton_codes = []
        for vertex in vertices:
            scaled_coords = [int(round(coord * scale_factor)) for coord in vertex]
            code = pymorton.interleave(*scaled_coords)
            morton_codes.append(code)
        sorted_indices = np.argsort(morton_codes)
        sorted_vertices = vertices[sorted_indices].flatten()
        return pd.Series(sorted_vertices, index=tetra_columns)

    t1_sorted = data.apply(sort_tetrahedron, axis=1, tetra_columns=t1_columns)
    t2_sorted = data.apply(sort_tetrahedron, axis=1, tetra_columns=t2_columns)

    return pd.concat([t1_sorted, t2_sorted, data[metadata_columns]], axis=1)

def sort_by_intersection_volume_whole_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the dataset by IntersectionVolume while keeping HasIntersection=0 entries in their original order.
    HasIntersection=1 entries are sorted by IntersectionVolume in ascending order.
    """
    # Separate the rows where HasIntersection is 0 and 1
    no_intersection = data[data['HasIntersection'] == 0]
    has_intersection = data[data['HasIntersection'] == 1].sort_values(by='IntersectionVolume', ascending=True)

    # Concatenate back while preserving original order of no_intersection
    return pd.concat([no_intersection, has_intersection]).reset_index(drop=True)


def apply_affine_linear_transformation(tetrahedron_pair_vertices_flat: pd.Series) -> pd.Series:
    """Transform second tetrahedron's vertices relative to the first one"""
    input_tensor = torch.tensor(tetrahedron_pair_vertices_flat.values, dtype=torch.float32).flatten()
    
    # Reshape into two tetrahedrons
    tetra1_vertices = input_tensor[:12].reshape(4, 3)
    tetra2_vertices = input_tensor[12:].reshape(4, 3)
    
    # Get first tetrahedron's basis vectors
    v0, v1, v2, v3 = tetra1_vertices[0], tetra1_vertices[1], tetra1_vertices[2], tetra1_vertices[3]
    edge_vectors = torch.stack([v1 - v0, v2 - v0, v3 - v0])
    
    # Calculate inverse transformation matrix
    try:
        inv_transform = torch.inverse(edge_vectors)
    except RuntimeError:
        raise ValueError("The tetrahedron basis matrix is singular and cannot be inverted")
    
    # Transform second tetrahedron's vertices
    transformed = []
    for vertex in tetra2_vertices:
        translated = vertex - v0
        transformed_vert = inv_transform.T @ translated.unsqueeze(-1)
        transformed.append(transformed_vert.squeeze())
    
    # Convert back to a Pandas Series
    result = torch.stack(transformed).flatten().detach().numpy()
    return pd.Series(result, index=[f'v{i//3 + 1}_{"xyz"[i % 3]}' for i in range(12)])

def calculate_tetrahedron_volume(tetrahedron: np.ndarray) -> float:
    """Calculate tetrahedron volume using signed tetrahedral volume formula."""
    v0, v1, v2, v3 = tetrahedron.reshape(4, 3)
    return np.abs(np.linalg.det(np.stack([v1-v0, v2-v0, v3-v0]))) / 6