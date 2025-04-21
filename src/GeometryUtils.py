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
    Sorts only the rows with HasIntersection==1 by IntersectionVolume (ascending)
    while leaving the rows with HasIntersection==0 in their original positions.
    """

    data_sorted = data.copy()
    
    intersection_mask = data_sorted['HasIntersection'] == 1
    intersection_indices = data_sorted.index[intersection_mask]

    sorted_intersections = data_sorted.loc[intersection_indices].sort_values(by='IntersectionVolume', ascending=True)
    
    data_sorted.loc[intersection_indices] = sorted_intersections
    
    return data_sorted

def apply_affine_linear_transformation(tetrahedron_pair_vertices_flat: pd.Series) -> pd.Series:
    """Transform second tetrahedron's vertices relative to the first one"""
    # Convert to tensor only once
    input_tensor = torch.tensor(tetrahedron_pair_vertices_flat.values, dtype=torch.float32).flatten()
    
    # Reshape into two tetrahedrons
    tetra1_vertices = input_tensor[:12].reshape(4, 3)
    tetra2_vertices = input_tensor[12:].reshape(4, 3)
    
    # Get first tetrahedron's basis vectors
    v0 = tetra1_vertices[0]
    edge_vectors = tetra1_vertices[1:] - v0.unsqueeze(0)  # More efficient way to get edges
    
    # Calculate inverse transformation matrix
    try:
        inv_transform = torch.inverse(edge_vectors)
    except RuntimeError:
        raise ValueError("The tetrahedron basis matrix is singular and cannot be inverted")
    
    # Vectorized transformation - no loop needed
    translated = tetra2_vertices - v0.unsqueeze(0)  # Broadcasting for all vertices at once
    transformed_verts = translated @ inv_transform  # Matrix multiplication for all vertices
    
    # Convert back to a Pandas Series - only one conversion
    column_names = [f'T1_v{i//3 + 1}_{"xyz"[i % 3]}' for i in range(12)]
    return pd.Series(transformed_verts.flatten().detach().numpy(), index=column_names)

def apply_rigid_transformation(tetrahedron_pair_vertices_flat: pd.Series) -> pd.Series:
    """
    Apply a rigid transformation to both tetrahedrons to bring them into a canonical pose,
    preserving their relative configuration (and intersection status).
    The canonical frame is defined by tetrahedron 1's (t1) centroid and its principal axes.
    """
    # Convert input series to a tensor and reshape into two tetrahedrons (4 vertices each, 3 coordinates)
    input_tensor = torch.tensor(tetrahedron_pair_vertices_flat.values, dtype=torch.float32).flatten()
    tetra1 = input_tensor[:12].reshape(4, 3)
    tetra2 = input_tensor[12:].reshape(4, 3)
    
    # Compute t1's centroid
    centroid1 = tetra1.mean(dim=0)
    tetra1_centered = tetra1 - centroid1

    # Use PCA on tetra1 to obtain principal axes:
    # Compute the covariance matrix of t1's centered vertices
    cov_matrix = tetra1_centered.T @ tetra1_centered
    # Get eigenvalues and eigenvectors (the eigenvectors form the rotation basis)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    # Sort eigenvectors by eigenvalues in descending order (largest variance first)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    R_canonical = eigenvectors[:, sorted_indices]  # each column is one principal axis

    # Define the transformation: subtract t1's centroid and rotate using R_canonical^T
    transform = lambda x: (R_canonical.T @ (x - centroid1).T).T

    # Apply the same transformation to both tetrahedrons
    tetra1_transformed = transform(tetra1)
    tetra2_transformed = transform(tetra2)

    result_t1 = tetra1_transformed.flatten().detach().numpy()
    result_t2 = tetra2_transformed.flatten().detach().numpy()
    result = np.concatenate((result_t1, result_t2))
    
    index = [f'T{i+1}_v{j+1}_{axis}' 
             for i in range(2) 
             for j in range(4) 
             for axis in "xyz"]
    
    return pd.Series(result, index=index)

def calculate_tetrahedron_volume(tetrahedron: np.ndarray) -> float:
    """Calculate tetrahedron volume using signed tetrahedral volume formula."""
    v0, v1, v2, v3 = tetrahedron.reshape(4, 3)
    return np.abs(np.linalg.det(np.stack([v1-v0, v2-v0, v3-v0]))) / 6