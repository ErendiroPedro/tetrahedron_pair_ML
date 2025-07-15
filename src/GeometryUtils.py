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
    Randomly permute points within each tetrahedron.
    Handles input tensor representing one or two tetrahedrons per item.
    
    Args:
        X: Tensor of shape (batch_size, 12) for single tetrahedrons,
        or (batch_size, 24) for pairs of tetrahedrons.
        Each tetrahedron has 4 vertices of 3 coordinates (x, y, z).
        
    Returns:
        Tensor of the same shape as X, with vertices permuted within each tetrahedron.
    """
    batch_size = X.size(0)
    if batch_size == 0:
        return X

    num_features = X.size(1)
    device = X.device

    if num_features == 24:  # Two tetrahedrons
        # Split into two tetrahedrons
        tetra1 = X[:, :12].reshape(batch_size, 4, 3)  # First tetrahedron
        tetra2 = X[:, 12:].reshape(batch_size, 4, 3)  # Second tetrahedron
        
        # Permute vertices within each tetrahedron
        for b in range(batch_size):
            perm1 = torch.randperm(4)
            perm2 = torch.randperm(4)
            tetra1[b] = tetra1[b][perm1]
            tetra2[b] = tetra2[b][perm2]
        
        # Recombine
        result = torch.cat([tetra1.reshape(batch_size, 12), tetra2.reshape(batch_size, 12)], dim=1)
        return result

    elif num_features == 12:  # Single tetrahedron
        tetra = X.reshape(batch_size, 4, 3)
        
        # Permute vertices within the tetrahedron
        for b in range(batch_size):
            perm = torch.randperm(4)
            tetra[b] = tetra[b][perm]
        
        return tetra.reshape(batch_size, 12)
        
    else:
        raise ValueError(f"Unsupported tensor shape: {X.shape}")

def volume_reordering(data: pd.DataFrame, larger: bool = True) -> pd.DataFrame:
    """
    Reorder tetrahedra so that the larger volume tetrahedron comes first (or second if larger=False).
    Uses vectorized operations for better performance.
    
    Args:
        data: DataFrame with tetrahedron pair coordinates
        larger: If True, larger tetrahedron becomes T1. If False, smaller becomes T1.
    
    Returns:
        DataFrame with potentially swapped tetrahedra
    """
    if data.empty:
        return data
    
    print(f"  Applying volume-based reordering ({'larger' if larger else 'smaller'} tetrahedron first)...")
    
    # Separate features and labels
    features = data.iloc[:, :-2].copy()
    labels = data.iloc[:, -2:].copy()
    
    # Convert to tensor for vectorized operations
    try:
        # Use vectorized volume reordering
        features_processed = _volume_reordering_vectorized(features, larger)
        
        # Recombine with labels - ensure indices align
        features_processed.index = data.index
        labels.index = data.index
        data_processed = pd.concat([features_processed, labels], axis=1)
        
        print(f"    Volume reordering complete (vectorized)")
        return data_processed
        
    except Exception as e:
        print(f"    Vectorized reordering failed ({e}), using fallback")
        return _volume_reordering_fallback(data, larger)

def _volume_reordering_vectorized(features: pd.DataFrame, larger: bool = True) -> pd.DataFrame:
    """
    Vectorized volume-based reordering using torch operations.
    
    Args:
        features: DataFrame with coordinate columns only
        larger: If True, larger tetrahedron becomes T1
        
    Returns:
        DataFrame with reordered coordinates
    """
    # Convert to tensor
    data_tensor = torch.tensor(features.values, dtype=torch.float64)
    batch_size = data_tensor.shape[0]
    
    if data_tensor.shape[1] != 24:
        raise ValueError(f"Expected 24 coordinate columns, got {data_tensor.shape[1]}")
    
    # Reshape into tetrahedra: (batch_size, 2, 4, 3)
    tetrahedra = data_tensor.reshape(batch_size, 2, 4, 3)
    tetra1_batch = tetrahedra[:, 0, :, :]  # (batch_size, 4, 3)
    tetra2_batch = tetrahedra[:, 1, :, :]  # (batch_size, 4, 3)
    
    # Calculate volumes for all tetrahedra using vectorized operations
    volumes1 = _calculate_tetrahedron_volumes_vectorized_torch(tetra1_batch)  # (batch_size,)
    volumes2 = _calculate_tetrahedron_volumes_vectorized_torch(tetra2_batch)  # (batch_size,)
    
    # Determine which samples need swapping
    if larger:
        # Larger tetrahedron should be T1: swap if vol2 > vol1
        swap_mask = volumes2 > volumes1
    else:
        # Smaller tetrahedron should be T1: swap if vol1 > vol2
        swap_mask = volumes1 > volumes2
    
    # Create result tensor - start with original
    result_tetrahedra = tetrahedra.clone()
    
    # Swap tetrahedra for samples that need it
    if swap_mask.any():
        # More explicit swapping to avoid indexing issues
        swap_indices = torch.where(swap_mask)[0]
        for idx in swap_indices:
            # Swap T1 and T2 for this sample
            temp = result_tetrahedra[idx, 0, :, :].clone()
            result_tetrahedra[idx, 0, :, :] = result_tetrahedra[idx, 1, :, :]
            result_tetrahedra[idx, 1, :, :] = temp
    
    # Reshape back to original format: (batch_size, 24)
    result_flat = result_tetrahedra.reshape(batch_size, 24)
    
    # Convert back to DataFrame with original column names and index
    result_df = pd.DataFrame(
        result_flat.cpu().numpy(),
        columns=features.columns,
        index=features.index
    )
    
    swapped_count = swap_mask.sum().item()
    print(f"    Volume reordering complete: {swapped_count}/{batch_size} samples swapped")
    
    return result_df

def _calculate_tetrahedron_volumes_vectorized_torch(tetrahedra_batch):
    """
    Calculate volumes for a batch of tetrahedra using PyTorch operations.
    
    Args:
        tetrahedra_batch: torch.Tensor of shape (batch_size, 4, 3)
        
    Returns:
        torch.Tensor of shape (batch_size,) containing volumes
    """
    # Use first vertex as reference point for all tetrahedra
    v0_batch = tetrahedra_batch[:, 0, :]  # (batch_size, 3)
    
    # Calculate edge vectors from v0 to other vertices
    v1_batch = tetrahedra_batch[:, 1, :] - v0_batch  # (batch_size, 3)
    v2_batch = tetrahedra_batch[:, 2, :] - v0_batch  # (batch_size, 3)
    v3_batch = tetrahedra_batch[:, 3, :] - v0_batch  # (batch_size, 3)
    
    # Stack edge vectors: (batch_size, 3, 3)
    edge_vectors_batch = torch.stack([v1_batch, v2_batch, v3_batch], dim=2)
    
    # Calculate determinants for all samples: Volume = |det(edge_vectors)| / 6
    determinants_batch = torch.det(edge_vectors_batch)  # (batch_size,)
    volumes_batch = torch.abs(determinants_batch) / 6.0
    
    # Check for invalid values and crash if found
    if torch.isnan(volumes_batch).any():
        nan_indices = torch.where(torch.isnan(volumes_batch))[0]
        raise ValueError(f"NaN volumes detected at batch indices: {nan_indices.tolist()}")
    
    if torch.isinf(volumes_batch).any():
        inf_indices = torch.where(torch.isinf(volumes_batch))[0]
        raise ValueError(f"Infinite volumes detected at batch indices: {inf_indices.tolist()}")
    
    if (volumes_batch < 0).any():
        neg_indices = torch.where(volumes_batch < 0)[0]
        raise ValueError(f"Negative volumes detected at batch indices: {neg_indices.tolist()}")
    
    return volumes_batch

def _volume_reordering_fallback(data: pd.DataFrame, larger: bool = True) -> pd.DataFrame:
    """
    Fallback row-by-row volume reordering.
    
    Args:
        data: DataFrame with tetrahedron pair coordinates
        larger: If True, larger tetrahedron becomes T1
        
    Returns:
        DataFrame with reordered coordinates
    """
    features = data.iloc[:, :-2].copy()
    labels = data.iloc[:, -2:].copy()
    
    def reorder_row(row):
        try:
            X = row.values.astype(np.float32)
            
            # Split into two tetrahedrons
            T1, T2 = X[:12], X[12:]
            
            # Calculate volumes
            vol1 = calculate_tetrahedron_volume(T1.reshape(4, 3))
            vol2 = calculate_tetrahedron_volume(T2.reshape(4, 3))
            
            # Reorder if necessary
            if (vol2 > vol1 and larger) or (vol1 > vol2 and not larger):
                X = np.concatenate([T2, T1])
            
            return pd.Series(X, index=row.index)
            
        except Exception as e:
            print(f"    Warning: Row reordering failed: {e}")
            return row  # Return original if failed
    
    # Apply reordering to each row
    features_processed = features.apply(reorder_row, axis=1)
    
    # Recombine features and labels
    data_processed = pd.concat([features_processed, labels], axis=1)
    
    print(f"    Fallback volume reordering complete")
    return data_processed

def calculate_tetrahedron_volume(vertices):
    """
    Calculate volume of a single tetrahedron.
    
    Args:
        vertices: numpy array of shape (4, 3) with tetrahedron vertices
        
    Returns:
        float: Volume of the tetrahedron
    """
    # Use first vertex as reference
    v0 = vertices[0]
    v1, v2, v3 = vertices[1] - v0, vertices[2] - v0, vertices[3] - v0
    
    # Volume = |det([v1, v2, v3])| / 6
    det = np.linalg.det(np.column_stack([v1, v2, v3]))
    volume = abs(det) / 6.0
    
    # Check for invalid values and crash if found
    if np.isnan(volume):
        raise ValueError(f"NaN volume calculated from vertices:\n{vertices}")
    
    if np.isinf(volume):
        raise ValueError(f"Infinite volume calculated from vertices:\n{vertices}")
    
    if volume < 0:
        raise ValueError(f"Negative volume calculated: {volume} from vertices:\n{vertices}")
    
    return volume

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

def apply_unitary_tetrahedron_transformation(tetrahedron_pair_vertices_flat) -> pd.Series:
    """
    Transform a tetrahedron pair using unitary tetrahedron transformation.
    Returns only the transformed coordinates of the second tetrahedron (12 features).
    
    Args:
        tetrahedron_pair_vertices_flat: pandas Series with 24 values (flattened coordinates)
        
    Returns:
        pandas Series with 12 values (transformed T2 coordinates only)
    """
    try:
        # Convert to numpy array and reshape
        coords = tetrahedron_pair_vertices_flat.values.reshape(2, 4, 3)
        tetra1 = coords[0]  # First tetrahedron (4x3)
        tetra2 = coords[1]  # Second tetrahedron (4x3)
        
        # Get reference point (first vertex of first tetrahedron)
        v0 = tetra1[0]
        
        # Calculate transformation matrix from first tetrahedron
        edge_vectors = tetra1[1:] - v0  # 3x3 matrix of edge vectors
        
        # Check if matrix is invertible
        if np.linalg.det(edge_vectors) == 0:
            # Use identity transformation for degenerate cases
            transformed_tetra2 = tetra2 - v0
        else:
            # Apply unitary transformation to second tetrahedron
            inv_transform = np.linalg.inv(edge_vectors)
            translated_tetra2 = tetra2 - v0
            transformed_tetra2 = translated_tetra2 @ inv_transform
        
        # Create column names for the 12 transformed coordinates
        column_names = []
        for v_idx in range(4):
            for coord in ['x', 'y', 'z']:
                column_names.append(f'T2_transformed_v{v_idx+1}_{coord}')
        
        # Return as Series with 12 values
        return pd.Series(transformed_tetra2.flatten(), index=column_names)
        
    except Exception as e:
        print(f"Warning: Unitary transformation failed: {e}")
        # Return zeros for failed transformation
        column_names = []
        for v_idx in range(4):
            for coord in ['x', 'y', 'z']:
                column_names.append(f'T2_transformed_v{v_idx+1}_{coord}')
        return pd.Series(np.zeros(12), index=column_names)

def apply_principal_axis_transformation(tetrahedron_pair_vertices_flat) -> pd.Series:
    """
    Apply a rigid transformation to both tetrahedrons to bring them into a canonical pose,
    preserving their relative configuration (and intersection status).
    The canonical frame is defined by tetrahedron 1's (t1) centroid and its principal axes.
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(tetrahedron_pair_vertices_flat, 'values'):
        # It's a pandas Series
        input_data = tetrahedron_pair_vertices_flat.values
    else:
        # It's already a numpy array
        input_data = tetrahedron_pair_vertices_flat
    
    # Convert input to a tensor and reshape into two tetrahedrons (4 vertices each, 3 coordinates)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).flatten()
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

def _calculate_centroid_distance(row) -> float:
    """
    Calculate Euclidean distance between centroids of two tetrahedra.
    
    Args:
        row: DataFrame row containing tetrahedron pair coordinates OR numpy array
        
    Returns:
        Euclidean distance between the two tetrahedron centroids
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(row, 'iloc'):
        # It's a pandas Series/DataFrame row
        tetrahedron_coords = row.iloc[:-2].values.astype(np.float32)
    elif hasattr(row, 'values'):
        # It's a pandas Series
        tetrahedron_coords = row.values[:-2].astype(np.float32)
    else:
        # It's already a numpy array
        tetrahedron_coords = row[:-2].astype(np.float32)
    
    # Split into two tetrahedra (12 coordinates each)
    tetra1_coords = tetrahedron_coords[:12].reshape(4, 3)
    tetra2_coords = tetrahedron_coords[12:].reshape(4, 3)
    
    # Calculate centroids
    centroid1 = tetra1_coords.mean(axis=0)
    centroid2 = tetra2_coords.mean(axis=0)
    
    # Calculate Euclidean distance
    centroid_distance = np.linalg.norm(centroid2 - centroid1)
    
    return centroid_distance

def validate_tetrahedron_volumes(coordinate_series, min_volume_threshold):
    """
    Validate that both tetrahedra in a sample meet minimum volume requirements.
    
    Args:
        coordinate_series: Pandas Series or numpy array containing 24 coordinates (2 tetrahedra)
        min_volume_threshold: Minimum acceptable volume for each tetrahedron
        
    Returns:
        bool: True if both tetrahedra meet volume requirements, False otherwise
    """
    try:
        # Handle both pandas Series and numpy arrays
        if hasattr(coordinate_series, 'values'):
            coordinates_array = coordinate_series.values.astype(np.float32)
        else:
            coordinates_array = coordinate_series.astype(np.float32)
        
        # Split coordinates into two tetrahedra (4 vertices × 3 coordinates each)
        tetrahedron1_coords = coordinates_array[:12].reshape(4, 3)
        tetrahedron2_coords = coordinates_array[12:24].reshape(4, 3)
        
        # Calculate volumes for both tetrahedra
        volume1 = calculate_tetrahedron_volume(tetrahedron1_coords)
        volume2 = calculate_tetrahedron_volume(tetrahedron2_coords)
        
        # Check if both volumes meet the minimum threshold
        return volume1 >= min_volume_threshold and volume2 >= min_volume_threshold
        
    except Exception as e:
        # Exclude problematic samples for safety
        print(f"Warning: Volume validation failed for sample: {e}")
        return False
    
def calculate_tetrahedron_volume(tetrahedron: np.ndarray) -> float:
    """Calculate tetrahedron volume using signed tetrahedral volume formula."""
    v0, v1, v2, v3 = tetrahedron.reshape(4, 3)
    return np.abs(np.linalg.det(np.stack([v1-v0, v2-v0, v3-v0]))) / 6

def sort_by_difficulty(data: pd.DataFrame, easier_first: bool = True) -> pd.DataFrame:
    """
    Sort dataset by difficulty level while preserving random order of intersection types.
    Within each intersection type, samples are sorted by difficulty.
    
    Args:
        data: DataFrame with tetrahedron pairs and labels
        easier_first: If True, easier samples first; if False, harder samples first
        
    Returns:
        DataFrame sorted by difficulty within intersection type groups
    """
    data_with_difficulty = data.copy()
    
    # Calculate difficulty scores for all samples
    difficulty_scores = []
    
    for idx, row in data.iterrows():
        has_intersection = row['HasIntersection']
        
        if has_intersection == 0:
            # Non-intersecting: difficulty based on centroid distance (closer = harder)
            centroid_distance = _calculate_centroid_distance(row)
            difficulty_scores.append(centroid_distance)
        else:
            # Intersecting: difficulty based on intersection volume (smaller = harder)
            intersection_volume = row['IntersectionVolume']
            difficulty_scores.append(intersection_volume)
    
    data_with_difficulty['_difficulty_score'] = difficulty_scores
    
    # Sort by difficulty score within intersection type
    # For non-intersecting: bigger distance = easier (descending for easier_first)
    # For intersecting: bigger volume = easier (descending for easier_first)
    ascending_order = not easier_first
    
    # Separate and sort each intersection type independently
    intersecting_mask = data_with_difficulty['HasIntersection'] == 1
    
    # Sort non-intersecting samples by difficulty
    non_intersecting_data = data_with_difficulty[~intersecting_mask].sort_values(
        by='_difficulty_score', ascending=ascending_order
    ).reset_index(drop=True)
    
    # Sort intersecting samples by difficulty  
    intersecting_data = data_with_difficulty[intersecting_mask].sort_values(
        by='_difficulty_score', ascending=ascending_order
    ).reset_index(drop=True)
    
    # Create mapping of original positions to sorted samples
    non_intersecting_iter = iter(range(len(non_intersecting_data)))
    intersecting_iter = iter(range(len(intersecting_data)))
    
    # Rebuild dataset maintaining original intersection type order
    result_rows = []
    
    for original_idx, original_row in data.iterrows():
        if original_row['HasIntersection'] == 0:
            # Use next non-intersecting sample from sorted list
            try:
                sorted_idx = next(non_intersecting_iter)
                sorted_row = non_intersecting_data.iloc[sorted_idx].drop('_difficulty_score')
                result_rows.append(sorted_row)
            except StopIteration:
                break
        else:
            # Use next intersecting sample from sorted list
            try:
                sorted_idx = next(intersecting_iter)
                sorted_row = intersecting_data.iloc[sorted_idx].drop('_difficulty_score')
                result_rows.append(sorted_row)
            except StopIteration:
                break
    
    # Reconstruct DataFrame
    result_data = pd.DataFrame(result_rows).reset_index(drop=True)
    return result_data

def sort_by_difficulty_stepwise(data: pd.DataFrame, easier_first: bool = True, num_steps: int = 10) -> pd.DataFrame:
    """
    Create discrete difficulty steps instead of smooth progression - optimized for large datasets
    """
    data_size = len(data)
    
    # Vectorized difficulty score calculation
    has_intersection = data['HasIntersection'].values
    intersection_volume = data['IntersectionVolume'].values
    
    # Calculate centroid distances vectorized for non-intersecting samples
    non_intersecting_mask = has_intersection == 0
    difficulty_scores = np.zeros(data_size)
    
    if non_intersecting_mask.any():
        # Vectorized centroid distance calculation
        coords = data.iloc[non_intersecting_mask, :-2].values.astype(np.float32)
        tetra1_coords = coords[:, :12].reshape(-1, 4, 3)
        tetra2_coords = coords[:, 12:].reshape(-1, 4, 3)
        
        centroid1 = tetra1_coords.mean(axis=1)
        centroid2 = tetra2_coords.mean(axis=1)
        centroid_distances = np.linalg.norm(centroid2 - centroid1, axis=1)
        
        difficulty_scores[non_intersecting_mask] = centroid_distances
    
    # For intersecting samples, use intersection volume directly
    intersecting_mask = has_intersection == 1
    if intersecting_mask.any():
        difficulty_scores[intersecting_mask] = intersection_volume[intersecting_mask]
    
    # Sort indices instead of data
    ascending_order = not easier_first
    
    non_int_indices = np.where(non_intersecting_mask)[0]
    int_indices = np.where(intersecting_mask)[0]
    
    # Sort by difficulty scores
    if len(non_int_indices) > 0:
        non_int_sorted_idx = non_int_indices[np.argsort(difficulty_scores[non_int_indices])]
        if not ascending_order:
            non_int_sorted_idx = non_int_sorted_idx[::-1]
    else:
        non_int_sorted_idx = np.array([], dtype=int)
    
    if len(int_indices) > 0:
        int_sorted_idx = int_indices[np.argsort(difficulty_scores[int_indices])]
        if not ascending_order:
            int_sorted_idx = int_sorted_idx[::-1]
    else:
        int_sorted_idx = np.array([], dtype=int)
    
    # Apply stepwise logic using indices
    def create_stepwise_indices(sorted_indices, num_steps, easier_first):
        if len(sorted_indices) == 0:
            return np.array([], dtype=int)
            
        step_size = len(sorted_indices) // num_steps
        if step_size == 0:
            step_size = 1
            num_steps = len(sorted_indices)
        
        result_indices = []
        for step in range(num_steps):
            start_idx = step * step_size
            end_idx = (step + 1) * step_size if step < num_steps - 1 else len(sorted_indices)
            
            if easier_first:
                # Alternate between easy and hard samples
                if step % 2 == 0:
                    # Easy samples (from beginning of sorted array)
                    result_indices.extend(sorted_indices[start_idx:end_idx])
                else:
                    # Hard samples (from end of sorted array)
                    hard_start = len(sorted_indices) - end_idx
                    hard_end = len(sorted_indices) - start_idx
                    result_indices.extend(sorted_indices[hard_start:hard_end])
            else:
                # harder_first: reverse the pattern
                if step % 2 == 0:
                    # Hard samples first
                    hard_start = len(sorted_indices) - end_idx
                    hard_end = len(sorted_indices) - start_idx
                    result_indices.extend(sorted_indices[hard_start:hard_end])
                else:
                    # Easy samples
                    result_indices.extend(sorted_indices[start_idx:end_idx])
        
        return np.array(result_indices)
    
    # Create stepwise indices for both groups
    stepwise_non_int_idx = create_stepwise_indices(non_int_sorted_idx, num_steps, easier_first)
    stepwise_int_idx = create_stepwise_indices(int_sorted_idx, num_steps, easier_first)
    
    # Rebuild using index mapping to maintain original intersection type order
    stepwise_non_int_iter = iter(stepwise_non_int_idx)
    stepwise_int_iter = iter(stepwise_int_idx)
    
    result_indices = []
    for i in range(data_size):
        if has_intersection[i] == 0:
            try:
                result_indices.append(next(stepwise_non_int_iter))
            except StopIteration:
                break
        else:
            try:
                result_indices.append(next(stepwise_int_iter))
            except StopIteration:
                break
    
    # Single iloc operation instead of row-by-row reconstruction
    return data.iloc[result_indices].reset_index(drop=True)

def sort_by_difficulty_mixed(data: pd.DataFrame, easier_first: bool = True, mix_ratio: float = 0.5) -> pd.DataFrame:
    """
    Mix easy and hard samples instead of pure curriculum - optimized for large datasets
    
    Args:
        data: DataFrame with tetrahedron pairs and labels
        easier_first: If True, prioritize easier samples; if False, prioritize harder samples
        mix_ratio: Proportion of hard samples to include (0.0 = all easy, 1.0 = all hard)
        
    Returns:
        DataFrame with mixed difficulty ordering
    """
    data_size = len(data)
    has_intersection = data['HasIntersection'].values
    
    # Step 1: Calculate difficulty scores for all samples
    difficulty_scores = _calculate_difficulty_scores_vectorized(data, has_intersection)
    
    # Step 2: Sort samples by difficulty within each intersection type
    non_int_sorted_idx, int_sorted_idx = _sort_samples_by_difficulty(
        has_intersection, difficulty_scores, easier_first
    )
    
    # Step 3: Mix easy and hard samples according to the specified ratio
    mixed_non_int_idx = _mix_samples_by_difficulty(non_int_sorted_idx, False, mix_ratio)
    mixed_int_idx = _mix_samples_by_difficulty(int_sorted_idx, easier_first, mix_ratio)
    
    # Step 4: Rebuild dataset maintaining original intersection type distribution
    result_indices = _rebuild_original_order(
        has_intersection, mixed_non_int_idx, mixed_int_idx, data_size
    )
    
    return data.iloc[result_indices].reset_index(drop=True)

def _calculate_difficulty_scores_vectorized(data: pd.DataFrame, has_intersection: np.ndarray) -> np.ndarray:
    """Calculate difficulty scores for all samples using vectorized operations."""
    data_size = len(data)
    difficulty_scores = np.zeros(data_size)
    
    # For non-intersecting samples: difficulty = centroid distance (closer = harder)
    non_intersecting_mask = has_intersection == 0
    if non_intersecting_mask.any():
        coords = data.iloc[non_intersecting_mask, :-2].values.astype(np.float32)
        tetra1_coords = coords[:, :12].reshape(-1, 4, 3)
        tetra2_coords = coords[:, 12:].reshape(-1, 4, 3)
        
        centroid1 = tetra1_coords.mean(axis=1)
        centroid2 = tetra2_coords.mean(axis=1)
        centroid_distances = np.linalg.norm(centroid2 - centroid1, axis=1)
        
        difficulty_scores[non_intersecting_mask] = centroid_distances
    
    # For intersecting samples: difficulty = intersection volume (smaller = harder)
    intersecting_mask = has_intersection == 1
    if intersecting_mask.any():
        difficulty_scores[intersecting_mask] = data.loc[intersecting_mask, 'IntersectionVolume'].values
    
    return difficulty_scores

def _sort_samples_by_difficulty(has_intersection: np.ndarray, difficulty_scores: np.ndarray, easier_first: bool) -> tuple:
    """Sort sample indices by difficulty within each intersection type."""
    ascending_order = not easier_first
    
    # Get indices for each intersection type
    non_int_indices = np.where(has_intersection == 0)[0]
    int_indices = np.where(has_intersection == 1)[0]
    
    # Sort non-intersecting samples
    if len(non_int_indices) > 0:
        sort_order = np.argsort(difficulty_scores[non_int_indices])
        non_int_sorted_idx = non_int_indices[sort_order]
        if not ascending_order:
            non_int_sorted_idx = non_int_sorted_idx[::-1]
    else:
        non_int_sorted_idx = np.array([], dtype=int)
    
    # Sort intersecting samples
    if len(int_indices) > 0:
        sort_order = np.argsort(difficulty_scores[int_indices])
        int_sorted_idx = int_indices[sort_order]
        if not ascending_order:
            int_sorted_idx = int_sorted_idx[::-1]
    else:
        int_sorted_idx = np.array([], dtype=int)
    
    return non_int_sorted_idx, int_sorted_idx

def _mix_samples_by_difficulty(sorted_indices: np.ndarray, easier_first: bool, mix_ratio: float) -> np.ndarray:
    """Mix easy and hard samples according to the specified ratio."""
    if len(sorted_indices) == 0:
        return np.array([], dtype=int)
    
    if easier_first:
        easy_portion = 1.0 - mix_ratio
        easy_count = int(len(sorted_indices) * easy_portion)
        hard_count = len(sorted_indices) - easy_count
        
        if hard_count > 0:
            # Take easy samples from beginning, hard samples from end
            mixed_indices = np.concatenate([
                sorted_indices[:easy_count],      # Easy samples
                sorted_indices[-hard_count:]     # Hard samples
            ])
        else:
            mixed_indices = sorted_indices[:easy_count]
    else:
        # harder_first
        hard_portion = mix_ratio
        hard_count = int(len(sorted_indices) * hard_portion)
        easy_count = len(sorted_indices) - hard_count
        
        if hard_count > 0:
            # Take hard samples from end, easy samples from beginning
            mixed_indices = np.concatenate([
                sorted_indices[-hard_count:],    # Hard samples first
                sorted_indices[:easy_count]      # Easy samples
            ])
        else:
            mixed_indices = sorted_indices[:easy_count]
    
    return mixed_indices

def _rebuild_original_order(has_intersection: np.ndarray, mixed_non_int_idx: np.ndarray, 
                           mixed_int_idx: np.ndarray, data_size: int) -> list:
    """Rebuild dataset maintaining the original intersection type distribution."""
    non_int_iter = iter(mixed_non_int_idx)
    int_iter = iter(mixed_int_idx)
    
    result_indices = []
    for i in range(data_size):
        if has_intersection[i] == 0:
            try:
                result_indices.append(next(non_int_iter))
            except StopIteration:
                break
        else:
            try:
                result_indices.append(next(int_iter))
            except StopIteration:
                break
    
    return result_indices

def calculate_tetrahedron_volume(tetrahedron_vertices):
    """
    Calculate tetrahedron volume using signed tetrahedral volume formula.
    
    Args:
        tetrahedron_vertices: numpy array of shape (4, 3) or torch tensor
        
    Returns:
        float: Volume of the tetrahedron
    """
    if hasattr(tetrahedron_vertices, 'numpy'):
        # It's a torch tensor
        vertices = tetrahedron_vertices.numpy()
    else:
        vertices = tetrahedron_vertices
    
    v0, v1, v2, v3 = vertices.reshape(4, 3)
    return np.abs(np.linalg.det(np.stack([v1-v0, v2-v0, v3-v0]))) / 6.0

def validate_tetrahedron_volumes_vectorized(coordinates_array, min_volume_threshold):
    """
    Vectorized tetrahedron volume validation for all samples at once.
    
    Args:
        coordinates_array: numpy array of shape (n_samples, 24)
        min_volume_threshold: minimum acceptable volume
        
    Returns:
        numpy array of booleans indicating valid samples
    """
    try:
        # Reshape coordinates: (n_samples, 24) -> (n_samples, 2, 4, 3)
        n_samples = coordinates_array.shape[0]
        tetrahedra = coordinates_array.reshape(n_samples, 2, 4, 3)
        
        # Split into two tetrahedra
        tetra1 = tetrahedra[:, 0, :, :]  # (n_samples, 4, 3)
        tetra2 = tetrahedra[:, 1, :, :]  # (n_samples, 4, 3)
        
        # Calculate volumes for all tetrahedra at once
        volumes1 = calculate_tetrahedron_volumes_vectorized(tetra1)
        volumes2 = calculate_tetrahedron_volumes_vectorized(tetra2)
        
        # Check if both volumes meet threshold
        valid_mask = (volumes1 >= min_volume_threshold) & (volumes2 >= min_volume_threshold)
        
        return valid_mask
        
    except Exception as e:
        print(f"Warning: Volume calculation failed: {e}")
        return np.zeros(coordinates_array.shape[0], dtype=bool)

def calculate_tetrahedron_volumes_vectorized(tetrahedra):
    """
    Vectorized tetrahedron volume calculation for multiple tetrahedra.
    
    Args:
        tetrahedra: numpy array of shape (n_samples, 4, 3)
        
    Returns:
        numpy array of volumes
    """
    try:
        # Use first vertex as reference point
        v0 = tetrahedra[:, 0, :]  # (n_samples, 3)
        
        # Calculate edge vectors from v0 to other vertices
        edge_vectors = tetrahedra[:, 1:, :] - v0[:, np.newaxis, :]  # (n_samples, 3, 3)
        
        # Calculate determinants for all samples at once
        determinants = np.linalg.det(edge_vectors)  # (n_samples,)
        
        # Calculate volumes: |det| / 6
        volumes = np.abs(determinants) / 6.0
        
        return volumes
    except Exception as e:
        print(f"Error in volume calculation: {e}")
        return np.zeros(tetrahedra.shape[0])

def validate_coordinate_bounds_vectorized(coordinate_data):
    """
    Validate that all coordinates are within [0, 1] bounds.
    
    Args:
        coordinate_data: numpy array of shape (n_samples, 24)
        
    Returns:
        numpy array of booleans indicating valid samples
    """
    min_coords = np.min(coordinate_data, axis=1)
    max_coords = np.max(coordinate_data, axis=1)
    return (min_coords >= 0.0) & (max_coords <= 1.0)

def apply_quality_filters_vectorized(data, min_volume_threshold=1e-10):
    """
    Apply all quality filters to data using vectorized operations.
    
    Args:
        data: pandas DataFrame with coordinate and label columns
        min_volume_threshold: minimum volume threshold for tetrahedra
        
    Returns:
        Filtered pandas DataFrame
    """
    if len(data) == 0:
        return data
        
    try:
        # Get coordinate columns (exclude last 2 label columns)
        coordinate_columns = data.columns[:-2]
        
        # Step 1: Vectorized normalization check
        coordinate_data = data[coordinate_columns].values.astype(np.float32)
        
        # Check bounds efficiently
        normalization_mask = validate_coordinate_bounds_vectorized(coordinate_data)
        
        # Apply normalization filter
        data_normalized = data[normalization_mask].copy()
        
        if len(data_normalized) == 0:
            return data_normalized
        
        # Step 2: Vectorized volume check
        normalized_coords = data_normalized[coordinate_columns].values.astype(np.float32)
        
        # Calculate volumes for all samples at once
        volume_mask = validate_tetrahedron_volumes_vectorized(normalized_coords, min_volume_threshold)
        
        # Apply volume filter
        volume_filtered_data = data_normalized[volume_mask].copy()
        
        return volume_filtered_data.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error in quality filtering: {e}")
        return pd.DataFrame()

def fix_data_types_vectorized(chunk_df):
    """
    Fix data types for a chunk of data to prevent object array issues.
    
    Args:
        chunk_df: pandas DataFrame chunk
        
    Returns:
        DataFrame with fixed data types
    """
    # Fix object arrays and mixed types
    for col_idx, col in enumerate(chunk_df.columns):
        if col_idx < len(chunk_df.columns) - 2:  # Coordinate columns
            # Handle object arrays by converting to string then float
            if chunk_df[col].dtype == 'object':
                chunk_df[col] = chunk_df[col].astype(str)
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').astype(np.float32)
            
        elif col_idx == len(chunk_df.columns) - 2:  # Volume column
            if chunk_df[col].dtype == 'object':
                chunk_df[col] = chunk_df[col].astype(str)
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').astype(np.float32)
            
        else:  # Label column
            if chunk_df[col].dtype == 'object':
                chunk_df[col] = chunk_df[col].astype(str)
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
            # Convert to int, but first ensure no infinite or NaN values
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
            chunk_df[col] = chunk_df[col].fillna(0)  # Replace NaN with 0
            chunk_df[col] = chunk_df[col].replace([np.inf, -np.inf], 0)  # Replace inf with 0
            chunk_df[col] = chunk_df[col].astype(np.int32)
    
    # Remove any rows with NaN values in coordinate columns
    coordinate_cols = chunk_df.columns[:-2]
    chunk_df = chunk_df.dropna(subset=coordinate_cols)
    
    return chunk_df

def apply_unitary_tetrahedron_transformation_batch(data_tensor):
    """
    Vectorized version of unitary tetrahedron transformation for batch processing.
    
    Args:
        data_tensor: torch.Tensor of shape (batch_size, 24) containing tetrahedron pairs
        
    Returns:
        torch.Tensor of shape (batch_size, 12) with transformed T2 coordinates only
    """
    batch_size = data_tensor.shape[0]
    
    # Reshape into tetrahedra: (batch_size, 2, 4, 3)
    tetrahedra = data_tensor.reshape(batch_size, 2, 4, 3)
    tetra1_batch = tetrahedra[:, 0, :, :]  # (batch_size, 4, 3) - first tetrahedra
    tetra2_batch = tetrahedra[:, 1, :, :]  # (batch_size, 4, 3) - second tetrahedra
    
    # Get reference points (first vertices) for all samples
    v0_batch = tetra1_batch[:, 0, :]  # (batch_size, 3)
    
    # Calculate edge vectors for all first tetrahedra
    edge_vectors_batch = tetra1_batch[:, 1:, :] - v0_batch.unsqueeze(1)  # (batch_size, 3, 3)
    
    # Batch matrix inversion - handle singular matrices
    try:
        inv_transform_batch = torch.inverse(edge_vectors_batch)  # (batch_size, 3, 3)
    except RuntimeError:
        # Handle singular matrices individually
        inv_transform_batch = torch.zeros_like(edge_vectors_batch)
        for i in range(batch_size):
            try:
                inv_transform_batch[i] = torch.inverse(edge_vectors_batch[i])
            except RuntimeError:
                # Use identity matrix for singular cases
                inv_transform_batch[i] = torch.eye(3, dtype=data_tensor.dtype, device=data_tensor.device)
    
    # Transform all second tetrahedra
    translated_batch = tetra2_batch - v0_batch.unsqueeze(1)  # (batch_size, 4, 3)
    
    # Batch matrix multiplication: (batch_size, 4, 3) @ (batch_size, 3, 3) -> (batch_size, 4, 3)
    transformed_tetra2_batch = torch.bmm(translated_batch, inv_transform_batch)
    
    # Return only the transformed T2 coordinates (12 features per sample)
    return transformed_tetra2_batch.reshape(batch_size, 12)

def apply_principal_axis_transformation_batch(data_tensor):
    """
    Vectorized version of principal axis transformation for batch processing.
    Identical results to the single-sample version.
    
    Args:
        data_tensor: torch.Tensor of shape (batch_size, 24) containing tetrahedron pairs
        
    Returns:
        torch.Tensor of same shape with transformed data
    """
    batch_size = data_tensor.shape[0]
    
    # Reshape into tetrahedra: (batch_size, 2, 4, 3)
    tetrahedra = data_tensor.reshape(batch_size, 2, 4, 3)
    tetra1_batch = tetrahedra[:, 0, :, :]  # (batch_size, 4, 3)
    tetra2_batch = tetrahedra[:, 1, :, :]  # (batch_size, 4, 3)
    
    # Compute centroids for all first tetrahedra
    centroid1_batch = tetra1_batch.mean(dim=1)  # (batch_size, 3)
    
    # Center all first tetrahedra
    tetra1_centered_batch = tetra1_batch - centroid1_batch.unsqueeze(1)  # (batch_size, 4, 3)
    
    # Compute covariance matrices: cov = tetra1_centered.T @ tetra1_centered
    cov_matrices_batch = torch.bmm(
        tetra1_centered_batch.transpose(1, 2),  # (batch_size, 3, 4)
        tetra1_centered_batch                   # (batch_size, 4, 3)
    )  # (batch_size, 3, 3)
    
    # Compute eigendecomposition
    eigenvalues_batch, eigenvectors_batch = torch.linalg.eigh(cov_matrices_batch)
    
    # Sort eigenvalues and eigenvectors in descending order (largest variance first)
    sorted_indices_batch = torch.argsort(eigenvalues_batch, dim=1, descending=True)
    
    # FIXED: Proper eigenvector selection to match original
    # For each sample, select columns (eigenvectors) according to sorted eigenvalue indices
    R_canonical_batch = torch.zeros_like(eigenvectors_batch)
    for i in range(batch_size):
        R_canonical_batch[i] = eigenvectors_batch[i][:, sorted_indices_batch[i]]
    
    # Apply transformation: R_canonical.T @ (tetrahedra - centroid1)
    # This matches the original: transform = lambda x: (R_canonical.T @ (x - centroid1).T).T
    
    # Transform first tetrahedra
    tetra1_transformed_batch = torch.bmm(
        R_canonical_batch.transpose(1, 2),  # (batch_size, 3, 3) -> R.T
        tetra1_centered_batch.transpose(1, 2)  # (batch_size, 3, 4)
    ).transpose(1, 2)  # (batch_size, 4, 3)
    
    # Transform second tetrahedra
    tetra2_centered_batch = tetra2_batch - centroid1_batch.unsqueeze(1)  # (batch_size, 4, 3)
    tetra2_transformed_batch = torch.bmm(
        R_canonical_batch.transpose(1, 2),  # (batch_size, 3, 3) -> R.T
        tetra2_centered_batch.transpose(1, 2)  # (batch_size, 3, 4)
    ).transpose(1, 2)  # (batch_size, 4, 3)
    
    # Combine results
    result_tetrahedra = torch.stack([tetra1_transformed_batch, tetra2_transformed_batch], dim=1)
    
    # Flatten back to original shape
    return result_tetrahedra.reshape(batch_size, 24)