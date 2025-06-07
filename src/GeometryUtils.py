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

    if num_features == 24:  # Standard case: a pair of tetrahedrons
        # Split into two tetrahedrons
        tetra1_data = X[:, :12].view(batch_size, 4, 3)  # (B, 4, 3)
        tetra2_data = X[:, 12:].view(batch_size, 4, 3) # (B, 4, 3)
        
        # Generate permutations for the first tetrahedron in each pair
        perm1 = torch.stack([torch.randperm(4, device=device) for _ in range(batch_size)])
        perm1_expanded = perm1.unsqueeze(-1).expand(-1, 4, 3)
        permuted_tetra1 = torch.gather(tetra1_data, 1, perm1_expanded)
        
        # Generate permutations for the second tetrahedron in each pair
        perm2 = torch.stack([torch.randperm(4, device=device) for _ in range(batch_size)])
        perm2_expanded = perm2.unsqueeze(-1).expand(-1, 4, 3)
        permuted_tetra2 = torch.gather(tetra2_data, 1, perm2_expanded)
        
        # Reconstruct features
        return torch.cat([
            permuted_tetra1.view(batch_size, 12),
            permuted_tetra2.view(batch_size, 12)
        ], dim=1)

    elif num_features == 12:  # Case: a single tetrahedron
        tetra_data = X.view(batch_size, 4, 3)  # (B, 4, 3)
        
        # Generate permutations for the single tetrahedron
        perm = torch.stack([torch.randperm(4, device=device) for _ in range(batch_size)])
        perm_expanded = perm.unsqueeze(-1).expand(-1, 4, 3)
        permuted_tetra = torch.gather(tetra_data, 1, perm_expanded)
        
        # Reconstruct features
        return permuted_tetra.view(batch_size, 12)
        
    else:
        print(f"Warning: permute_points_within_tetrahedrons received input with {num_features} features. Expected 12 or 24. Returning input unchanged.")
        return X

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

def apply_unitary_tetrahedron_transformation(tetrahedron_pair_vertices_flat: pd.Series) -> pd.Series:
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

def apply_principal_axis_transformation(tetrahedron_pair_vertices_flat: pd.Series) -> pd.Series:
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

def _calculate_centroid_distance(row: pd.Series) -> float:
    """
    Calculate Euclidean distance between centroids of two tetrahedra.
    
    Args:
        row: DataFrame row containing tetrahedron pair coordinates
        
    Returns:
        Euclidean distance between the two tetrahedron centroids
    """
    # Extract tetrahedron coordinates (skip first 2 columns if they're metadata)
    tetrahedron_coords = row.iloc[:-2].values.astype(np.float32)
    
    # Split into two tetrahedra (12 coordinates each)
    tetra1_coords = tetrahedron_coords[:12].reshape(4, 3)
    tetra2_coords = tetrahedron_coords[12:].reshape(4, 3)
    
    # Calculate centroids
    centroid1 = tetra1_coords.mean(axis=0)
    centroid2 = tetra2_coords.mean(axis=0)
    
    # Calculate Euclidean distance
    centroid_distance = np.linalg.norm(centroid2 - centroid1)
    
    return centroid_distance