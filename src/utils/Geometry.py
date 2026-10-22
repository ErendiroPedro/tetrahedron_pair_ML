import torch
import numpy as np
import pandas as pd

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
