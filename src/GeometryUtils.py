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

def permute_points_within_tetrahedrons(X):
    """
    Randomly permute points within each tetrahedron in the input tensor.
    Args:
        X (torch.Tensor): Input tensor of shape (batch_size, 24).
    Returns:
        torch.Tensor: Tensor with permuted points within tetrahedrons.
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

def larger_tetrahedron_first(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
    """Reorder tetrahedron_pair so larger tetrahedron comes first."""
    T1, T2 = tetrahedron_pair[:12], tetrahedron_pair[12:]
    
    vol1 = self.calculate_tetrahedron_volume(T1)
    vol2 = self.calculate_tetrahedron_volume(T2)
    
    return tetrahedron_pair if vol1 >= vol2 else torch.cat([T2, T1])

def sort_by_X_coordinate(tetrahedron_pair_vertices_flat: pd.DataFrame, column_name: str = "T1_v1_x") -> pd.DataFrame:
    """
    Sorts the given DataFrame by the specified column.
    
    Args:
        tetrahedron_pair_vertices_flat (pd.DataFrame): The input dataset containing the column to sort by.
        column_name (str): The name of the column to sort the DataFrame by.
    
    Returns:
        pd.DataFrame: Sorted DataFrame by the specified column in ascending order.
    """
    if column_name not in tetrahedron_pair_vertices_flat.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    return tetrahedron_pair_vertices_flat.sort_values(by=column_name, ascending=True).reset_index(drop=True)

def sort_by_morton_code(data: pd.DataFrame) -> pd.DataFrame:
    # Define a scaling factor to convert high precision coordinates into integers.
    # Adjust the factor based on the actual range of your data.
    scale_factor = 1e18 
    
    morton_codes = []
    
    # Iterate over each row/sample in the DataFrame.
    for idx, row in data.iterrows():
        # Extract the 24 coordinate features. Assuming the first 2 columns are not coordinates.
        coords = row.iloc[2:].to_numpy(dtype=float)
        # First tetrahedron: first 12 features (4 vertices × 3 coordinates)
        tetra1 = coords[:12].reshape((4, 3))
        # Second tetrahedron: next 12 features
        tetra2 = coords[12:].reshape((4, 3))
        
        # Compute centroids of each tetrahedron
        centroid1 = tetra1.mean(axis=0)
        centroid2 = tetra2.mean(axis=0)
        
        # Calculate the average (combined) centroid
        avg_centroid = (centroid1 + centroid2) / 2.0
        
        # Scale and convert each coordinate to an integer.
        # Rounding is applied to preserve numerical differences.
        x, y, z = [int(round(coord * scale_factor)) for coord in avg_centroid]
        
        # Compute the Morton code (Z-order) for the scaled centroid.
        code = pymorton.interleave(x, y, z)
        morton_codes.append(code)
    
    # Add a temporary column for sorting purposes.
    data['_morton_code'] = morton_codes
    # Sort by the Morton code and drop the temporary column before returning.
    sorted_df = data.sort_values(by='_morton_code').drop(columns=['_morton_code']).reset_index(drop=True)
    
    return sorted_df

def apply_rigid_transformation(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
    pass

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

def calculate_tetrahedron_volume(tetrahedron: torch.Tensor) -> torch.Tensor:
        """Calculate tetrahedron volume using signed tetrahedral volume formula."""
        v0, v1, v2, v3 = tetrahedron.view(4, 3)
        return torch.abs(torch.det(torch.stack([v1-v0, v2-v0, v3-v0]))) / 6
