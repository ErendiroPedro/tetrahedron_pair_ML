import numpy as np
import torch


def swap_tetrahedrons(X):
    """Exchanges the two tetrahedra of every (batch, 24) pair."""
    return torch.cat([X[:, 12:], X[:, :12]], dim=1)


def permute_points_within_tetrahedrons(X: torch.Tensor) -> torch.Tensor:
    """Returns a copy of X (batch, 12 or 24) with the 4 vertices of every tetrahedron in a random order."""
    batch_size, num_features = X.shape
    if num_features not in (12, 24):
        raise ValueError(f"Unsupported tensor shape: {X.shape}")

    vertices = X.reshape(batch_size, num_features // 12, 4, 3)
    order = torch.rand(batch_size, num_features // 12, 4, device=X.device).argsort(dim=-1)
    return vertices.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 3)).reshape(batch_size, num_features)


def apply_unitary_tetrahedron_transformation_batch(data_tensor):
    """(batch, 24) pairs -> (batch, 12): T2 in the basis spanned by T1's edges from its first vertex.

    Pairs whose T1 is degenerate (singular edge matrix) are only translated.
    """
    tetrahedra = data_tensor.reshape(-1, 2, 4, 3)
    v0 = tetrahedra[:, 0, :1]
    inverse, info = torch.linalg.inv_ex(tetrahedra[:, 0, 1:] - v0)
    identity = torch.eye(3, dtype=data_tensor.dtype, device=data_tensor.device)
    inverse = torch.where((info > 0)[:, None, None], identity, inverse)
    return ((tetrahedra[:, 1] - v0) @ inverse).reshape(-1, 12)


def apply_principal_axis_transformation_batch(data_tensor):
    """(batch, 24) -> (batch, 24): both tetrahedra centred on T1's centroid, rotated onto T1's principal axes."""
    tetrahedra = data_tensor.reshape(-1, 2, 4, 3)
    centered = tetrahedra - tetrahedra[:, 0].mean(dim=1)[:, None, None, :]
    t1 = centered[:, 0]
    eigenvalues, eigenvectors = torch.linalg.eigh(t1.transpose(1, 2) @ t1)
    largest_variance_first = eigenvalues.argsort(dim=1, descending=True)
    axes = eigenvectors.gather(2, largest_variance_first[:, None, :].expand(-1, 3, -1))
    return (centered @ axes[:, None]).reshape(-1, 24)


def calculate_tetrahedron_volume(tetrahedron_vertices):
    """Volume of one tetrahedron given as 4 vertices (array or CPU tensor, anything reshapeable to (4, 3))."""
    v0, v1, v2, v3 = np.asarray(tetrahedron_vertices).reshape(4, 3)
    return abs(np.linalg.det(np.stack([v1 - v0, v2 - v0, v3 - v0]))) / 6.0
