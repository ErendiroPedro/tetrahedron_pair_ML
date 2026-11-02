import torch


def swap_tetrahedra(pair_coordinates):
    """Exchanges the two tetrahedra of every (batch, 24) pair."""
    return torch.cat([pair_coordinates[:, 12:], pair_coordinates[:, :12]], dim=1)


def permute_vertices_within_tetrahedra(coordinates: torch.Tensor) -> torch.Tensor:
    """Returns a copy of `coordinates` (batch, 12 or 24) with the 4 vertices of every tetrahedron in a random order."""
    batch_size, num_coordinates = coordinates.shape
    if num_coordinates not in (12, 24):
        raise ValueError(f"Unsupported tensor shape: {coordinates.shape}")

    vertices = coordinates.reshape(batch_size, num_coordinates // 12, 4, 3)
    random_order = torch.rand(batch_size, num_coordinates // 12, 4, device=coordinates.device).argsort(dim=-1)
    return vertices.gather(2, random_order.unsqueeze(-1).expand(-1, -1, -1, 3)).reshape(batch_size, num_coordinates)


def unitary_tetrahedron_transform(pair_coordinates):
    """(batch, 24) pairs -> (batch, 12): T2 in the basis spanned by T1's edges from its first vertex.

    Pairs whose T1 is degenerate (singular edge matrix) are only translated.
    """
    tetrahedra = pair_coordinates.reshape(-1, 2, 4, 3)
    first_vertex = tetrahedra[:, 0, :1]
    inverse_edges, info = torch.linalg.inv_ex(tetrahedra[:, 0, 1:] - first_vertex)
    identity = torch.eye(3, dtype=pair_coordinates.dtype, device=pair_coordinates.device)
    inverse_edges = torch.where((info > 0)[:, None, None], identity, inverse_edges)
    return ((tetrahedra[:, 1] - first_vertex) @ inverse_edges).reshape(-1, 12)


def principal_axis_transform(pair_coordinates):
    """(batch, 24) -> (batch, 24): both tetrahedra centred on T1's centroid, rotated onto T1's principal axes."""
    tetrahedra = pair_coordinates.reshape(-1, 2, 4, 3)
    centered = tetrahedra - tetrahedra[:, 0].mean(dim=1)[:, None, None, :]
    t1 = centered[:, 0]
    eigenvalues, eigenvectors = torch.linalg.eigh(t1.transpose(1, 2) @ t1)
    largest_variance_first = eigenvalues.argsort(dim=1, descending=True)
    principal_axes = eigenvectors.gather(2, largest_variance_first[:, None, :].expand(-1, 3, -1))
    return (centered @ principal_axes[:, None]).reshape(-1, 24)
