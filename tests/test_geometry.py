import pytest
import torch

from src.utils.geometry import (
    permute_vertices_within_tetrahedra,
    principal_axis_transform,
    swap_tetrahedra,
    unitary_tetrahedron_transform,
)

SIMPLEX = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)


@pytest.fixture
def pairs():
    return torch.randn(50, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(0))


def test_swap_tetrahedra_exchanges_the_halves_and_is_an_involution(pairs):
    swapped = swap_tetrahedra(pairs)
    assert torch.equal(swapped[:, :12], pairs[:, 12:])
    assert torch.equal(swapped[:, 12:], pairs[:, :12])
    assert torch.equal(swap_tetrahedra(swapped), pairs)


@pytest.mark.parametrize("width", [12, 24])
def test_permute_vertices_only_reorders_vertices_within_each_tetrahedron(width):
    coordinates = torch.randn(200, width)
    original = coordinates.clone()
    permuted = permute_vertices_within_tetrahedra(coordinates)

    assert torch.equal(coordinates, original), "the input must not be modified"
    assert not torch.equal(permuted, coordinates)
    before, after = (c.reshape(200, width // 12, 4, 3) for c in (coordinates, permuted))
    same_vertex = ((after.unsqueeze(3) - before.unsqueeze(2)).abs().sum(-1) == 0)  # (n, tetrahedron, new, old)
    assert same_vertex.sum(dim=3).eq(1).all() and same_vertex.sum(dim=2).eq(1).all(), "not a permutation"


def test_permute_vertices_rejects_other_widths():
    with pytest.raises(ValueError, match="Unsupported tensor shape"):
        permute_vertices_within_tetrahedra(torch.zeros(3, 10))


def test_unitary_transform_expresses_t2_in_the_edge_basis_of_t1(pairs):
    t1 = pairs[:, :12]
    out = unitary_tetrahedron_transform(torch.cat([t1, t1], dim=1)).reshape(-1, 4, 3)  # T2 = T1
    torch.testing.assert_close(out, SIMPLEX.expand(len(pairs), 4, 3), atol=1e-9, rtol=0)


def test_unitary_transform_only_translates_when_t1_is_degenerate():
    coplanar_t1 = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float64)
    t2 = torch.randn(4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    out = unitary_tetrahedron_transform(torch.cat([coplanar_t1.flatten(), t2.flatten()]).unsqueeze(0))
    torch.testing.assert_close(out.reshape(4, 3), t2 - coplanar_t1[0])


def test_principal_axis_transform_is_rigid_and_centres_t1(pairs):
    before, after = (c.reshape(-1, 2, 4, 3) for c in (pairs, principal_axis_transform(pairs)))
    zeros = torch.zeros(len(pairs), 3, dtype=torch.float64)
    torch.testing.assert_close(after[:, 0].mean(dim=1), zeros, atol=1e-12, rtol=0)
    for tetrahedron in (0, 1):
        torch.testing.assert_close(torch.cdist(after[:, tetrahedron], after[:, tetrahedron]),
                                   torch.cdist(before[:, tetrahedron], before[:, tetrahedron]))


def test_principal_axis_transform_puts_t1_variance_in_descending_axis_order(pairs):
    t1 = principal_axis_transform(pairs).reshape(-1, 2, 4, 3)[:, 0]
    covariance = t1.transpose(1, 2) @ t1
    variances = torch.diagonal(covariance, dim1=1, dim2=2)
    assert (variances[:, 0] >= variances[:, 1] - 1e-12).all()
    assert (variances[:, 1] >= variances[:, 2] - 1e-12).all()
    assert (covariance - torch.diag_embed(variances)).abs().max() < 1e-10, "axes must be uncorrelated"
