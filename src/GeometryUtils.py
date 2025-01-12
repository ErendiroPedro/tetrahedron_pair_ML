import torch
import numpy as np
import pandas as pd

class GeometryUtils:
    @staticmethod
    def larger_tetrahedron_first(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        """Reorder tetrahedron_pair so larger tetrahedron comes first."""
        T1, T2 = tetrahedron_pair[:4], tetrahedron_pair[4:]
        
        vol1 = self._calculate_tetrahedron_volume(T1)
        vol2 = self._calculate_tetrahedron_volume(T2)
        
        return tetrahedron_pair if vol1 >= vol2 else torch.cat([T2, T1])
    
    def sort_by_X_coordinate(self, tretrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass

    def sort_by_space_filling_curve(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass

    def permute_vertices(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass

    def permute_tetrahedron(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass

    def apply_rigid_transformation(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass

    def apply_affine_linear_transformation(self, tetrahedron_pair: torch.Tensor) -> torch.Tensor:
        pass


    # Helper functions
    
    def _calculate_tetrahedron_volume(vertices: torch.Tensor) -> torch.Tensor:
        """Calculate tetrahedron volume using signed tetrahedral volume formula."""
        v0, v1, v2, v3 = vertices
        return torch.abs(torch.det(torch.stack([v1-v0, v2-v0, v3-v0]))) / 6