"""Implementation of a rehearsal-based KAN model and a self-organizing maps-based
layer it relies on"""

import torch
from torch import nn

from src.nn_modules.kan import KANLayer


class SOMLayer(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr: float = 1e-1,
        sharp_order: int = 3,
        sparse_min: float = 0.5,
        train_prototypes: bool = True,
    ):
        # A square grid is used for self-organizing maps, therefore som_features
        # must be a square of an integer
        assert int(out_features**0.5) ** 2 == out_features
        super().__init__()
        self.grid_var = 0.3 * (out_features**0.5)
        self.weights_var = 1
        self.lr = lr
        self.sharp_order = sharp_order
        self.sparse_min = sparse_min
        self.train_prototypes = train_prototypes
        self.register_buffer("prototypes", torch.rand(out_features, in_features))
        idx = torch.arange(out_features).expand(out_features, -1)
        # Tensor with grid coordinates for each prototype
        self.prototypes_coords = torch.stack(
            [idx.T.reshape(-1), idx.reshape(-1)]
        ).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sparse_similarities = self._get_neural_activity(x)
        preds = self.classifier(sparse_similarities)
        if self.training and self.train_prototypes:
            with torch.no_grad():
                self._update_prototypes(x)
        return preds

    def _get_neural_activity(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate neural activity tensor by measuring space similarity between
        input tensor and the prototypes"""
        similarities = self._similarity_function(x)
        sparse_similarities = self._transfer_function(similarities)
        return sparse_similarities

    def _update_prototypes(self, x: torch.Tensor) -> None:
        """"""
        bmu = self._best_matching_units(x)
        grid_neighbourhood = self._grid_neighbourhood(bmu)

    def _similarity_function(self, x: torch.Tensor) -> torch.Tensor:
        """Measure similarity between an input tensor and the prototypes via
        zero-mean Gaussian function"""
        distances = torch.norm(self.prototype - x, dim=1)
        return _gaussian(distances, self.grid_var)

    def _transfer_function(self, x: torch.Tensor) -> torch.Tensor:
        """Sparsify similarities between an input tensor and prototypes"""
        sharp_x = x.pow(self.sharp_order) / (x.max() + 1e-8).pow(self.sharp_order - 1)
        sparse_x = torch.where(sharp_x >= self.sparse_min, sharp_x, torch.tensor(0.0))
        return sparse_x

    def _best_matching_units(self, x: torch.Tensor) -> torch.Tensor:
        """Get indices for nearest prototypes to each of the input vectors"""
        return torch.norm(self.prototype.unsqueeze(0) - x.unsqueeze(1), dim=2).argmin(
            dim=1
        )

    def _grid_neighbourhood(self, bmu: torch.Tensor) -> torch.Tensor:
        """Calculate a grid similarity between given best matching units and other
        prototypes."""

        grid_coords = torch.stack(
            bmu // self.prototypes.size(), bmu % self.prototypes.size(), dim=1
        ).float()
        grid_distances = torch.norm(
            grid_coords.unsqueeze(1) - self.prototypes_coords.unsqueeze(0), dim=2
        )
        grid_similarities = _gaussian(grid_distances, self.grid_var)
        return grid_similarities


def _gaussian(x: torch.Tensor, variance: float = 1) -> torch.Tensor:
    """Calculate zero-mean Gauss function with a given variance"""
    return torch.exp(-x.pow(2) / (2 * variance**2))


# class SOMClassifier(torch.nn.Module):

#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int,
#         out_features: int,
#         grid_size: int = 3,
#     ):
#         super().__init__()
#         self.fc = nn.Sequential(
#             KANLayer(in_features, hidden_features, grid_size),
#             KANLayer(hidden_features, out_features, grid_size),
#         )

#     def forward(self, x):
#         return self.fc(x)


t = SOMLayer(3, 4)
bums = torch.tensor([1, 2, 3])
