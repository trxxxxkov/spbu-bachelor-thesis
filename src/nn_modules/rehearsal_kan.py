"""Implementation of a rehearsal-based KAN model and a self-organizing maps-based
layer it relies on"""

import torch

from src.nn_modules.kan import KANLayer


class SOMLayer(torch.nn.Module):
    """Self-Organizing Map layer used in the Rehearsal KAN architecture. The map
    has a square topology and is updated manually, once per training epoch, by
    calling an update() method."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sharp_order: int = 3,
        sparse_min: float = 0.5,
    ):
        # A square grid topology is used for the self-organizing map and
        # out_features is equal to a number of prototypes in it
        if int(out_features**0.5) ** 2 != out_features:
            raise ValueError("out_features must be a perfect square")
        super().__init__()
        self.side = int(out_features**0.5)
        self.sharp_order = sharp_order
        self.sparse_min = sparse_min
        # self.prototypes: torch.Tensor
        # self.prototypes_coords: torch.Tensor
        # self.input_variance: float
        # self.grid_variance: float
        self.register_buffer("prototypes", torch.rand(out_features, in_features))
        idx = torch.arange(self.side).expand(self.side, -1)
        # Tensor with grid coordinates for each prototype
        self.register_buffer(
            "prototypes_coords",
            torch.stack([idx.T.reshape(-1), idx.reshape(-1)], dim=1).float(),
        )
        self.register_buffer("input_variance", torch.tensor(1.0))
        self.register_buffer("grid_variance", torch.tensor(0.3 * self.side))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return neural activity tensor"""
        similarities = self._get_input_space_similarities(x)
        sparse_similarities = self._sparsify_similarities(similarities)
        return sparse_similarities

    @torch.no_grad()
    def update(self, trainloader: torch.utils.data.DataLoader) -> None:
        """Move prototypes toward corresponding BMUs and update input_variance"""
        numerator = torch.zeros_like(self.prototypes)
        denominator = torch.zeros(self.side**2, device=self.prototypes.device)
        epoch_num_samples = 0
        epoch_mean_dist = 0.0
        for batch_inputs, _ in trainloader:
            batch_inputs = batch_inputs.to(self.prototypes.device)
            # Get indices for the best matching units (nearest prototypes to
            # each of the input vectors)
            batch_dist = torch.cdist(batch_inputs, self.prototypes)
            bmu = batch_dist.argmin(dim=1)
            # Prototypes-inputs mean distance is used to update input space
            # variance involved in calculation of input-space similarities
            epoch_mean_dist += batch_dist.mean()
            epoch_num_samples += len(batch_inputs)
            grid_similarities = self._get_grid_similarities(bmu)
            # Batch-SOM update rule: sum(similarities * x) / sum(similarities)
            numerator += grid_similarities.T @ batch_inputs
            denominator += grid_similarities.sum(dim=0)
        self.prototypes.copy_(numerator / denominator.clamp_(min=1e-8).unsqueeze(1))
        # Input space similarity is bound with the current batch's
        # prototypes-inputs distances
        epoch_mean_dist /= epoch_num_samples
        self.input_variance.copy_(0.9 * self.input_variance + 0.1 * epoch_mean_dist)

    def _sparsify_similarities(self, similarities: torch.Tensor) -> torch.Tensor:
        """Sparsify similarities by sharpening it and comparing with a threshold"""
        sharp_similarities = similarities.pow(self.sharp_order) / (
            similarities.max(dim=1, keepdim=True).values + 1e-6
        ).pow(self.sharp_order - 1)
        sparse_mask = (sharp_similarities >= self.sparse_min).float()
        return sharp_similarities * sparse_mask

    def _get_grid_similarities(self, bmu: torch.Tensor) -> torch.Tensor:
        """Calculate a grid similarity between given best matching units and
        other prototypes."""
        bmu_grid_coords = torch.stack(
            (bmu // self.side, bmu % self.side), dim=1
        ).float()
        grid_distances = torch.norm(
            bmu_grid_coords.unsqueeze(1) - self.prototypes_coords.unsqueeze(0), dim=2
        )
        grid_similarities = _gaussian(grid_distances, self.grid_variance)
        return grid_similarities

    def _get_input_space_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Measure similarity between an input tensor and self.prototypes"""
        input_space_distances = torch.norm(
            self.prototypes.unsqueeze(0) - x.unsqueeze(1), dim=2
        )
        return _gaussian(input_space_distances, self.input_variance)


def _gaussian(x: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Calculate zero-mean Gauss function with a given variance"""
    return torch.exp(-x.pow(2) / (2 * variance.clamp(min=1e-6) ** 2))


class RehearsalKAN(torch.nn.Module):
    """"""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 64,
        output_dim: int = 200,
    ):
        super().__init__()
        self.som = SOMLayer(input_dim, hidden_dim)
        self.classifier = KANLayer(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        sparse_similarities = self.som(x)
        return self.classifier(sparse_similarities)
