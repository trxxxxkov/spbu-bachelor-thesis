"""Implementation of a rehearsal-based KAN model and a self-organizing maps-based
layer it relies on"""

import torch
from tqdm import tqdm

from src.nn_modules.kan import KANLayer


class SOMLayer(torch.nn.Module):
    """Self-Organizing Map layer used in the Rehearsal KAN architecture. The map
    has a square topology and is updated manually, once per training epoch, by
    calling an update() method."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        variance_init: float = 0.5,
        variance_lr: float = 0.05,
    ):
        # A square grid topology is used for the self-organizing map and
        # out_features is equal to a number of prototypes in it
        if int(out_features**0.5) ** 2 != out_features:
            raise ValueError("out_features must be a perfect square")
        super().__init__()

        self.fl = torch.nn.Flatten()

        self.side = int(out_features**0.5)
        self.var_lr = variance_lr
        self.var_init = variance_init
        self.weight: torch.Tensor
        self.register_buffer("weight", torch.rand(out_features, in_features))
        idx = torch.arange(self.side).expand(self.side, -1)
        # Tensor with grid coordinates for each prototype
        self.register_buffer(
            "coords",
            torch.stack([idx.T.reshape(-1), idx.reshape(-1)], dim=1).float(),
        )
        self.register_buffer("input_variance", torch.tensor(self.var_init))
        self.register_buffer("grid_variance", torch.tensor(self.var_init * self.side))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return neural activity tensor"""

        x = self.fl(x)

        similarities = self._get_input_space_similarities(x)
        if self.training:
            self.update(x)
        else:
            self.grid_variance.copy_(self.var_init * self.side)
        # sparse_similarities = self._sparsify_similarities(similarities)
        # return sparse_similarities
        return similarities

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Move prototypes toward corresponding BMUs and update variances"""
        x = self.fl(x)
        input_dist = torch.cdist(x, self.weight)
        # Get indices for the best matching units (nearest prototypes to
        # each of the input vectors)
        bmu_idx = input_dist.argmin(dim=1)
        grid_similarities = self._get_grid_similarities(bmu_idx)
        numerator = grid_similarities.T @ x
        denominator = grid_similarities.sum(dim=0)
        # Batch-SOM update rule: sum(similarities * x) / sum(similarities)
        self.weight.copy_((numerator / denominator.unsqueeze(1)).nan_to_num(0.0))
        self.input_variance.copy_(
            (1 - self.var_lr) * self.input_variance + self.var_lr * input_dist.mean()
        )
        self.grid_variance.copy_((1 - self.var_lr) * self.grid_variance)

    # def _sparsify_similarities(self, similarities: torch.Tensor) -> torch.Tensor:
    #     """Sparsify similarities by sharpening it and comparing with a threshold"""
    #     sharp_similarities = similarities.pow(self.sharp_order) / (
    #         similarities.max(dim=1, keepdim=True).values + 1e-6
    #     ).pow(self.sharp_order - 1)
    #     sparse_mask = (sharp_similarities >= self.sparse_min).float()
    #     return sharp_similarities * sparse_mask

    def _get_grid_similarities(self, bmu_idx: torch.Tensor) -> torch.Tensor:
        """Calculate a grid similarity between given best matching units and
        other prototypes."""
        bmu_coords = self.coords[bmu_idx]
        grid_distances = torch.cdist(bmu_coords, self.coords).to(bmu_idx.device)
        grid_similarities = _gaussian(grid_distances, self.grid_variance)
        return grid_similarities

    def _get_input_space_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Measure similarity between an input tensor and self.prototypes"""
        input_space_distances = torch.cdist(x, self.weight)
        return _gaussian(input_space_distances, self.input_variance)


def _gaussian(x: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Calculate zero-mean Gauss function with a given variance"""
    return torch.exp(-x.pow(2) / (2 * variance.clamp(min=1e-6) ** 2)).to(x.device)


class RehearsalKAN(torch.nn.Module):
    """Kolmogorov-Arnold Network model improved by using a self-organazing map
    that is filled with data distribution information.

    This model is supposed to be trained in a continual learning settings it
    requires initialization of the SOM layer with samples from the base dataset
    before training begins.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 64,
        output_dim: int = 200,
        som_variance_init: float = 0.5,
        som_variance_lr: float = 0.05,
    ):
        super().__init__()
        self.som = SOMLayer(input_dim, hidden_dim, som_variance_init, som_variance_lr)
        self.classifier = KANLayer(hidden_dim, output_dim)
        # self.classifier = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        similarities = self.som(x)
        preds = self.classifier(similarities)
        return preds

    def som_init(
        self,
        trainloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Perform initial optimization of the SOM layer by moving prototypes
        towards samples from dataset's base classes"""
        for batch_inputs, _ in tqdm(
            trainloader, desc="SOMLayer initialization", leave=False
        ):
            self.som.update(batch_inputs.to(device))
