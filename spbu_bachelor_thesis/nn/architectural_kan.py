"""Implementation of a FELLayer class based on FEL method and a ArchitecturalKAN
that contains the FELLayer and is used in the CL experiments."""

import math
import torch
import torch.nn.functional as F

from spbu_bachelor_thesis.nn.kan import KANLayer


class FELLayer(torch.nn.Module):
    """Fixed-weight expansion layer with k-WTA gating & zero-affine LayerNorm."""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        expansion_factor: int = 10,
        inputs_fraction: float = None,
        outputs_fraction: float = None,
        device: torch.device = None,
    ):
        super().__init__()
        # Hyperparameters derivation
        self.in_features = in_features
        self.out_features = out_features or in_features * expansion_factor
        # Amount of inputs per node.
        self.inputs_per_node = max(1, int((inputs_fraction or 0.3) * in_features))
        if outputs_fraction is not None:
            self.k = max(1, int(outputs_fraction * self.out_features))
        else:
            self.k = max(1, int(math.sqrt(self.out_features)))
        weight = torch.zeros(
            self.out_features, in_features, device=device, dtype=torch.float
        )
        # Weights initialization with Kaiming normal: N(0, 2/inputs_per_node)
        for row in range(self.out_features):
            idx = torch.randperm(in_features, device=device)[: self.inputs_per_node]
            weight[row, idx] = torch.randn(
                self.inputs_per_node, device=device, dtype=torch.float
            ) * math.sqrt(2.0 / self.inputs_per_node)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = F.linear(x, self.weight)  # pylint: disable=E1102
        # Non-trainable normalization
        outputs = F.layer_norm(outputs, (self.out_features,), None, None, 1e-5)
        # Get indices of k neurons with the largest activation values
        winners_idx = torch.topk(outputs.abs(), self.k, dim=1, sorted=False).indices
        sparse_mask = torch.zeros_like(outputs, dtype=torch.bool)
        # Remove signal from the neurons other than the winners
        sparse_mask.scatter_(1, winners_idx, True)
        return outputs * sparse_mask


class ArchitecturalKAN(torch.nn.Module):
    """KAN-based architecture that uses FELLayer to overcome the catastrophic
    forgetting"""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 300,
        output_dim: int = 200,
        expansion_factor: int = 10,
    ):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            KANLayer(input_dim, hidden_dim),
            FELLayer(hidden_dim, expansion_factor=expansion_factor),
            KANLayer(hidden_dim * expansion_factor, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
