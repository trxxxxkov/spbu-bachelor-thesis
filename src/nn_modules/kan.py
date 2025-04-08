"""Implementation of a KAN layer as a custom PyTorch class"""

import math

import torch
import torch.nn.functional as F


class KANLayer(torch.nn.Module):
    """Highly optimized KAN layer that parametrizes the activation functions with
    basis splines of fixed order.

    Key differences from other implementations:
    1. B-splines has a fixed order of 4 (polynomial functions of degree 3).
    2. Output is a B-splines linear combination only (no classic activation
        function is added and no scaling is performed).
    3. Input range is fixed (and so is B-spline grid): [0, 1];
    4. Has a batch normalization layer built-in;"""

    spline_order = 3

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_step = 1 / grid_size
        self.bn = torch.nn.BatchNorm1d(in_features)
        knots_num = math.ceil(1 / self.grid_step) + 1 + self.spline_order
        # Coefficients for basis functions (aka control points)
        self.bspline_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, knots_num)
        )
        grid = (
            torch.arange(
                -self.spline_order, knots_num - self.spline_order, dtype=torch.float
            )
            * self.grid_step
        ).contiguous()
        # Explicit formulas for cubic B-spline on the four intervals
        cubic_bspline_formula = torch.tensor(
            [
                [1 / 6, 0, 0, 0],  # B(s)=s³/6, s ∈[0,1)
                [-3 / 6, 12 / 6, -12 / 6, 4 / 6],  # B(s)=(-3s³+12s²-12s+4)/6, s ∈[1,2)
                [3 / 6, -24 / 6, 60 / 6, -44 / 6],  # B(s)=(3s³-24s²+60s-44)/6, s ∈[2,3)
                [-1 / 6, 12 / 6, -48 / 6, 64 / 6],  # B(s)=(-s³+12s²-48s+64)/6, s ∈[3,4)
            ],
            dtype=torch.float,
        ).contiguous()
        self.register_buffer("grid", grid)
        self.register_buffer("cubic_bspline_formula", cubic_bspline_formula)
        self._initialize_params()

    def _initialize_params(self):
        """Initialize B-splines' coefficients by interpolating uniform noise.

        A least squares problem is considered for Ax = B, where
            x - B-splines' values computed at grid knots (4 per each value in x),
            A - optimal control points, that interpolate reference values,
            B - reference values sampled from a scaled ~U(0, 1) distribution.
        """
        with torch.no_grad():
            # Get knots inside the grid range
            interpolation_nodes = self.grid.expand(self.in_features, -1).T[
                self.spline_order :
            ]
            # B-splines' values for the chosen points. This is "x" in the system
            # of linear equations mentioned in the method's docstring.
            current_values = self._bsplines_values_at(interpolation_nodes).transpose(
                0, 1
            )
            # Spline values to be interpolated by adjusting coefficients of
            # B-splines considering current_values. This is "B" in the
            # Ax = B system of linear equations.
            reference_values = (
                torch.randn(
                    self.in_features,
                    len(self.grid) - self.spline_order,
                    self.out_features,
                )
            ) / (10 * (len(self.grid) - self.spline_order))
            # Optimal B-splines' coefficients that interpolate the reference
            # values
            coefficients = torch.linalg.lstsq(  # pylint: disable=E1102
                current_values, reference_values
            ).solution
            self.bspline_coeffs.data.copy_(coefficients.permute(2, 0, 1))

    def _bsplines_values_at(self, x: torch.Tensor):
        """Compute values of all cubic B-splines at input point x.

        Normalize x to grid scale, find spline_order+1 B-splines per input that
        defines the output value and compute B-splines at those positions.

            Args:
                x: Input tensor with values in [0,1] range

            Returns:
                bsplines_values: Tensor of shape [batch_size, in_features, len(grid))
                    where each element contains the evaluated i-th left B-splines
                    at point x.
        """
        x_norm = (x - self.grid[0]) / self.grid_step
        closest_left_knot = torch.floor(x_norm).long()
        bsplines_values = torch.zeros((*x.shape, len(self.grid)), device=x.device)
        for i in range(4):
            # Retrieve coefficients for the explicit formula of a B-spline for the
            # current interval
            coeff_row = self.cubic_bspline_formula[3 - i]
            ith_left_knot = closest_left_knot - 3 + i
            # Calculate input value for the i-th left knot considering it's distance
            ith_left_bspline_arg = (3 - i) + (x_norm - closest_left_knot)
            ith_left_bspline_value = (
                coeff_row[0] * ith_left_bspline_arg.pow(3)
                + coeff_row[1] * ith_left_bspline_arg.pow(2)
                + coeff_row[2] * ith_left_bspline_arg
                + coeff_row[3]
            )
            bsplines_values.scatter_add_(
                dim=2,
                index=ith_left_knot.unsqueeze(-1),
                src=ith_left_bspline_value.unsqueeze(-1),
            )
        return bsplines_values.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate output values as a linear combination of B-splines(x)"""
        # Normalize and clip input vales to fit in the fixed spline grids' range
        x = torch.clip(self.bn(x), self.grid[self.spline_order], self.grid[-1])
        output = F.linear(  # pylint: disable=E1102
            self._bsplines_values_at(x).view(x.shape[0], -1),
            self.bspline_coeffs.view(self.out_features, -1),
        )
        return output


class BaselineKAN(torch.nn.Module):
    """Kolmogorov-Arnold Network model used as a reference for performance comparison

    This model is trained offline and serves as a baseline for evaluating the
    performance of proposed models trained in an online learning setting.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 400,
        output_dim: int = 200,
    ):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            KANLayer(input_dim, hidden_dim), KANLayer(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
