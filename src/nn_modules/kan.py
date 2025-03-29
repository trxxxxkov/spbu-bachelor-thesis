"""Implementation of a KAN layer as a custom PyTorch class"""

import math

import torch
import torch.nn.functional as F


class KANLayer(torch.nn.Module):
    """KAN layer with activation functions parametrized as a linear
    combination of cubic B-splines."""

    spline_order = 3

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_step: int = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_step = grid_step
        self.bn = torch.nn.BatchNorm1d(in_features)
        # Coefficients for B-spline basis functions
        knots_num = math.ceil(1 / grid_step) + 1 + self.spline_order
        self.bspline_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, knots_num)
        )
        # Each coefficient corresponds to a scaling factor for the activation function (spline)
        # on the connection between an input and an output node.
        # Note: activation functions are parameterized as a linear combinations of
        # B-splines with self.bspline_coeffs serving as the weights for this combination.
        self.spline_coeffs = torch.nn.Parameter(torch.Tensor(out_features, in_features))
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
        """Initialize layer parameters with tailored strategies for each
        coefficient type.

        Perform several distinct initialization procedures:
            1. Spline scaling coefficients (self.spline_coeffs):
                Initialized using Kaiming uniform distribution with fan-in mode.
            2. B-spline basis coefficients (self.bspline_coeffs):
                Initialized by approximating uniform noise: a least squares problem
                is solved for Ax = B, where
                x - B-spline basis values computed at grid knots,
                A - optimal basis coefficients, that approximate reference values,
                B - reference values sampled from scaled ~U(-0.5, 0.5).
        """
        with torch.no_grad():
            # Initialise spline scaling coefficients.
            torch.nn.init.kaiming_uniform_(self.spline_coeffs, a=3)
            # Get inner grid knots as they affected by the same B-splines that
            # influence input values inside the grid range.
            interpolation_nodes = self.grid.expand(self.in_features, -1).T[
                self.spline_order :
            ]
            # B-splines values for the chosen points. This is "x" in the system of
            # linear equations mentioned in the method's docstring.
            current_values = self._bsplines_values_at(interpolation_nodes).transpose(
                0, 1
            )
            # Spline values to be interpolated by adjusting coefficients of
            # B-spline basis functions with current_values. This is "B" in the
            # Ax = B system of linear equations.
            reference_values = (
                torch.randn(
                    self.in_features,
                    len(self.grid) - self.spline_order,
                    self.out_features,
                )
            ) / (10 * (len(self.grid) - self.spline_order))
            # Optimal coefficients of B-spline basis functions that interpolate the reference
            # values
            coefficients = torch.linalg.lstsq(  # pylint: disable=E1102
                current_values, reference_values
            ).solution
            self.bspline_coeffs.data.copy_(coefficients.permute(2, 0, 1))

    def _bsplines_values_at(self, x: torch.Tensor):
        """Compute values of all cubic B-spline basis functions at input points x.

        Normalizes x to grid scale, finds spline_order+1 influencing left knots
        per input, and evaluates normalized position in each knot's basis polynomial.

            Args:
                x: Input tensor with values in [0,1] range

            Returns:
                bsplines_values: Tensor of shape [batch_size, in_features, len(grid))
                    where each element contains the evaluated k-th left B-spline basis
                    function's contribution at x (for k in range(0, spline_order+1))
        """
        s = (x - self.grid[0]) / self.grid_step
        closest_left_knot = torch.floor(s).long()
        s_frac = s - closest_left_knot
        bsplines_values = torch.zeros((*x.shape, len(self.grid)), device=x.device)
        for i in range(4):
            # Retrieve coefficients for the explicit formula of B-spline for the
            # current interval
            coeff_row = self.cubic_bspline_formula[3 - i]
            # A spline knot to evaluate x at
            ith_left_knot = closest_left_knot - 3 + i
            # An argument to for the corresponding explicit formula
            ith_left_bspline_arg = (3 - i) + s_frac
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
        """Calculate output values as a linear combination of B-splines(x) and
        scaled coefficients of the corresponding B-splines"""
        # Normalize and clip input vales to fit in the fixed spline grids' range
        x = torch.clip(self.bn(x), self.grid[self.spline_order], self.grid[-1])
        scaled_bspline_coeffs = self.bspline_coeffs * self.spline_coeffs.unsqueeze(-1)
        output = F.linear(  # pylint: disable=E1102
            self._bsplines_values_at(x).view(x.shape[0], -1),
            scaled_bspline_coeffs.view(self.out_features, -1),
        )
        return output
