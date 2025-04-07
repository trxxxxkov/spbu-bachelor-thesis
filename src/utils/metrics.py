"""Custom metrics implementations and related private utility functions."""

import os

import torch
from torch.utils.data import DataLoader

from src.utils.global_constants import MODELS_DIR
from src.utils.datasets import EmbeddingDataset
from src.nn_modules.kan import BaselineKAN
from src.nn_modules.mlp import BaselineMLP


class CustomMetric:
    """A base class with essential functionality for a metric to be used in other
    utility functions."""

    def __init__(self):
        self.to_be_maximized = True
        self.min_increase = 1e-3
        self.best = -float("inf")

    def set_new_best(self, new_best: float) -> bool:
        """Check if a new metric's value is better than the stored one and return
        True if the stored value was updated."""
        sign = 1 if self.to_be_maximized else -1
        if (new_best - self.best) * sign > self.min_increase:
            self.best = new_best
            return True
        return False

    def reset(self) -> None:
        """Set attributes to initial value"""
        self.best = (-1 if self.to_be_maximized else 1) * float("inf")


class MeanPerClassAccuracy(CustomMetric):
    """A metric: classification accuracy averaged over selected classes"""

    def __call__(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        allowed_labels: torch.Tensor = None,
    ) -> float:
        return _mpc_accuracy_with_filter(preds, targets, allowed_labels)


class OmegaBase(CustomMetric):
    """A metric: measures the model's retention of the first session, after
    learning in later study sessions.

        Omega_base = 1/(T-1) * sum_{i=2}^T (a_{base,i} / a_{ideal}),

    where
    T - total number of study sessions;
    i - index of the current session;
    a_{base,i} - MPC accuracy on the first session (base set) after i new
    sessions have been learned,
    a_{ideal} - offline KAN/MLP MPC accuracy."""

    def __init__(self, baseline_name: str):
        super().__init__()
        self.i = 0
        self.a_base = 0
        self.a_ideal = _get_baseline_accuracy(baseline_name)

    def __call__(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        allowed_labels: torch.Tensor = None,
    ) -> float:
        self.i += 1
        # Treat metric evaluation on a base set separately, because it shouldn't
        # be included in the total number of studied sessions.
        if self.i == 1:
            return (
                _mpc_accuracy_with_filter(preds, targets, allowed_labels) / self.a_ideal
            )
        else:
            self.a_base += _mpc_accuracy_with_filter(preds, targets, allowed_labels)
            return self.a_base / (self.a_ideal * (self.i - 1))

    def reset(self) -> None:
        """Set attributes to initial value"""
        self.best = (-1 if self.to_be_maximized else 1) * float("inf")
        self.i = 0
        self.a_base = 0


class OmegaNew(CustomMetric):
    """A metric: measures the model's ability to immediately recall new tasks.

        Omega_new = 1/(T-1) * sum_{i=2}^T (a_{new,i}),

    where
    T - total number of study sessions;
    i - index of the current session;
    a_{new,i} - MPC accuracy for session i immediately after it is learned."""

    def __init__(self):
        super().__init__()
        self.i = 0
        self.a_new = 0

    def __call__(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        allowed_labels: torch.Tensor = None,
    ) -> float:
        self.i += 1
        # Treat metric evaluation on a base set separately, because it shouldn't
        # be included in the total number of studied sessions.
        if self.i == 1:
            return _mpc_accuracy_with_filter(preds, targets, allowed_labels)
        else:
            self.a_new += _mpc_accuracy_with_filter(preds, targets, allowed_labels)
            return self.a_new / (self.i - 1)

    def reset(self) -> None:
        """Set attributes to initial values"""
        self.best = (-1 if self.to_be_maximized else 1) * float("inf")
        self.i = 0
        self.a_new = 0


class OmegaAll(CustomMetric):
    """A metric: Measures how well a model both retains prior knowledge and
    acquires new information.

        Omega_all = 1/(T-1) * sum_{i=2}^T (a_{all,i} / a_{ideal}),

    where
    T - total number of study sessions;
    i - index of the current session;
    a_{all,i} - MPC accuracy of all of the test data for the classes seen to
    this point,
    a_{ideal} - offline KAN/MLP MPC accuracy."""

    def __init__(self, baseline_name: str):
        super().__init__()
        self.i = 0
        self.a_all = 0
        self.a_ideal = _get_baseline_accuracy(baseline_name)

    def __call__(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        allowed_labels: torch.Tensor = None,
    ) -> float:
        self.i += 1
        # Treat metric evaluation on a base set separately, because it shouldn't
        # be included in the total number of studied sessions.
        if self.i == 1:
            return (
                _mpc_accuracy_with_filter(preds, targets, allowed_labels) / self.a_ideal
            )
        else:
            self.a_all += _mpc_accuracy_with_filter(preds, targets, allowed_labels)
            return self.a_all / (self.a_ideal * (self.i - 1))

    def reset(self) -> None:
        """Set attributes to initial value"""
        self.best = (-1 if self.to_be_maximized else 1) * float("inf")
        self.i = 0
        self.a_all = 0


def _get_baseline_accuracy(baseline_name: str, dataset_name="cub200_test_embed.pt"):
    """Download weights for a baseline and run a test loop to calculate accuracy"""
    # baseline_name - "kan" | "mlp"
    if baseline_name == "kan":
        model = BaselineKAN()
    else:
        model = BaselineMLP()
    state_dict = torch.load(os.path.join(MODELS_DIR, f"offline_{baseline_name}.pth"))
    model.load_state_dict(state_dict)
    model.eval()
    testset = EmbeddingDataset(dataset_name)
    testloader = DataLoader(testset, batch_size=128, num_workers=2)
    preds, targets = [], []
    with torch.no_grad():
        for batch_inputs, batch_targets in testloader:
            batch_outputs = model(batch_inputs)
            preds.append(batch_outputs)
            targets.append(batch_targets)
    return _mpc_accuracy_with_filter(torch.cat(preds), torch.cat(targets))


def _mpc_accuracy_with_filter(
    preds: torch.Tensor, targets: torch.Tensor, allowed_labels: torch.Tensor = None
) -> float:
    """Compute mean per class accuracy metric, but only considering labels from allowed_labels"""
    preds = preds.argmax(dim=1)
    class_total = torch.zeros(targets.max() + 1)
    class_correct = torch.zeros(targets.max() + 1)
    for i, label in enumerate(targets):
        if allowed_labels is None or label in allowed_labels:
            class_total[label] += 1
            class_correct[label] += (targets[i] == preds[i]).item()
        # +1e-8 for numerical stability in case if class_total == 0
    if allowed_labels is None:
        return (class_correct / (class_total + 1e-8)).mean().item()
    return (class_correct / (class_total + 1e-8)).sum().item() / len(allowed_labels)
