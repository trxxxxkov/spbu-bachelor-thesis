"""Utility functions to support the training and evaluation process. It includes
helper instruments for managing the train-test loop, calculating performance
metrics, and other related tasks."""

import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.global_constants import MODELS_DIR
from src.utils.visualization import plot_training_progress
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


class MeanPerClassAccuracy(CustomMetric):
    """A metric: classification accuracy averaged over all classes"""

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return _mpc_accuracy_with_filter(preds, targets)


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

    def __init__(self, study_sessions: tuple[torch.Tensor], baseline_name: str):
        super().__init__()
        self.study_sessions = study_sessions
        self.i = 1
        self.a_base = 0
        self.a_ideal = _get_baseline_accuracy(baseline_name)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        self.i += 1
        self.a_base += _mpc_accuracy_with_filter(preds, targets, self.study_sessions[0])
        return self.a_base / (self.a_ideal * (self.i - 1))


class OmegaNew:
    """A metric: measures the model's ability to immediately recall new tasks.

        Omega_new = 1/(T-1) * sum_{i=2}^T (a_{new,i}),

    where
    T - total number of study sessions;
    i - index of the current session;
    a_{new,i} - MPC accuracy for session i immediately after it is learned."""

    def __init__(self, study_sessions: tuple[torch.Tensor]):
        super().__init__()
        self.study_sessions = study_sessions
        self.i = 1
        self.a_new = 0

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        self.i += 1
        self.a_new += _mpc_accuracy_with_filter(
            preds, targets, self.study_sessions[self.i - 1]
        )
        return self.a_new / (self.i - 1)


class OmegaAll:
    """A metric: Measures how well a model both retains prior knowledge and
    acquires new information.

        Omega_all = 1/(T-1) * sum_{i=2}^T (a_{all,i} / a_{ideal}),

    where
    T - total number of study sessions;
    i - index of the current session;
    a_{all,i} - MPC accuracy of all of the test data for the classes seen to
    this point,
    a_{ideal} - offline KAN/MLP MPC accuracy."""

    def __init__(self, study_sessions: tuple[torch.Tensor], baseline_name: str):
        super().__init__()
        self.study_sessions = study_sessions
        self.i = 1
        self.a_all = 0
        self.a_ideal = _get_baseline_accuracy(baseline_name)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        self.i += 1
        self.a_all += _mpc_accuracy_with_filter(
            preds, targets, self.study_sessions[: self.i]
        )
        return self.a_all / (self.a_ideal * (self.i - 1))


def measure_forward_backward_time(
    model_cls: torch.nn.Module,
    nruns: int = 100,
    batch_size: int = 1024,
    in_features: int = 1024,
    out_features: int = None,
    device: torch.device = torch.device("cpu"),
):
    """Measures the average time for forward and backward passes of a given model.

    Returns:
        tuple:
            forward_time (float):
                The average time (in seconds) for the forward pass.
            backward_time (float):
                The average time (in seconds) for the backward pass.
    """
    nwarmup_runs = nruns // 10
    if out_features is not None:
        model = model_cls(in_features, out_features).to(device)
    else:
        model = model_cls(in_features).to(device)
    inputs = torch.randn(batch_size, in_features).to(device)
    # Forward pass
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(nwarmup_runs):
            _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Evaluation
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(nruns), desc="Forward pass"):
            _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start_time) / nruns
    # Forward + Backward pass
    model.train()
    outputs = model(inputs)
    grad_output = torch.ones_like(outputs)
    # Warmup
    for _ in range(nwarmup_runs):
        outputs = model(inputs)
        model.zero_grad()
        outputs.backward(grad_output, retain_graph=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Evaluation
    start_time = time.perf_counter()
    for _ in tqdm(range(nruns), desc="Backward pass"):
        outputs = model(inputs)
        model.zero_grad()
        outputs.backward(grad_output, retain_graph=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) / nruns
    backward_time = total_time - forward_time
    return forward_time, backward_time


def train_offline(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metrics: list,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    num_epochs: int = 100,
    early_stopping: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train a model on all classes and plot losses and metric at
    each epoch. The first metric in the metrics list is used for early stopping.
    """
    model.to(device)
    train_loss_history, test_loss_history = [], []
    test_metrics_history = [list() for metric in metrics]
    best_model_weights = None
    early_stopping_patience = early_stopping
    for epoch_idx in range(num_epochs):
        epoch_logs = epoch_loop(
            model, criterion, optimizer, trainloader, testloader, device
        )
        train_loss_history.append(epoch_logs["train_loss"])
        test_loss_history.append(epoch_logs["test_loss"])
        for metric_idx, metric in enumerate(metrics):
            test_metrics_history[metric_idx].append(
                metric(torch.cat(epoch_logs["preds"]), torch.cat(epoch_logs["targets"]))
            )
        plot_training_progress(
            train_loss_history,
            test_loss_history,
            test_metrics_history,
            title=f"Training progress of the {save_path} over epochs",
        )
        if metrics[0].set_new_best(test_metrics_history[0][-1]):
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_patience = early_stopping
        else:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            print(f"Early stopping is triggered at the epoch {epoch_idx}.")
            break
    print(f"Best metric's value: {metrics[0].best:.5f}.")
    print(f"The model's weights are saved to {os.path.join(MODELS_DIR, save_path)}")
    torch.save(best_model_weights, os.path.join(MODELS_DIR, save_path))
    model.load_state_dict(best_model_weights)


def epoch_loop(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader = None,
    testloader: torch.utils.data.DataLoader = None,
    device: torch.device = torch.device("cpu"),
    train: bool = True,
    test: bool = True,
) -> dict:
    """Perform one epoch, optionally including the training phase and/or the
    testing phase. Return (if available) loss, predictions and targets for metrics
    evaluation.

    Returns:
        logs: a dictionary with data collected during the epoch. Following
            keys are available:
            train_loss: float = model's loss collected during the training stage;
            test_loss: float = model's loss collected during the test stage;
            preds: torch.Tensor = 1-D tensor with model's prediction on the test
                dataset;
            targets: torch.Tensor = 1-D tensor with true labels of the test
                dataset;
    """
    logs = {}
    if train:
        logs["train_loss"] = _train_loop(
            model, trainloader, criterion, optimizer, device
        )
    if test:
        logs["test_loss"], logs["preds"], logs["targets"] = _test_loop(
            model, testloader, criterion, device
        )
    return logs


def get_study_sessions(
    dataset: torch.utils.data.Dataset, base_size: int, session_size: int = 1
) -> tuple[torch.Tensor]:
    """Separate labels into non-overlapping groups for study sessions in CIL

    Args:
        dataset: a PyTorch dataset to be used in CIL;
        base_size: number of classes in the first study session (usually
            base_size >> session_size);
        session_size: number of classes added in each study session.

    Returns:
        A tuple of 1-D tensors with class indices for i-th study session."""

    labels = torch.tensor([labels for _, labels in dataset]).unique()
    shuffled_labels = labels[torch.randperm(labels.shape[0])]
    # The first study session may include more classes than others
    init_labels = shuffled_labels[:base_size]
    shuffled_labels = torch.split(shuffled_labels[base_size:], session_size)
    return (init_labels,) + shuffled_labels


def _train_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Optimize model on a train dataset and return loss averaged over batches"""
    model.train()
    epoch_loss = 0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        batch_outputs = model(batch_inputs)
        batch_loss = criterion(batch_outputs, batch_targets)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    return epoch_loss / len(dataloader)


def _test_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, list[torch.Tensor], list[torch.Tensor]]:
    """Get model's predictions on a test dataset and report logs"""
    model.eval()
    epoch_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_outputs = model(batch_inputs)
            if criterion is not None:
                batch_loss = criterion(batch_outputs, batch_targets)
                epoch_loss += batch_loss.item()
            preds.append(batch_outputs)
            targets.append(batch_targets)
    return epoch_loss / len(dataloader), preds, targets


def _get_baseline_accuracy(baseline_name: str, dataset_name="cub200_test_embed.pt"):
    """Download weights for a baseline and run a test loop to calculate accuracy"""
    # baseline_name - "kan" | "mlp"
    if baseline_name == "kan":
        model = BaselineKAN()
    else:
        model = BaselineMLP()
    state_dict = torch.load(os.path.join(MODELS_DIR, f"baseline_{baseline_name}.pth"))
    model.load_state_dict(state_dict)
    testset = EmbeddingDataset(dataset_name)
    testloader = DataLoader(testset, batch_size=128, num_workers=2)
    _, preds, targets = _test_loop(model, testloader)
    metric = MeanPerClassAccuracy()
    return metric(torch.cat(preds), torch.cat(targets))


def _mpc_accuracy_with_filter(
    preds: torch.Tensor, targets: torch.Tensor, valid_labels: torch.Tensor = None
) -> float:
    """Compute mean per class accuracy metric, but only considering labels from valid_labels"""
    preds = preds.argmax(dim=1)
    class_total = torch.zeros(targets.max() + 1)
    class_correct = torch.zeros(targets.max() + 1)
    for i, label in enumerate(targets):
        if label in valid_labels or valid_labels is None:
            class_total[label] += 1
            class_correct[label] += (targets[i] == preds[i]).item()
        # +1e-8 for numerical stability in case if class_total == 0
    if valid_labels is None:
        return (class_correct / (class_total + 1e-8)).mean().item()
    return (class_correct / (class_total + 1e-8)).sum().item() / len(valid_labels)
