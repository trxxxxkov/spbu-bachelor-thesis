"""Utility functions to support the training and evaluation process. It includes
helper instruments for managing the train-test loop, calculating performance
metrics, and other related tasks."""

import copy
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.utils.global_constants import MODELS_DIR
from src.utils.visualization import plot_training_progress


def measure_forward_backward_time(
    model_cls: nn.Module,
    nruns: int = 100,
    batch_size: int = 1024,
    in_features: int = 1024,
    out_features: int = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Measures the average time for forward and backward passes of a given model.

    Args:
        model_cls:
            The class of the model to benchmark. The model should accept
            `in_features` and `out_features` as positional arguments.
        nruns:
            The number of runs to average the timing over.
        batch_size:
            The batch size of the input data.
        in_features:
            The number of input features for the model.
        out_features:
            The number of output features for the model.
        warmup:
            The number of warmup iterations to perform before timing. Default is 50.
        device:
            The device to run the model on ("cpu" or "cuda"). Default is "cpu".

    Returns:
        tuple:
            forward_time (float):
                The average time (in seconds) for the forward pass.
            backward_time (float):
                The average time (in seconds) for the backward pass.
    """
    nwarmup_runs = nruns // 5
    if out_features is not None:
        model = model_cls(in_features, out_features).to(device)
    else:
        model = model_cls(in_features).to(device)
    inputs = torch.randn(batch_size, in_features).to(device)
    # Forward pass
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in tqdm(range(nwarmup_runs), desc="Warmup", leave=False):
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
    for _ in tqdm(range(nwarmup_runs), desc="Warmup", leave=False):
        outputs = model(inputs)
        model.zero_grad()
        outputs.backward(grad_output, retain_graph=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Evaluation
    start_time = time.perf_counter()
    for _ in tqdm(range(nruns), desc="Forward+Backward pass"):
        outputs = model(inputs)
        model.zero_grad()
        outputs.backward(grad_output, retain_graph=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_time) / nruns
    backward_time = total_time - forward_time
    return forward_time, backward_time


def offline_train(
    model: nn.Module,
    save_path: str,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    metric,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    early_stopping: int = 10,
    metric_delta: float = 0.001,
    metric_mode: str = "max",
    device: str = "cpu",
) -> None:
    """
    Trains a neural network model with early stopping and performance monitoring.

    Args:
        model:
            Neural network module to train
        save_path:
            Filename/path for saving best model weights
        trainloader:
            Training data loader (batch generator)
        testloader:
            Validation/test data loader
        criterion:
            Loss function module
        metric:
            Evaluation metric callable (e.g., accuracy function)
        optimizer:
            Parameter optimizer instance
        num_epochs:
            Maximum training epochs
        early_stopping:
            Early stopping patience (epochs without improvement)
        metric_delta:
            Minimum improvement threshold for metric
        metric_mode:
            Optimization direction - 'max' or 'min'
        device:
            Compute device ('cpu' or 'cuda')
    """
    model.to(device)
    train_loss_history = []
    test_loss_history = []
    test_metric_history = []
    best_metric = -float("inf") if metric_mode == "max" else float("inf")
    best_model_weights = None
    early_stopping_patience = early_stopping

    for epoch_idx in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_inputs, batch_targets in trainloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
        train_epoch_loss = running_loss / len(trainloader)
        train_loss_history.append(train_epoch_loss)
        test_epoch_loss, test_epoch_metric = offline_test(
            model, testloader, criterion, metric, device
        )
        test_loss_history.append(test_epoch_loss)
        test_metric_history.append(test_epoch_metric)
        plot_training_progress(
            train_loss_history,
            test_loss_history,
            test_metric_history,
            title=f"Training progress of the {save_path} over epochs",
        )
        if (
            metric_mode == "max" and test_epoch_metric > best_metric + metric_delta
        ) or (metric_mode == "min" and test_epoch_metric < best_metric - metric_delta):
            best_metric = test_epoch_metric
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_patience = early_stopping
        else:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            print(f"Early stopping is triggered at the epoch {epoch_idx}.")
            break
    torch.save(best_model_weights, os.path.join(MODELS_DIR, save_path))
    print(f"Best metric's value: {best_metric:.5f}.")
    print(f"The model's weights are saved to {os.path.join(MODELS_DIR, save_path)}")
    model.load_state_dict(best_model_weights)


def offline_test(
    model: nn.Module, testloader: DataLoader, criterion: nn.Module, metric, device="cpu"
) -> tuple[float, float]:
    """Evaluates a model on test/validation data and computes performance metrics.

    Args:
        testloader:
            Data loader for test/validation data
        model:
            Trained model to evaluate
        criterion:
            Loss function for loss calculation
        metric:
            Evaluation metric callable (outputs, targets) -> float
        device:
            Compute device ('cpu' or 'cuda')

    Returns:
        tuple: (test_loss, test_metric) where
            test_loss: Average loss across all test batches
            test_metric: Computed metric on full test set
    """
    model.eval()
    outputs = []
    targets = []
    running_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in testloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_outputs = model(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_targets)
            running_loss += batch_loss.item()
            outputs.append(batch_outputs)
            targets.append(batch_targets)
    test_epoch_metric = metric(torch.cat(outputs, dim=0), torch.cat(targets, dim=0))
    test_epoch_loss = running_loss / len(testloader)
    return test_epoch_loss, test_epoch_metric


def mean_per_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy for each class independently, then average across all classes"""
    preds = outputs.argmax(dim=1)
    class_total = torch.zeros(targets.max() + 1)
    class_correct = torch.zeros(targets.max() + 1)
    for i, label in enumerate(targets):
        class_total[label] += 1
        class_correct[label] += (targets[i] == preds[i]).item()
    return (class_correct / (class_total + 1e-8)).mean().item()
