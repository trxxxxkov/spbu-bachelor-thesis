"""Utility functions to support the training and evaluation process. It includes
helper instruments for managing the train-test loop, calculating performance
metrics, and other related tasks."""

import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# A directory where trained models will be stored
from spbu_bachelor_thesis.global_constants import MODELS_DIR
from spbu_bachelor_thesis.visualization import plot_training_progress
from spbu_bachelor_thesis.datasets import ClassSpecificSampler, FeaturePermutation
from spbu_bachelor_thesis.metrics import (
    CustomMetric,
    MeanPerClassAccuracy,
    OmegaBase,
    OmegaAll,
    OmegaNew,
    _mpc_accuracy_with_filter,
    _get_baseline_accuracy,
)


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
    optimizer: torch.optim.Optimizer,
    save_path: str,
    num_epochs: int = 100,
    early_stopping: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train a model on all classes and plot losses and metric at
    each epoch."""

    model.to(device)
    logs = []
    metric = MeanPerClassAccuracy()
    best_model_weights = None
    # Number of epochs to train without the metric's improvement
    early_stopping_patience = early_stopping
    for epoch_idx in range(num_epochs):
        logs.append(
            _epoch_loop(model, criterion, optimizer, trainloader, testloader, device)
        )
        # Format logs in agreement with plot_training_progress() function:
        # logs[i]["metrics"] has the following structure: [[str, float], ...]
        logs[-1]["metrics"] = [
            ["MeanPerClassAccuracy", metric(logs[-1]["preds"], logs[-1]["targets"])]
        ]
        plot_training_progress(
            logs, title=f"Training progress of the {save_path} over epochs"
        )
        # Check if the metric's value acquired in the last epoch is the best one
        if metric.set_new_best(logs[-1]["metrics"][0][1]):
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_patience = early_stopping
        else:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            print(f"Early stopping is triggered at the epoch {epoch_idx}.")
            break
    print(f"Best metric's value: {metric.best:.5f}.")
    print(f"The model's weights are saved to {os.path.join(MODELS_DIR, save_path)}")
    torch.save(best_model_weights, os.path.join(MODELS_DIR, save_path))
    model.load_state_dict(best_model_weights)


def train_class_incremental(
    model: torch.nn.Module,
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    study_sessions: tuple[torch.Tensor],
    batch_size: int = 256,
    num_epochs: int = 100,
    early_stopping: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train a model in a class incremental learning settings.

    Divide the dataset into sessions (sets of labels to be studied at each step)
    and train model using samples with labels from this session only. During each
    study session, num_epochs is performed with early stopping and a plot is created."""

    model.to(device)
    logs = []
    baseline_name = "kan" if "kan" in save_path else "mlp"
    # Some Omega metrics are normalized by the baseline's "ideal" metric's value
    metrics = [OmegaBase(baseline_name), OmegaAll(baseline_name), OmegaNew()]
    for session_idx, study_session in enumerate(study_sessions):
        # Chooses samples with class labels that are in the current study session
        curr = ClassSpecificSampler(trainset, study_session)
        # Chooses samples with class labels that were in any of the previous
        # study sessions
        prev = ClassSpecificSampler(
            testset, torch.cat(study_sessions[: session_idx + 1])
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, num_workers=2, sampler=curr
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, num_workers=2, sampler=prev
        )
        _train(
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            num_epochs,
            early_stopping,
            study_sessions[session_idx],
            device=device,
        )
        # Update FIM and saved model's weights if EWCLoss is used
        if hasattr(criterion, "update"):
            criterion.update(trainset, device=device)
        logs.append({})
        logs[-1]["test_loss"], preds, targets = _test_loop(
            model, testloader, criterion, device
        )
        # Perform test loop to avoid weights changing
        logs[-1]["train_loss"], _, _ = _test_loop(model, trainloader, criterion, device)
        # logs[i]["metrics"] has the following structure: [[str, float], ...]
        logs[-1]["metrics"] = [
            [
                "OmegaBase",
                metrics[0](preds, targets, allowed_labels=study_sessions[0]),
            ],
            [
                "OmegaAll",
                metrics[1](
                    preds,
                    targets,
                    allowed_labels=torch.cat(study_sessions[: session_idx + 1]),
                ),
            ],
            [
                "OmegaNew",
                metrics[2](preds, targets, allowed_labels=study_session),
            ],
        ]
        plot_training_progress(
            logs,
            title=f"Training progress of the {save_path} over sessions",
            xlabel="Session",
        )
    print(f"The model's weights are saved to {os.path.join(MODELS_DIR, save_path)}")
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, save_path))


def get_study_sessions(
    dataset: torch.utils.data.Dataset,
    base_size: int,
    session_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor]:
    """Separate labels into non-overlapping groups for study sessions in CIL

    Args:
        dataset: a PyTorch dataset to be used in CIL;
        base_size: number of classes in the first study session (usually
            base_size >> session_size);
        session_size: number of classes added in each study session.

    Returns:
        A tuple of 1-D tensors with class indices for i-th study session."""

    labels = torch.tensor([labels for _, labels in dataset], device=device).unique()
    shuffled_labels = labels[torch.randperm(labels.shape[0])]
    # The first study session may include more classes than others
    init_labels = shuffled_labels[:base_size]
    other_labels_groups = torch.split(shuffled_labels[base_size:], session_size)
    return (init_labels,) + other_labels_groups


def train_data_permutation(
    model: torch.nn.Module,
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    num_sessions: int = 4,
    batch_size: int = 256,
    num_epochs: int = 100,
    early_stopping: int = 10,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train a model in data permutation experiment settings.

    Create new dataset's representations by randomly permuting the elements
    of the input feature vectors, with the random permutation changing
    between sessions (permutation is the same for train and test) and train a
    model, calculating its ability to retain multiple representations of the
    dataset. During each study session, num_epochs is performed with early
    stopping and a plot is created."""

    model.to(device)
    logs = []
    # Collect FeaturePermutation() objects (transforms) for Omega metrics that
    # are evaluated using all versions of the dataset seen so far
    permutations = []
    # Some Omega metrics are normalized by the baseline's "ideal" metric's value
    baseline_accuracy = _get_baseline_accuracy("kan" if "kan" in save_path else "mlp")
    # Omega metrics are accumulated here through all sessions. The list is
    # already formatted in agreeement with plot_training_progress()
    metrics = [["OmegaBase", 0], ["OmegaAll", 0], ["OmegaNew", 0]]
    for session_idx in range(num_sessions):
        permuted_indices = torch.randperm(trainset[0][0].shape[0])
        permutations.append(FeaturePermutation(permuted_indices))
        trainset.transform = permutations[-1]
        testset.transform = permutations[-1]
        trainloader = DataLoader(
            trainset, batch_size=batch_size, num_workers=2, shuffle=True
        )
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=2)
        _train(
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            num_epochs,
            early_stopping,
            device=device,
        )
        # Update FIM and saved model's weights if EWCLoss is used
        if hasattr(criterion, "update"):
            criterion.update(trainset, device=device)
        logs.append({})
        logs[-1]["test_loss"], preds, targets = _test_loop(
            model, testloader, criterion, device
        )
        logs[-1]["train_loss"], _, _ = _test_loop(model, trainloader, criterion, device)
        # Omega metrics should not be calculated on the 0-th session, so a slightly
        # changed (no normalization by number of total sessions) value is added
        # directly to the logs to avoid erroneous accumulation in "metrics" variable
        if session_idx == 0:
            mpc = _mpc_accuracy_with_filter(preds, targets)
            logs[-1]["metrics"] = [
                ["OmegaBase", mpc / baseline_accuracy],
                ["OmegaAll", mpc / baseline_accuracy],
                ["OmegaNew", mpc],
            ]
        else:
            a_base_i, a_all_i, a_new_i = _get_permutation_metrics(
                model, testset, permutations, batch_size=batch_size, device=device
            )
            metrics[0][1] += a_base_i / baseline_accuracy
            metrics[1][1] += a_all_i / baseline_accuracy
            metrics[2][1] += a_new_i
            logs[-1]["metrics"] = [
                [metric[0], metric[1] / session_idx] for metric in metrics
            ]
        plot_training_progress(
            logs,
            title=f"Training progress of the {save_path} over sessions",
            xlabel="Session",
        )
    print(f"The model's weights are saved to {os.path.join(MODELS_DIR, save_path)}")
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, save_path))


def _train(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    early_stopping: int = 10,
    allowed_labels: torch.Tensor = None,
    metric: CustomMetric = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train model with early stopping (until the metric's imrovement stops)."""
    model.to(device)
    if metric is None:
        metric = MeanPerClassAccuracy()
    best_model_weights = None
    # Number of epochs to train without the metric's improvement
    early_stopping_patience = early_stopping
    for _ in range(num_epochs):
        logs = _epoch_loop(model, criterion, optimizer, trainloader, testloader, device)
        curr_metric = metric(logs["preds"], logs["targets"], allowed_labels)
        # Check if the metric's value acquired in the last epoch is the best one
        if metric.set_new_best(curr_metric):
            early_stopping_patience = early_stopping
            best_model_weights = copy.deepcopy(model.state_dict())
        # If the network's output is inconsistent, it is given more time to adapt.
        # This helps in CIL, when training one class after another sometimes left
        # NN's weights in the area where no classes are recognizable
        elif curr_metric == 0:
            early_stopping_patience = early_stopping
        else:
            early_stopping_patience -= 1
        if early_stopping_patience == 0:
            break
    model.load_state_dict(best_model_weights)


def _epoch_loop(
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
    evaluation."""
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


def _train_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Perform a training over an entire dataset and return loss averaged over batches"""
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
    """Perform a validation over an entire dataset, return loss averaged over
    batches, targets and predictions"""
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
    return epoch_loss / len(dataloader), torch.cat(preds), torch.cat(targets)


def _get_permutation_metrics(
    model: torch.nn.Module,
    testset: torch.utils.data.Dataset,
    transforms: list[FeaturePermutation],
    batch_size: int = 256,
    num_workers: int = 2,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, float]:
    """Get a_base_i, a_new_i, a_all_i for Omega metrics calculation in a data
    permutation experiment. The returned values should be accumulated over several
    sessions."""
    a_base_i = 0
    a_all_i = 0
    a_new_i = 0
    for session_idx, transform in enumerate(transforms):
        # Apply features permutation
        testset.transform = transform
        dl = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        _, preds, targets = _test_loop(model, dl, device=device)
        mpc = _mpc_accuracy_with_filter(preds, targets)
        if session_idx == 0:
            a_base_i += mpc
        if session_idx == len(transforms) - 1:
            a_new_i += mpc
        a_all_i += mpc / len(transforms)
    return a_base_i, a_all_i, a_new_i
