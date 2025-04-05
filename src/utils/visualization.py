"""Custom functions for building plots and other visualisations-related instruments"""

import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_training_progress(
    train_loss_history: list[float],
    test_loss_history: list[float],
    test_metrics_history: list[list[float]],
    title: str = "Train, test losses and metric values over epochs",
) -> None:
    """Visualizes training progress by plotting loss curves and metrics in real-time

    Dynamically updates a dual-axis plot showing:
    - Training loss (main axis)
    - Test loss (main axis)
    - Test metric (secondary axis)
    Designed for real-time monitoring during model training."""

    clear_output(wait=True)
    plt.figure(figsize=(13, 5))
    # Main axis for the train, test losses
    ax1 = plt.gca()
    ax1.plot(
        train_loss_history,
        label="Training loss",
        color="tab:blue",
        linestyle="--",
    )
    ax1.plot(test_loss_history, color="tab:blue", label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    # Force integer ticks for all epochs
    ax1.set_xticks(range(len(train_loss_history)))
    ax1.set_xticklabels(
        range(len(train_loss_history)),
        rotation=60,
        fontsize=9,
        ha="center",
    )
    ax1.grid(True, axis="x", color="black", linestyle="-", alpha=0.3)
    # Secondary axis (on the right side) for test metric
    ax2 = ax1.twinx()
    for metric_idx, metric_history in enumerate(test_metrics_history):
        ax2.plot(
            metric_history,
            color=(0, (metric_idx + 1) / (len(test_metrics_history)), 0),
            label=f"Test metric #{metric_idx}",
        )
    ax2.set_ylabel("Metrics")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    # Combine legends of the left and right axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.title(title)
    plt.show()
