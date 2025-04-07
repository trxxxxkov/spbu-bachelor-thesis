"""Custom functions for building plots and other visualisations-related instruments"""

import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_training_progress(
    logs: list[dict],
    title: str = "Train, test losses and metric values over epochs",
    xlabel: str = "Epoch",
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
        [epoch["train_loss"] for epoch in logs],
        label="Training loss",
        color="tab:blue",
        linestyle="--",
    )
    ax1.plot(
        [epoch["test_loss"] for epoch in logs], color="tab:blue", label="Test loss"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    # Determine xticks positions based on the number of epochs
    if len(logs) > 11:
        xticks_positions = list(range(0, len(logs), 5))
        last_epoch = len(logs) - 1
        if last_epoch not in xticks_positions:
            xticks_positions.append(last_epoch)
    else:
        xticks_positions = list(range(len(logs)))
    ax1.set_xticks(xticks_positions)
    ax1.set_xticklabels(
        [str(pos) for pos in xticks_positions],
        rotation=45,
        ha="center",
    )
    ax1.grid(True, axis="x", color="black", linestyle="-", alpha=0.3)
    # Secondary axis (on the right side) for test metric
    ax2 = ax1.twinx()
    for i in range(len(logs[-1]["metrics"])):
        ax2.plot(
            [epoch["metrics"][i][1] for epoch in logs],
            color=(0, (i + 1) / (len(logs[-1]["metrics"])), 0),
            label=logs[-1]["metrics"][i][0],
        )
    ax2.set_ylabel("Metrics")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    # Combine legends of the left and right axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.title(title)
    plt.show()
