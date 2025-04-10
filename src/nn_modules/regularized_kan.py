"""Implementation of a custom loss function (based on EWC approach) which is a part
of regularized KAN"""

import torch
from torch.utils.data import DataLoader


class EWCLoss(torch.nn.Module):
    """A loss function implementing Elastic Weight Consolidation (EWC) approach
    using diagonal Fisher Information Matrix (FIM) to approximate posterior
    distribution for previous tasks.

    Key components:
    1. Fisher Information Matrix: Stored in self.fisher and updated with
        information from each new study session using exponential moving
        average (decay factor stored in self.decay).
    2. Previous model parameters: Stored in self.saved_params and overwritten
        during each study session (Online EWC variant).

    Regularization is not applied during the first study session.

    To update FIM and save current model parameters, call the update() method
    after each study session."""

    def __init__(
        self,
        base_loss: torch.nn.Module,
        model: torch.nn.Module,
        lambda_: float = 0.1,
        decay: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.lambda_ = lambda_
        self.decay = decay
        self.device = device
        # Diagonal Fisher Information Matrix
        self.fisher: dict = None
        # Model parameters from the previous study session
        self.saved_params: dict = None

    def update(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        num_batches: int = 200,
        empirical_estimate: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Determines current model's weights importance by updating Fisher
        Information Matrix and saving current model's parameters.

        Args:
            dataset: Training data for FIM estimation
            batch_size: Batch size for FIM calculation
            num_batches: Number of batches to process
            empirical_estimate: Use model predictions instead of true labels
            device: Computation device
        """
        self.model.to(device)
        self.model.eval()
        # FIM is updated using exponential moving average:
        # [current FIM] = self.decay * [old FIM] + (1-self.decay) * [new FIM]
        new_fisher = {}
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            # Only a part of the dataset is processed for optimization reasons
            if batch_idx == num_batches:
                break
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_outputs = self.model(batch_inputs)
            # See https://arxiv.org/pdf/1905.12558 for more information on the
            # choice between FIM and empirical FIM
            labels = (
                batch_outputs.argmax(dim=1) if empirical_estimate else batch_targets
            )
            self.model.zero_grad()
            batch_loss = self.base_loss(batch_outputs, labels)
            batch_loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_squared = param.grad.detach().clone().pow(2)
                    if name in new_fisher:
                        new_fisher[name] += grad_squared
                    else:
                        new_fisher[name] = grad_squared
        # Average and update a FIM
        for name in new_fisher:
            new_fisher[name] /= num_batches
        if self.fisher:
            # EMA if there is a FIM calculated during previous tasks
            for name, new_val in new_fisher.items():
                old_val = self.fisher.get(name, torch.zeros_like(new_val))
                self.fisher[name] = self.decay * old_val + (1 - self.decay) * new_val
        else:
            # Just a current FIM if it's the first study session
            self.fisher = new_fisher
        # The is only one set of model's parameters which is overwritten at each
        # study session in contrast with the original EWC approach
        self.saved_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        self.model.train()

    def forward(self, preds, targets, *args, **kwargs):
        """Computes total loss: base_loss + lambda/2 * EWC regularization."""
        base_loss = self.base_loss(preds, targets, *args, **kwargs)
        ewc_loss = torch.tensor(0, dtype=float, device=preds.device)
        # Regularization is not applied during the first study session
        if self.fisher:
            for name, param in self.model.named_parameters():
                if param.requires_grad and (name in self.fisher):
                    ewc_loss += (
                        self.fisher[name] * (param - self.saved_params[name]).pow(2)
                    ).sum()
        return base_loss + (self.lambda_ / 2) * ewc_loss
