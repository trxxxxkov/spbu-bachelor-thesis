""""""

import torch
from torch.utils.data import DataLoader


class EWCLoss(torch.nn.Module):

    def __init__(
        self,
        base_loss: torch.nn.Module,
        model: torch.nn.Module,
        lambda_: float = 0.1,
        decay: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.base_loss = base_loss
        self.model = model
        self.lambda_ = lambda_
        self.decay = decay
        self.device = device
        self.fisher: dict = None
        self.saved_params: dict = None

    def update(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        num_batches: int = 100,
        empirical_estimate: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.model.to(device)
        self.model.eval()
        new_fisher = {}
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_outputs = self.model(batch_inputs)
            labels = (
                batch_outputs.argmax(dim=1) if empirical_estimate else batch_targets
            )
            self.model.zero_grad()
            batch_loss = self.base_loss(batch_outputs, labels)
            batch_loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name in new_fisher:
                        new_fisher[name] += param.grad.detach().clone().pow(2)
                    else:
                        new_fisher[name] = param.grad.detach().clone().pow(2)
            if batch_idx + 1 == num_batches:
                break

        for name in new_fisher:
            new_fisher[name] /= num_batches
        if self.fisher:
            for name, new_val in new_fisher.items():
                old_val = self.fisher.get(name, torch.zeros_like(new_val))
                self.fisher[name] = self.decay * old_val + (1 - self.decay) * new_val
        else:
            self.fisher = new_fisher
        self.saved_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        self.model.train()

    def forward(self, preds, targets, *args, **kwargs):
        base_loss_val = self.base_loss(preds, targets, *args, **kwargs)
        ewc_loss = torch.tensor(0, dtype=float, device=preds.device)
        if self.fisher:
            for name, param in self.model.named_parameters():
                if param.requires_grad and (name in self.fisher):
                    ewc_loss += (
                        self.fisher[name] * (param - self.saved_params[name]).pow(2)
                    ).sum()
        return base_loss_val + (self.lambda_ / 2) * ewc_loss
