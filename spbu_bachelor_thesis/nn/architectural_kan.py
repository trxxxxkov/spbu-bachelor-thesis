import torch
import torch.nn as nn
import torch.nn.functional as F


class FELLayer(nn.Module):
    """
    Fixed Expansion Layer (FEL) — модульный разреженный слой для PyTorch.
    Параметры
    ----------
    in_features  : int  — размер входа
    out_features : int  — размер расширенного пространства (>> in_features)
    k_pos        : int  — сколько нейронов с макс. активацией оставить
    k_neg        : int  — сколько нейронов с мин. активацией оставить
    excitatory_ratio : float  — доля возбуждающих связей для каждого FEL-нейрона
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_pos: int = 25,
        k_neg: int = 25,
        excitatory_ratio: float = 0.5,
        weight_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_pos = k_pos
        self.k_neg = k_neg

        # --- разрежённая фиксированная матрица W ---
        #   +1 / -1 с одинаковым числом входов на нейрон
        conn_per_neuron = max(1, int(excitatory_ratio * in_features))
        mask = torch.zeros(out_features, in_features, dtype=torch.float32)
        for j in range(out_features):
            idx = torch.randperm(in_features)[:conn_per_neuron]
            half = conn_per_neuron // 2
            mask[j, idx[:half]] = weight_scale  # возбуждающие
            mask[j, idx[half:]] = -weight_scale  # ингибирующие
        self.register_buffer("W_fixed", mask)  # градиент не требуется

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  (batch, in_features)
        out: (batch, out_features) — триггерованная разрежённая активация
        """
        # линейное преобразование (нет обучения весов)
        z = F.linear(x, self.W_fixed)  # (B, out_features) # pylint: disable=E1102

        # --- триггерование: top-K+, bottom-K- ---
        with torch.no_grad():
            # индексы нейронов с макс/мин активациями
            topk_pos = torch.topk(z, self.k_pos, dim=1, largest=True).indices
            topk_neg = torch.topk(z, self.k_neg, dim=1, largest=False).indices

            trigger_mask = torch.zeros_like(z, dtype=torch.bool)
            trigger_mask.scatter_(1, topk_pos, True)
            trigger_mask.scatter_(1, topk_neg, True)

        # фиксированные значения для активных узлов;
        # detach() предотвращает градиент через константы
        pos_val = z.max(dim=1, keepdim=True).values.detach()
        neg_val = z.min(dim=1, keepdim=True).values.detach()

        out = torch.zeros_like(z)
        out[trigger_mask & (z >= 0)] = pos_val.expand(-1, self.out_features)[
            trigger_mask & (z >= 0)
        ]
        out[trigger_mask & (z < 0)] = neg_val.expand(-1, self.out_features)[
            trigger_mask & (z < 0)
        ]

        # в остальном — нули, градиент течёт через out к предыдущим слоям
        return out
