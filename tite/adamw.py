from typing import Iterable, Tuple

from torch import Tensor
from torch.optim import AdamW


class AdamWNoWeightDecayBiasNorm(AdamW):

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float | Tensor = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None
    ):
        decay_params = []
        nodecay_params = []
        for param in params:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(
            param_groups,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
