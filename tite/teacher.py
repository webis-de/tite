import torch
from torch import Tensor
from torch.nn import Module


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        return input_ids


class MLMTeacher(Module):
    def __init__(self, padid: int) -> None:
        super().__init__()
        self._pad_id = padid

    def forward(self, input_ids: Tensor, mlm_mask: Tensor, **kwargs) -> Tensor:
        targets = torch.where(input_ids.eq(self._pad_id) | ~mlm_mask, -100, input_ids)
        return targets
