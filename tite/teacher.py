from torch import Tensor
from torch.nn import Module


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_input_ids: Tensor, **kwargs) -> Tensor:
        return input_input_ids


class MLMPredictor(Module):
    def __init__(self, padid: int, maskid: int, train_only_masked: bool = True) -> None:
        super().__init__()
        self._pad_id = padid
        self._mask_id = maskid
        self.train_only_masked = train_only_masked

    def forward(self, input_input_ids: Tensor, target_input_ids: Tensor, **kwargs) -> Tensor:
        if self.train_only_masked:
            target_input_ids = target_input_ids.masked_fill(input_input_ids != self._mask_id, -100)
        else:
            targets = target_input_ids.masked_fill(input_input_ids == self._pad_id, -100)
        return targets
