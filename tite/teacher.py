from torch import Tensor
from torch.nn import Module


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, student_input_ids: Tensor, **kwargs) -> Tensor:
        return student_input_ids


class MLMPredictor(Module):
    def __init__(self, padid: int, maskid: int, train_only_masked: bool = True) -> None:
        super().__init__()
        self._pad_id = padid
        self._mask_id = maskid
        self.train_only_masked = train_only_masked

    def forward(self, student_input_ids: Tensor, teacher_input_ids: Tensor, **kwargs) -> Tensor:
        if self.train_only_masked:
            teacher_input_ids = teacher_input_ids.masked_fill(student_input_ids != self._mask_id, -100)
        else:
            targets = teacher_input_ids.masked_fill(teacher_input_ids == self._pad_id, -100)
        return targets
