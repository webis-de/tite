import torch
from torch import Tensor
from torch.nn import Module


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        return input_ids


class OrderTeacher(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, student_batch_idcs: tuple[int], **kwargs) -> Tensor:
        block_size = torch.bincount(torch.tensor(student_batch_idcs, device=input_ids.device))
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.arange(bs, device=input_ids.device) for bs in block_size], batch_first=True, padding_value=-100
        )
        return targets


class MLMTeacher(Module):
    def __init__(self, padid: int) -> None:
        super().__init__()
        self.pad_id = padid

    def forward(self, original_input_ids: Tensor, mlm_mask: Tensor, **kwargs) -> Tensor:
        # Everything that is masked out (mlm_mask == True) should be predicted... except for padding tokens.
        # Note that in MaskTokens (transformations.py) does not mask out [CLS] and [SEP] so we don't need to consider
        # them here.
        targets = torch.where(original_input_ids.eq(self.pad_id) | ~mlm_mask, -100, original_input_ids)
        return targets


class MAETeacher(MLMTeacher):

    def forward(self, original_input_ids: Tensor, mlm_mask: Tensor, special_tokens_mask: Tensor, **kwargs) -> Tensor:
        if not mlm_mask.any():
            # enhanced decoding, predict every token
            return torch.where(original_input_ids.eq(self.pad_id) | special_tokens_mask, -100, original_input_ids)
        return super().forward(original_input_ids, mlm_mask, **kwargs)


class CopyStudent(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        return input_ids
