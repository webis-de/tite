import torch

from .teacher import Teacher


class MLMTeacher(Teacher):
    def __init__(self, padid: int) -> None:
        super().__init__()
        self.pad_id = padid

    def map_targets(self, original_input_ids: torch.Tensor, mlm_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # Everything that is masked out (mlm_mask == True) should be predicted... except for padding tokens.
        # Note that in MaskTokens (transformations.py) does not mask out [CLS] and [SEP] so we don't need to consider
        # them here.
        targets = torch.where(original_input_ids.eq(self.pad_id) | ~mlm_mask, -100, original_input_ids)
        return targets


class MAETeacher(MLMTeacher):

    def __init__(self, padid: int, enhanced: bool) -> None:
        super().__init__(padid)
        self.enhanced = enhanced

    def map_targets(
        self, original_input_ids: torch.Tensor, mlm_mask: torch.Tensor, special_tokens_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if self.enhanced:
            # enhanced decoding, predict every token
            return torch.where(original_input_ids.eq(self.pad_id) | special_tokens_mask, -100, original_input_ids)
        return super().map_targets(original_input_ids, mlm_mask, **kwargs)


class BOWTeacher(Teacher):
    def __init__(self, vocab_size: int, padid: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = padid

    def map_targets(
        self, original_input_ids: torch.Tensor, special_tokens_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        targets = torch.zeros(original_input_ids.shape[0], self.vocab_size, device=original_input_ids.device)
        input_ids = original_input_ids.clone()
        input_ids[special_tokens_mask] = self.pad_id
        targets = targets.scatter(1, input_ids, 1)
        targets[:, self.pad_id] = -100
        return targets
