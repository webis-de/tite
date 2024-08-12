from typing import Any

import torch
from torch import LongTensor, Tensor
from torch.nn import Module


class Transformation(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwds: Any) -> tuple[dict, dict]:
        raise NotImplementedError


class MLMMaskTokens(Transformation):
    def __init__(self, vocab_size: int, maskid: int, clsid: int, sepid: int, mask_prob: float = 0.3) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._mask_id = maskid
        self._cls_id = clsid
        self._sep_id = sepid
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        probability_matrix = torch.rand(attention_mask.shape, device=input_ids.device)
        mask_mask = mlm_mask & (probability_matrix < 0.8)
        mask_random = mlm_mask & (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        input_ids = torch.where(mask_mask, self._mask_id, input_ids)
        input_ids = torch.where(
            mask_random, torch.randint(self._vocab_size, input_ids.shape, device=input_ids.device), input_ids
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, {"mlm_mask": mlm_mask}


class DeleteTokens(Transformation):
    def __init__(self, padid: int, clsid: int, sepid: int, delete_prob: float = 0.3):
        super().__init__()
        self._pad_id = padid
        self._cls_id = clsid
        self._sep_id = sepid
        self._delete_prob = delete_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        delete_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._delete_prob
        delete_mask = delete_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        num_delete = delete_mask.sum(-1)
        num_tokens = input_ids.shape[1] - num_delete
        new_input_ids = torch.nn.utils.rnn.pad_sequence(
            torch.split(input_ids[~delete_mask], num_tokens.tolist()), batch_first=True, padding_value=self._pad_id
        )
        new_attention_mask = torch.nn.utils.rnn.pad_sequence(
            torch.split(attention_mask[~delete_mask], num_tokens.tolist()), batch_first=True
        )
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}, {}


class InsertRandomTokens(Transformation):
    def __init__(self, vocab_size: int, insert_prob: float):
        super().__init__()
        self._insert_prob = insert_prob
        self._vocab_size = vocab_size

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[dict, dict]:
        shape = attention_mask.shape
        num_non_zero = attention_mask.sum(-1)
        # TODO add random offset to token idcs
        batch_idx, token_idx = attention_mask.nonzero(as_tuple=True)
        # TODO mu = random prob * len, sigma = ?!
        # num_insert_tokens = torch.randn()
        insert_mask = torch.rand(shape, device=input_ids.device) < self._insert_prob
        insert_idx = insert_mask.cumsum(-1)
        num_added = insert_idx[:, -1]
        new_shape = (shape[0], shape[1] + num_added.max().item())
        new_token_idx = token_idx + insert_idx[attention_mask.bool()]
        new_input_ids = torch.full(new_shape, -100, device=input_ids.device)
        new_input_ids[batch_idx, new_token_idx] = input_ids[batch_idx, token_idx]
        new_input_ids
