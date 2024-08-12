from typing import Any

import torch
from torch import LongTensor, Tensor
from torch.nn import Module


class Transformation(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwds: Any) -> tuple[list[dict], list[dict]]:
        raise NotImplementedError


class IdentTransform(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[list[dict], list[dict]]:
        return [{"input_ids": input_ids, "attention_mask": attention_mask}], []


# class SwapTokens(Transformation):
#     def __init__(self, num_swaps: int | float = 1):
#         """Swaps two random (unmasked) tokens. The random swap is repeated `num_swaps` times if `num_swaps` is an
#         integer and `num_swaps * input_len` times if it is a float.

#         Args:
#             num_swaps (int | float, optional): The number of swaps to perform. Defaults to 1.
#         """
#         super().__init__()
#         assert num_swaps > 0
#         assert not isinstance(num_swaps, float) or num_swaps <= 1
#         self._nswaps = num_swaps

#     def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> list[dict]:
#         inputlen = attention_mask.sum(-1)
#         B, L = input_ids.shape
#         nswaps = torch.ceil((torch.ones(B) if isinstance(self._nswaps, int) else inputlen) * self._nswaps).int()
#         idx = torch.repeat_interleave(torch.arange(B), nswaps)
#         randmin = idx * L
#         randmax = idx * L + inputlen[idx] - 1

#         idx1 = (torch.rand(idx.shape) * (randmax - randmin) + randmin).round().int()
#         idx2 = (torch.rand(idx.shape) * (randmax - randmin) + randmin).round().int()
#         flattened = input_ids.flatten()
#         for i1, i2 in zip(idx1, idx2):
#             flattened[[i1, i2],] = flattened[[i2, i1],]
#         # Alternative: This does not only swap tokens though but copies and overrides
#         # flattened[idx1], flattened[idx2] = flattened[idx2], flattened[idx1]
#         return [{"input_ids": input_ids, "attention_mask": attention_mask}]


class MLMMaskTokens(Transformation):
    def __init__(self, vocab_size: int, maskid: int, clsid: int, sepid: int, mask_prob: float = 0.3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self._mask_id = maskid
        self._cls_id = clsid
        self._sep_id = sepid
        self._mask_prob = mask_prob

    def forward(self, input_ids: Tensor, attention_mask: LongTensor, **kwargs) -> tuple[list[dict], list[dict]]:
        mlm_mask = torch.rand(attention_mask.shape, device=input_ids.device) < self._mask_prob
        mlm_mask = mlm_mask.logical_and(input_ids != self._cls_id).logical_and(input_ids != self._sep_id)
        probability_matrix = torch.rand(attention_mask.shape, device=input_ids.device)
        mask_mask = mlm_mask & (probability_matrix < 0.8)
        mask_random = mlm_mask & (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        input_ids = torch.where(mask_mask, self._mask_id, input_ids)
        input_ids = torch.where(
            mask_random, torch.randint(self.vocab_size, input_ids.shape, device=input_ids.device), input_ids
        )
        return [{"input_ids": input_ids, "attention_mask": attention_mask}], [{"mlm_mask": mlm_mask}]


# class HardMLMMaskTokens:
#     def __init__(self, mask_prob: float = 0.3) -> None:
#         self._mask_prob = mask_prob

#     def __call__(
#         self, input_ids: Tensor, attention_mask: LongTensor, **kwargs
#     ) -> list[dict]:
#         attention_mask = torch.logical_and(
#             attention_mask, torch.rand_like(attention_mask) > self._mask_prob
#         )
#         return [{"input_ids": input_ids, "attention_mask": attention_mask}]


# class RandomTransformation(Transformation):
#     def __init__(self, transformations: list[Transformation], numsamples: int) -> None:
#         super().__init__()
#         self._transformations = transformations
#         self._num = numsamples

#     def __call__(self, *args: Any, **kwds: Any) -> list[dict]:
#         return [t for trans in sample(self._transformations, self._num) for t in trans(*args, **kwds)]
