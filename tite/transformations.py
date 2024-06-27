from random import sample
from typing import Any, Callable

import torch
from torch import LongTensor, Tensor

Transformation = Callable[[Any], list[dict | tuple[dict, Any]]]


class SwapTokens:
    def __init__(self, num_swaps: int | float = 1):
        """Swaps two random (unmasked) tokens. The random swap is repeated `num_swaps` times if `num_swaps` is an
        integer and `num_swaps * input_len` times if it is a float.

        Args:
            num_swaps (int | float, optional): The number of swaps to perform. Defaults to 1.
        """
        assert num_swaps > 0
        assert not isinstance(num_swaps, float) or num_swaps <= 1
        self._nswaps = num_swaps

    def __call__(
        self, input_ids: Tensor, attention_mask: LongTensor, **kwargs
    ) -> list[dict]:
        inputlen = attention_mask.sum(-1)
        B, L = input_ids.shape
        nswaps = torch.ceil(
            (torch.ones(B) if isinstance(self._nswaps, int) else inputlen)
            * self._nswaps
        ).int()
        idx = torch.repeat_interleave(torch.arange(B), nswaps)
        randmin = idx * L
        randmax = idx * L + inputlen[idx] - 1

        idx1 = (torch.rand(idx.shape) * (randmax - randmin) + randmin).round().int()
        idx2 = (torch.rand(idx.shape) * (randmax - randmin) + randmin).round().int()
        flattened = input_ids.flatten()
        for i1, i2 in zip(idx1, idx2):
            flattened[[i1, i2],] = flattened[[i2, i1],]
        # Alternative: This does not only swap tokens though but copies and overrides
        # flattened[idx1], flattened[idx2] = flattened[idx2], flattened[idx1]
        return [{"input_ids": input_ids, "attention_mask": attention_mask}]


class MaskTokens:
    def __init__(self, maskid: int, mask_prob: float = 0.3) -> None:
        self._maskid = maskid
        self._mask_prob = mask_prob

    def __call__(
        self, input_ids: Tensor, attention_mask: LongTensor, **kwargs
    ) -> list[dict]:
        mask = torch.rand_like(attention_mask) < self._mask_prob
        input_ids = torch.masked_fill(input_ids, mask, self._maskid)
        return [{"input_ids": input_ids, "attention_mask": attention_mask}]


"""
class HardMaskTokens:
    def __init__(self, mask_prob: float = 0.3) -> None:
        self._mask_prob = mask_prob

    def __call__(
        self, input_ids: Tensor, attention_mask: LongTensor, **kwargs
    ) -> list[dict]:
        attention_mask = torch.logical_and(
            attention_mask, torch.rand_like(attention_mask) > self._mask_prob
        )
        return [{"input_ids": input_ids, "attention_mask": attention_mask}]
"""


class RandomTransformation:

    def __init__(self, transformations: list[Transformation], numsamples: int) -> None:
        self._transformations = transformations
        self._num = numsamples

    def __call__(self, *args: Any, **kwds: Any) -> list[dict]:
        return [
            t
            for trans in sample(self._transformations, self._num)
            for t in trans(*args, **kwds)
        ]
