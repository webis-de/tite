# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""Helper functions for padding and unpadding batches.

These functions are used extensively throughout the Mosaic BERT implementation
in `bert_layers.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from einops import rearrange, repeat

if TYPE_CHECKING:
    from .pool import PackedMetaData


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, idcs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idcs)
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]  # type: ignore
        second_dim = other_shape.numel()  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(x, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(idcs, "z -> z d", d=second_dim),  # (idcs,) -> (idcs, second_dim)
        ).reshape(
            -1, *other_shape
        )  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (idcs,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_x = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_x[idcs] = grad_output
        grad_x.scatter_(0, repeat(idcs, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_x.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, idcs: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(idcs)
        assert idcs.ndim == 1
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[idcs] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (idcs,) = ctx.saved_tensors
        grad_values = grad_output[idcs]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(x: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
    rearranged = rearrange(x, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, packed_meta_data.idcs)  # type: ignore


def pad_input(hidden_states: torch.Tensor, packed_meta_data: PackedMetaData) -> torch.Tensor:
    output = index_put_first_axis(
        hidden_states, packed_meta_data.idcs, len(packed_meta_data.seq_lens) * packed_meta_data.max_seq_len
    )
    return rearrange(output, "(b s) ... -> b s ...", b=len(packed_meta_data.seq_lens))  # type: ignore
