from typing import Tuple

import torch
from einops import rearrange, repeat

# unpad from
# https://github.com/mosaicml/examples/blob/daddaeff7a535273ff984b0134da4839c70a45b3/examples/benchmarks/bert/src/bert_padding.py


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # type: ignore
        second_dim = other_shape.numel()  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, "z -> z d", d=second_dim),  # (indices,) -> (indices, second_dim)
        ).reshape(
            -1, *other_shape
        )  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Like unpad_input, but only return the unpadded first tensor.

    Save a small amount of overhead.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
    """
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, indices)  # type: ignore


def repad(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Add padding to sequences.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        batch: int batch_size
        seqlen: int max sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)  # type: ignore
