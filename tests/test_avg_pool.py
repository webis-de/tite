from typing import Tuple

import pytest
import torch

from tite.model import compute_output_shapes
from tite.pool import PackedAvgPool1d, PackedMetaData


@pytest.mark.parametrize("kernel_size, stride, seq_length", [(3, 1, 8), (3, 2, 8), (3, 3, 8)])
def test_masked_avg_pool1d_dimensions(kernel_size: int, stride: int, seq_length: int):
    layer = PackedAvgPool1d(kernel_size, stride, implementation="eager")

    x = torch.randn(2, seq_length, 4)
    mask = torch.ones(2, seq_length, dtype=torch.bool)
    mask[0, -seq_length // 2 :] = False

    output_shapes = compute_output_shapes(seq_length, (kernel_size,), (stride,))

    output, output_mask = layer(x, mask)
    assert output.shape[1] == output_shapes[-1]
    assert ((output != 0).all(-1) == output_mask).all()
    assert output_mask.shape[1] == output_shapes[-1]


def test_masked_avg_pool1d_2_5_4_3_3():
    layer = PackedAvgPool1d(3, 3, implementation="eager")
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    out, out_mask = layer(x, mask)
    assert torch.allclose(
        out,
        torch.tensor(
            [[[4.0, 5.0, 6.0, 7.0], [14.0, 15.0, 16.0, 17.0]], [[24.0, 25.0, 26.0, 27.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
    )
    assert torch.equal(out_mask, torch.tensor([[True, True], [True, False]]))


def test_masked_avg_pool1d_2_5_4_8_1():
    layer = PackedAvgPool1d(8, 1, implementation="eager")
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    out, out_mask = layer(x, mask)
    assert torch.allclose(out, torch.tensor([[[8.0, 9.0, 10.0, 11.0]], [[24.0, 25.0, 26.0, 27.0]]]))
    assert torch.equal(out_mask, torch.tensor([[True], [True]]))


@pytest.mark.parametrize("dim", [1, 4, 8, 16, 64, 768])
@pytest.mark.parametrize("seq_length", [2, 3, 4, 8, 16, 64, 256, 768])
@pytest.mark.parametrize("stride", range(1, 10))
@pytest.mark.parametrize("kernel_size", range(1, 10))
def test_packed_avg_pool1d(
    kernel_size: int, stride: int, seq_length: int, dim: int, dtype: torch.dtype = torch.float32
):
    if stride > kernel_size:
        with pytest.raises(ValueError):
            PackedAvgPool1d(kernel_size, stride)
        return

    eager = PackedAvgPool1d(kernel_size, stride, implementation="eager")
    triton = PackedAvgPool1d(kernel_size, stride, implementation="triton")

    x = torch.randn(2, seq_length, dim, requires_grad=True, device="cuda", dtype=dtype)
    mask = torch.ones(2, seq_length, dtype=torch.bool, device="cuda")
    mask[0, min(-1, -seq_length // 2) :] = False
    seq_lens = mask.sum(-1)
    cu_seq_lens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device="cuda")
    cu_seq_lens[1:] = seq_lens.cumsum(0)

    idcs = mask.nonzero(as_tuple=True)
    eager_x = x[idcs].detach().clone().requires_grad_(True)
    triton_x = x[idcs].detach().clone().requires_grad_(True)

    meta_data = PackedMetaData(seq_lens, cu_seq_lens, int(seq_lens.max().int()), idcs)

    output_eager, _ = eager(eager_x, meta_data)
    output_triton, _ = triton(triton_x, meta_data)

    (output_eager * (torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) / dim)).sum().backward()
    (output_triton * (torch.arange(1, dim + 1, device=x.device, dtype=x.dtype) / dim)).sum().backward()

    atol = 1e-6 if dtype == torch.float32 else 1e-2
    assert torch.allclose(output_eager, output_triton, atol=atol)
    assert eager_x.grad is not None and triton_x.grad is not None
    assert torch.allclose(eager_x.grad, triton_x.grad, atol=atol)
