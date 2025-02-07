from typing import Tuple

import pytest
import torch

from tite.model import compute_output_shapes
from tite.pool import PackedAvgPool1d, PackedMetaData


class ReferenceMaskedAvgPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] == 1:
            return x, mask

        batch_size, _, dim = x.shape

        seq_lens = mask.sum(-1)
        if self.kernel_size > x.shape[-2]:
            padding = self.kernel_size - x.shape[-2]
        else:
            padding = (self.kernel_size - seq_lens - self.stride) % self.stride

        new_seq_lens = seq_lens + padding
        pad_x = torch.zeros(batch_size, int(new_seq_lens.max().item()), dim, device=x.device, dtype=x.dtype)
        pad_mask = torch.zeros(batch_size, int(new_seq_lens.max().item()), device=mask.device, dtype=mask.dtype)
        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx]
            new_seq_len = new_seq_lens[batch_idx]
            pad_x[batch_idx, :seq_len] = x[batch_idx, :seq_len]
            pad_mask[batch_idx, :new_seq_len] = True

        x_blocks = pad_x.unfold(-2, self.kernel_size, self.stride)
        mask_blocks = pad_mask.unfold(-1, self.kernel_size, self.stride).unsqueeze(-2)
        y = x_blocks.mean(-1)
        mask_blocks[:, 0, 0, :] = 1
        y_mask = mask_blocks.amin(-1).squeeze(-1)
        y = y.masked_fill(~y_mask[..., None].expand_as(y), 0)
        return y, y_mask


@pytest.mark.parametrize("kernel_size, stride, seq_length", [(3, 1, 8), (3, 2, 8), (3, 3, 8)])
def test_masked_avg_pool1d_dimensions(kernel_size: int, stride: int, seq_length: int):
    layer = ReferenceMaskedAvgPool1d(kernel_size, stride)

    x = torch.randn(2, seq_length, 4)
    mask = torch.ones(2, seq_length, dtype=torch.bool)
    mask[0, -seq_length // 2 :] = False

    output_shapes = compute_output_shapes(seq_length, (kernel_size,), (stride,))

    output, output_mask = layer(x, mask)
    assert output.shape[1] == output_shapes[-1]
    assert ((output != 0).all(-1) == output_mask).all()
    assert output_mask.shape[1] == output_shapes[-1]


def test_masked_avg_pool1d_2_5_4_3_3():
    layer = ReferenceMaskedAvgPool1d(3, 3)
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
    layer = ReferenceMaskedAvgPool1d(8, 1)
    x = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    out, out_mask = layer(x, mask)
    assert torch.allclose(out, torch.tensor([[[8.0, 9.0, 10.0, 11.0]], [[24.0, 25.0, 26.0, 27.0]]]))
    assert torch.equal(out_mask, torch.tensor([[True], [True]]))


@pytest.mark.parametrize("kernel_size", range(1, 5))
@pytest.mark.parametrize("stride", range(1, 5))
@pytest.mark.parametrize("seq_length", [2, 3, 4, 8, 16, 64, 256, 768])
@pytest.mark.parametrize("k", [1, 4, 8, 16, 64, 768])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"]
)
def test_packed_avg_pool1d(kernel_size: int, stride: int, seq_length: int, k: int, dtype: torch.dtype):
    masked = ReferenceMaskedAvgPool1d(kernel_size, stride)
    packed = PackedAvgPool1d(kernel_size, stride)

    x = torch.randn(2, seq_length, k, requires_grad=True, device="cuda", dtype=dtype)
    mask = torch.ones(2, seq_length, dtype=torch.bool, device="cuda")
    mask[0, min(-1, -seq_length // 2) :] = False
    seq_lens = mask.sum(-1)
    cu_seq_lens = torch.zeros(mask.shape[0] + 1, dtype=torch.int64, device="cuda")
    cu_seq_lens[1:] = seq_lens.cumsum(0)

    idcs = mask.nonzero(as_tuple=True)
    packed_x = x[idcs].detach().clone().requires_grad_(True)

    meta_data = PackedMetaData(seq_lens, cu_seq_lens, int(seq_lens.max().int()))

    output_masked, out_mask = masked(x, mask)
    output_packed, _ = packed(packed_x, meta_data)

    (output_masked * (torch.arange(1, k + 1, device=x.device, dtype=x.dtype) / k)).sum().backward()
    (output_packed * (torch.arange(1, k + 1, device=x.device, dtype=x.dtype) / k)).sum().backward()

    atol = 1e-6 if dtype == torch.float32 else 1e-2
    assert torch.allclose(output_masked[out_mask], output_packed, atol=atol)
    assert x.grad is not None and packed_x.grad is not None
    assert torch.allclose(x.grad[mask], packed_x.grad, atol=atol)
